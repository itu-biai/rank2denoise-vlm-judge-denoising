# scripts/s2s_expert.py
"""Dataset-level Self2Self expert: training (GPU) and validation / test outputs. GT is never read.

Training (`train`): one partial-convolution U-Net with dropout (`expert_methods.S2SPartialConvUNet`) on all noisy slices
of the training split (899, sorted file order). Each step draws a Bernoulli mask M (keep probability 0.7; M = 1 visible,
M = 0 hidden); the loss is the MSE on the hidden pixels against the noisy input. Adam, lr 1e-4, 30 epochs, no scheduler,
no AMP, no gradient clipping; seed 42. Checkpoint selection without GT: after every epoch, the same loss on the 182 noisy
validation slices with one fixed mask per slice (seed 4242 + slice index, dropout on); the lowest value is kept.
Epoch-level resume (`last_checkpoint.pth`).
Batch size: 8 (the value of the paper run); it is stored in the run's `done.json` (`batch_size`).

Inference (`infer`): 50 stochastic passes per slice (random flip -> Bernoulli mask -> dropout on -> flip back), mean,
clipped to [0, 1] (`expert_methods.s2s_ensemble`), batch 8, seed 4242 + batch start. PyTorch default precision flags.

Outputs: `<R2D_RESULTS>/experts/s2s_train/{best_selection_checkpoint.pth, last_checkpoint.pth, training_log.csv, done.json}` and
`<R2D_RESULTS>/experts/outputs/s2s/{val,test}.npy`.
Usage (cwd = repository root):
  python -m scripts.s2s_expert train [batch_size]      (data from R2D_DATA; batch size 8 by default)
  python -m scripts.s2s_expert infer
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts import expert_methods as em
from scripts.expert_outputs import (
    OUT_ROOT,
    noisy_inputs,
    output_done,
    save_output,
    split_files,
)
from src.paths import DATA_ROOT

TRAIN_DIR = OUT_ROOT / "s2s_train"
PAPER_BATCH_SIZE = 8  # batch size of the S2S run of the paper
KEY = "s2s"
LOG_COLUMNS = (
    "unit",
    "loss_s2s",
    "selection_metric",
    "sec_per_unit",
    "gpu_mem_gb",
    "lr",
    "wall_clock_s",
)


@dataclass(frozen=True)
class S2SConfig:
    """Training, selection and inference settings of the dataset-level Self2Self expert."""

    noisy_key: str = "noisy_clipped_100k"
    expected_train_size: int = 899
    expected_val_size: int = 182
    seed: int = 42
    learning_rate: float = 1e-4
    bernoulli_keep_prob: float = em.S2S_KEEP_PROB
    dropout_p: float = em.S2S_DROPOUT
    inference_samples: int = em.S2S_SAMPLES
    ds_epochs: int = 30
    ds_num_workers: int = 2
    selection_seed: int = em.S2S_SELECTION_SEED
    selection_dropout_on: bool = True
    eval_batch: int = em.S2S_EVAL_BATCH


DEFAULT_S2S_CONFIG = S2SConfig()


# ----------------------------------------------------------------------------- data (noisy only)


def list_split_files(root: Path, split: str, expected: int) -> list[Path]:
    """Sorted `.npz` files of one split; an empty or differently sized split raises early."""
    files = sorted((root / split).glob("*.npz"))
    if not files:
        raise RuntimeError(f"empty split: {root}/{split}")
    if len(files) != expected:
        raise RuntimeError(f"split size {split} = {len(files)}, expected {expected}")
    return files


def load_noisy_image(path: Path, config: S2SConfig, max_retries: int = 3, retry_delay: float = 1.0) -> np.ndarray:
    """Only the noisy key of one `.npz` is read; (512, 512) float32 in [0, 1]."""
    for attempt in range(max_retries):
        try:
            with np.load(path) as npz_file:
                if config.noisy_key not in npz_file.files:
                    raise KeyError(f"{config.noisy_key} missing: {path.name}")
                noisy = npz_file[config.noisy_key].astype(np.float32)
            break
        except (
            OSError,
            zipfile.BadZipFile,
        ) as exc:  # BadZipFile is not an OSError subclass
            if attempt == max_retries - 1:
                raise OSError(f"could not read {path} ({max_retries} attempts)") from exc
            time.sleep(retry_delay)
    if noisy.shape != (512, 512):
        raise ValueError(f"expected shape (512, 512), found {noisy.shape} ({path.name})")
    if float(noisy.min()) < -1e-6 or float(noisy.max()) > 1.0 + 1e-6:
        raise ValueError(f"image outside [0, 1]: {path.name}")
    return noisy


class NoisyOnlyDataset(Dataset):
    """Noisy image + index; GT is never opened."""

    def __init__(self, file_paths: list[Path], config: S2SConfig) -> None:
        self.file_paths = [Path(p) for p in file_paths]
        self.config = config

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(load_noisy_image(self.file_paths[idx], self.config)).unsqueeze(0), idx


# ----------------------------------------------------------------------------- mask and loss


def bernoulli_visible_mask(
    shape: torch.Size,
    keep_prob: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """0/1 mask, P(M = 1) = keep_prob (M = 1 visible). Not `nn.Dropout`, which rescales the kept values."""
    rand = torch.rand(shape, device=device, generator=generator)
    return (rand < keep_prob).to(torch.float32)


def s2s_masked_mse(pred: torch.Tensor, target: torch.Tensor, visible_mask: torch.Tensor) -> torch.Tensor:
    """MSE on the hidden pixels only (M = 0), divided by the number of hidden pixels."""
    hidden = 1.0 - visible_mask
    denom = hidden.sum()
    if float(denom) == 0.0:
        raise ValueError("no hidden pixel: degenerate mask")
    return ((pred - target) ** 2 * hidden).sum() / denom


# ----------------------------------------------------------------------------- seeds, checkpoints, log


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    """Current states of all random generators."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _as_cpu_byte(t: torch.Tensor) -> torch.Tensor:
    """`torch.load(map_location=cuda)` moves RNG states to the GPU; `set_rng_state` needs a CPU ByteTensor."""
    return t.detach().cpu().to(torch.uint8)


def restore_rng(state: dict) -> None:
    """Restore the random generator states saved by `rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_cpu_byte(state["torch"]))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_cpu_byte(c) for c in state["cuda"]])


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Save to a temporary file, then rename it to the target path."""
    tmp = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _append_log_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8") as fh:
        if not exists:
            fh.write(",".join(LOG_COLUMNS) + "\n")
        fh.write(",".join(str(row.get(c, "")) for c in LOG_COLUMNS) + "\n")
        fh.flush()


# ----------------------------------------------------------------------------- training


@torch.no_grad()
def selection_metric(model: nn.Module, val_files: list[Path], config: S2SConfig, device: torch.device) -> float:
    """GT-free selection: masked loss on noisy validation slices, one fixed mask per slice (seed + slice index)."""
    was_training = model.training
    model.train(config.selection_dropout_on)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(config.selection_seed)
    try:
        total, n = 0.0, 0
        for idx, path in enumerate(val_files):
            noisy = torch.from_numpy(load_noisy_image(Path(path), config))
            noisy = noisy.view(1, 1, *noisy.shape).to(device)
            gen = torch.Generator(device=device)
            gen.manual_seed(config.selection_seed + idx)
            m = bernoulli_visible_mask(noisy.shape, config.bernoulli_keep_prob, device, gen)
            total += float(s2s_masked_mse(model(noisy * m, m), noisy, m))
            n += 1
        return total / max(n, 1)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        model.train(was_training)


def train(data_root: Path, batch_size: int, config: S2SConfig = DEFAULT_S2S_CONFIG) -> dict:
    """Train the S2S network with epoch-level resume; returns the content of done.json."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_files = list_split_files(data_root, "train", config.expected_train_size)
    val_files = list_split_files(data_root, "val", config.expected_val_size)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    last, best, log, done = (
        TRAIN_DIR / "last_checkpoint.pth",
        TRAIN_DIR / "best_selection_checkpoint.pth",
        TRAIN_DIR / "training_log.csv",
        TRAIN_DIR / "done.json",
    )
    if done.exists():
        print("S2S training already done.", flush=True)
        return json.loads(done.read_text(encoding="utf-8"))
    for partial in TRAIN_DIR.glob("*.partial"):
        partial.unlink()
    loader = DataLoader(
        NoisyOnlyDataset(train_files, config),
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.ds_num_workers,
        drop_last=True,
        persistent_workers=config.ds_num_workers > 0,
    )
    set_seed(config.seed)
    model = em.S2SPartialConvUNet(channels=1, dropout_p=config.dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_epoch, best_sel, best_epoch = 0, float("inf"), -1
    if last.exists():
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch, best_sel, best_epoch = (
            int(ckpt["epoch"]),
            float(ckpt["best_selection"]),
            int(ckpt["best_epoch"]),
        )
        restore_rng(ckpt["rng_state"])
        print(
            f"[resume] epoch {start_epoch} (best {best_sel:.6g} @ {best_epoch})",
            flush=True,
        )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_wall = time.time()
    for epoch in range(start_epoch, config.ds_epochs):
        model.train()
        total, n_steps, t_epoch = 0.0, 0, time.time()
        for noisy, _ in loader:
            noisy = noisy.to(device, non_blocking=True)
            m = bernoulli_visible_mask(noisy.shape, config.bernoulli_keep_prob, device)
            loss = s2s_masked_mse(model(noisy * m, m), noisy, m)  # target: the noisy input
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            n_steps += 1
        sel = selection_metric(model, val_files, config, device)
        _append_log_row(
            log,
            {
                "unit": epoch + 1,
                "loss_s2s": total / max(n_steps, 1),
                "selection_metric": sel,
                "sec_per_unit": time.time() - t_epoch,
                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
                "lr": config.learning_rate,
                "wall_clock_s": time.time() - t_wall,
            },
        )
        payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_selection": best_sel,
            "best_epoch": best_epoch,
            "rng_state": rng_state(),
            "config": asdict(config),
        }
        if sel < best_sel:  # GT-free selection
            best_sel, best_epoch = sel, epoch + 1
            payload["best_selection"], payload["best_epoch"] = best_sel, best_epoch
            staged = TRAIN_DIR / f"best.epoch{epoch + 1:03d}.staged"
            atomic_torch_save(payload, staged)
            atomic_torch_save(payload, last)
            os.replace(staged, best)
        else:
            atomic_torch_save(payload, last)
        print(
            f"  epoch {epoch + 1} · loss {total / max(n_steps, 1):.6f} · selection {sel:.6g}",
            flush=True,
        )
    summary = {
        "epochs": config.ds_epochs,
        "batch_size": batch_size,
        "train_images": len(train_files),
        "val_images": len(val_files),
        "best_selection": best_sel,
        "best_epoch": best_epoch,
        "total_train_seconds": time.time() - t_wall,
        "gt_used": False,
    }
    done.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ----------------------------------------------------------------------------- inference


def load_s2s(device: torch.device) -> nn.Module:
    """Load the selected S2S checkpoint; raises if its settings differ from the inference settings."""
    checkpoint = torch.load(
        TRAIN_DIR / "best_selection_checkpoint.pth",
        map_location=device,
        weights_only=False,
    )
    config = checkpoint["config"]
    expected = {
        "bernoulli_keep_prob": em.S2S_KEEP_PROB,
        "dropout_p": em.S2S_DROPOUT,
        "inference_samples": em.S2S_SAMPLES,
        "selection_seed": em.S2S_SELECTION_SEED,
        "eval_batch": em.S2S_EVAL_BATCH,
    }
    different = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if different:
        raise RuntimeError(f"S2S checkpoint settings differ from the inference settings: {different}")
    model = em.S2SPartialConvUNet(channels=1, dropout_p=em.S2S_DROPOUT).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def infer() -> None:
    """Write the S2S outputs of the validation and test slices."""
    if output_done(KEY):
        print(f"{KEY}: done, skipped", flush=True)
        return
    torch.backends.cudnn.allow_tf32 = True  # PyTorch defaults
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = split_files()
    model = load_s2s(device)
    for split in ("val", "test"):
        ids, noisy = noisy_inputs(split, splits)
        parts = []
        for start in range(0, len(noisy), em.S2S_EVAL_BATCH):
            batch = torch.from_numpy(noisy[start : start + em.S2S_EVAL_BATCH]).unsqueeze(1).to(device)
            prediction = em.s2s_ensemble(model, batch, seed=em.S2S_SELECTION_SEED + start).clamp(0.0, 1.0)
            parts.append(prediction[:, 0].float().cpu().numpy())
        save_output(KEY, split, np.concatenate(parts), ids)
        print(f"{KEY} {split}: {len(ids)} slices", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:
        train(DATA_ROOT, int(sys.argv[2]) if len(sys.argv) > 2 else PAPER_BATCH_SIZE)
    elif sys.argv[1:2] == ["infer"]:
        infer()
    else:
        raise SystemExit(__doc__)
