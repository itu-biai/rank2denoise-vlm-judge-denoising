# scripts/train_experts.py
"""Train the N2V, B2U and R2R experts with the shared U-Net and write their validation / test outputs (GPU). GT is never read.

Same as the N2S setup (`src/train.py train_run`): data split (`load_splits`, split_seed 42, held_out 0.3), U-Net
(`src/unet.py`), AdamW lr 1e-4, batch 8, 30 epochs, ReduceLROnPlateau(min, 0.5, patience 2) on the validation
selection loss, best checkpoint = lowest validation selection loss, `set_seed`, epoch-level resume.
Differences:
  * noisy training slices are read into memory once; only `noisy_clipped_100k` is opened (the GT key is never read).
  * selection loss = the method's own loss on noisy validation inputs, with fixed randomness (seed 2027); epoch-dependent
    coefficients at their final value (B2U lambda = 20).
  * memory: batch 8 is kept; if needed, gradients are accumulated over micro-batches (the loss is a batch mean, so the
    gradient is the same). On CUDA OOM the micro-batch halves.

Outputs: `<R2D_RESULTS>/experts/train/<method>_seed<s>/{best_model.pth, last.pth, training_log.csv}` and
`<R2D_RESULTS>/experts/outputs/<method>_seed<s>/{val,test}.npy`.
Usage (cwd = repository root):  python -m scripts.train_experts --methods all --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts import expert_methods as em
from scripts.expert_outputs import (
    OUT_ROOT,
    noisy_inputs,
    output_done,
    save_output,
    split_files,
)
from scripts.noise_fields_seeds import seed42_fields
from scripts.stage1_search import seed_fields
from src.data import _load_npz_arrays
from src.unet import UNet

TRAIN_ROOT = OUT_ROOT / "train"
SELECTION_SEED = 2027
SEEDS = (42, 123, 456)
DEFAULT_MICRO = {"n2v": 8, "b2u": 2, "r2r": 8}  # initial micro-batch; halves on OOM
INFER_BATCH = {"n2v": 8, "b2u": 8, "r2r": 4}


@dataclass(frozen=True)
class ExpertTrainConfig:
    """Training settings of one expert run (method and seed)."""

    method: str
    seed: int
    input_key: str = "noisy_clipped_100k"
    batch_size: int = 8  # same as N2S (src/config.py)
    lr: float = 1e-4  # same as N2S
    epochs: int = 30  # same as N2S
    plateau_factor: float = 0.5  # same as N2S train_run
    plateau_patience: int = 2  # same as N2S train_run
    read_threads: int = 8

    @property
    def key(self) -> str:
        """Name of the run: method and seed."""
        return f"{self.method}_seed{self.seed}"

    @property
    def run_dir(self) -> Path:
        """Training folder of the run."""
        return TRAIN_ROOT / self.key


def set_seed(seed: int) -> None:  # same as train_n2s_seeds.set_seed
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _atomic_save(payload: dict, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_train_noisy(files: list[Path], key: str, threads: int) -> torch.Tensor:
    """(647, 1, 512, 512) float32; only the noisy key is read."""
    with ThreadPoolExecutor(threads) as pool:
        arrays = list(
            pool.map(
                lambda path: _load_npz_arrays(path, (key,))[key].astype(np.float32),
                files,
            )
        )
    return torch.from_numpy(np.stack(arrays)).unsqueeze(1)


def noise_bank_for(seed: int, stems: list[str], key: str) -> em.NoiseBank:
    """N2S noise fields of the same seed (`<R2D_RESULTS>/noise_fields/n2s_seed<s>/`)."""
    fields = seed42_fields(stems, key) if seed == 42 else seed_fields(seed, stems, key)
    return em.NoiseBank(fields.float())


class LossFn:
    """Training / selection loss of a method. `selection=True` -> epoch-dependent coefficient at its final value, fixed randomness."""

    def __init__(self, method: str, bank: em.NoiseBank | None, total_iters: int) -> None:
        self.method, self.bank, self.total_iters = method, bank, total_iters

    def __call__(
        self,
        model: torch.nn.Module,
        noisy: torch.Tensor,
        index: torch.Tensor | None,
        generator: torch.Generator,
        epoch: int,
        epochs: int,
        iteration: int,
        selection: bool,
    ) -> torch.Tensor:
        m = self.method
        if m == "n2v":
            return em.n2v_loss(model, noisy, generator)
        if m == "b2u":
            lam = em.B2U_LAMBDA_END if selection else em.b2u_lambda((epoch + 1) / epochs)
            return em.b2u_loss(model, noisy, lam)
        if m == "r2r":
            noise = self.bank.sample(len(noisy), None if index is None else index.cpu(), generator).to(noisy.device)
            return em.r2r_loss(model, noisy, noise)
        raise ValueError(m)


def _batch_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    noisy: torch.Tensor,
    index: torch.Tensor,
    generator: torch.Generator,
    micro: int,
    epoch: int,
    epochs: int,
    iteration: int,
    device: torch.device,
) -> tuple[float, int]:
    """One batch: accumulate gradients over micro-batches, then one optimizer step. On OOM the micro-batch halves and the batch restarts."""
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            total = 0.0
            for start in range(0, len(noisy), micro):
                part = noisy[start : start + micro].to(device, non_blocking=True)
                loss = loss_fn(
                    model,
                    part,
                    index[start : start + micro],
                    generator,
                    epoch,
                    epochs,
                    iteration,
                    False,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{loss_fn.method}: loss is not finite (epoch {epoch}, step {iteration}).")
                (loss * len(part) / len(noisy)).backward()
                total += float(loss) * len(part) / len(noisy)
            optimizer.step()
            return total, micro
        except torch.cuda.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro == 1:
                raise RuntimeError(f"{loss_fn.method}: CUDA OOM, micro-batch {micro}.")
            micro //= 2
            print(f"  CUDA OOM → micro-batch {micro}", flush=True)


@torch.no_grad()
def selection_loss(
    model: nn.Module,
    loss_fn: LossFn,
    val_noisy: torch.Tensor,
    micro: int,
    epochs: int,
    device: torch.device,
) -> float:
    """The method's own loss on noisy validation inputs, fixed seed (comparable across epochs)."""
    model.eval()
    generator = torch.Generator().manual_seed(SELECTION_SEED)
    total = 0.0
    for start in range(0, len(val_noisy), micro):
        part = val_noisy[start : start + micro].to(device)
        total += float(
            loss_fn(
                model,
                part,
                None,
                generator,
                epochs - 1,
                epochs,
                loss_fn.total_iters,
                True,
            )
        ) * len(part)
    model.train()
    return total / len(val_noisy)


def train_one(
    config: ExpertTrainConfig,
    train_noisy: torch.Tensor,
    val_noisy: torch.Tensor,
    bank: em.NoiseBank | None,
    device: torch.device,
) -> None:
    """Train one expert with epoch-level resume and keep the best epoch."""
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_dir / "best_model.pth", run_dir / "last.pth"
    steps_per_epoch = math.ceil(len(train_noisy) / config.batch_size)
    loss_fn = LossFn(config.method, bank, total_iters=steps_per_epoch * config.epochs)
    set_seed(config.seed)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.plateau_factor,
        patience=config.plateau_patience,
    )
    generator = torch.Generator().manual_seed(config.seed)  # mask / noise draws of the method
    order = torch.Generator().manual_seed(config.seed + 1)  # batch order
    history, best, micro, start_epoch = (
        [],
        float("inf"),
        DEFAULT_MICRO[config.method],
        0,
    )
    fingerprint = {
        "method": config.method,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "epochs": config.epochs,
        "n_train": len(train_noisy),
    }
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise RuntimeError(f"{last_path} comes from another configuration: {state['fingerprint']} ≠ {fingerprint}.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        order.set_state(state["order"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, micro = state["history"], state["micro"]
        start_epoch = len(history)
        best = min(row["val_selection_loss"] for row in history)
        best_epoch = min(history, key=lambda row: row["val_selection_loss"])["epoch"]
        saved_epoch = torch.load(best_path, map_location="cpu")["epoch"] if best_path.exists() else None
        if saved_epoch != best_epoch:  # same rule as N2S train_run
            if best_epoch != start_epoch - 1:
                raise RuntimeError(f"{best_path} epoch {saved_epoch}, history says {best_epoch}: inconsistent.")
            _atomic_save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "val_selection_loss": best,
                },
                best_path,
            )
        print(
            f"[resume] {config.key}: epoch {start_epoch}/{config.epochs} (best epoch {best_epoch})",
            flush=True,
        )
    model.train()
    for epoch in range(start_epoch, config.epochs):
        began, losses = time.time(), []
        permutation = torch.randperm(len(train_noisy), generator=order)
        for step in range(steps_per_epoch):
            index = permutation[step * config.batch_size : (step + 1) * config.batch_size]
            loss, micro = _batch_step(
                model,
                optimizer,
                loss_fn,
                train_noisy[index],
                index,
                generator,
                micro,
                epoch,
                config.epochs,
                epoch * steps_per_epoch + step,
                device,
            )
            losses.append(loss)
        val_loss = selection_loss(model, loss_fn, val_noisy, max(1, micro), config.epochs, device)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_selection_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "micro_batch": micro,
                "seconds": round(time.time() - began, 1),
                "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available()
                else 0.0,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "training_log.csv", index=False)
        # last.pth first (as in train_run): if the run stops in between, the best model can be rewritten from last.pth
        _atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "fingerprint": fingerprint,
                "generator": generator.get_state(),
                "order": order.get_state(),
                "torch_rng": torch.get_rng_state(),
                "micro": micro,
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            last_path,
        )
        if val_loss < best:
            best = val_loss
            _atomic_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_selection_loss": val_loss,
                },
                best_path,
            )
        print(
            f"  {config.key} epoch {epoch + 1}/{config.epochs}: train {history[-1]['train_loss']:.6f} · "
            f"val {val_loss:.6f} · {history[-1]['seconds']:.0f} s · micro {micro}",
            flush=True,
        )
    if not best_path.exists():
        raise RuntimeError(f"{best_path} missing.")
    saved = torch.load(best_path, map_location="cpu")
    best_epoch = min(history, key=lambda row: row["val_selection_loss"])["epoch"]
    if saved["epoch"] != best_epoch:
        raise RuntimeError(f"{best_path} epoch {saved['epoch']}, history says {best_epoch}.")


@torch.no_grad()
def infer_one(
    config: ExpertTrainConfig,
    splits: dict[str, list[Path]],
    bank: em.NoiseBank | None,
    device: torch.device,
) -> None:
    """Write the validation and test outputs of one trained expert."""
    if output_done(config.key):
        return
    model = UNet().to(device)
    model.load_state_dict(torch.load(config.run_dir / "best_model.pth", map_location="cpu")["model"])
    model.eval()
    for split in ("val", "test"):
        ids, noisy = noisy_inputs(split, splits)
        generator = torch.Generator().manual_seed(
            SELECTION_SEED + config.seed
        )  # random draws of the R2R re-corruption, reproducible
        outputs, batch = [], INFER_BATCH[config.method]
        for start in range(0, len(noisy), batch):
            part = torch.from_numpy(noisy[start : start + batch]).unsqueeze(1).to(device)
            output = em.infer_trained(config.method, model, part, generator, bank)
            outputs.append(output[:, 0].float().cpu().numpy())
        save_output(config.key, split, np.concatenate(outputs), ids)
        print(f"  {config.key}: {split} output written", flush=True)


def main(methods: list[str], seeds: list[int]) -> None:
    """Train the requested experts and seeds, then write their outputs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False  # as in N2S training (PyTorch default); cudnn default unchanged
    splits = split_files()
    stems = [path.stem for path in splits["train_noisy"]]
    base = ExpertTrainConfig(method=methods[0], seed=seeds[0])
    todo = [ExpertTrainConfig(method=m, seed=s) for s in seeds for m in methods]
    if all(output_done(c.key) for c in todo):
        print("outputs of all trained experts are ready.", flush=True)
        return
    print(
        f"reading noisy training slices: {len(stems)} slices (only {base.input_key})",
        flush=True,
    )
    train_noisy = load_train_noisy(splits["train_noisy"], base.input_key, base.read_threads)
    val_noisy = torch.from_numpy(noisy_inputs("val", splits)[1]).unsqueeze(1)
    for seed in seeds:
        bank = noise_bank_for(seed, stems, base.input_key) if "r2r" in methods else None
        for method in methods:
            config = ExpertTrainConfig(method=method, seed=seed)
            if output_done(config.key):
                print(f"{config.key}: done, skipped", flush=True)
                continue
            began = time.time()
            print(f"=== {config.key} ===", flush=True)
            train_one(
                config,
                train_noisy,
                val_noisy,
                bank if method == "r2r" else None,
                device,
            )
            infer_one(config, splits, bank if method == "r2r" else None, device)
            print(f"{config.key}: {(time.time() - began) / 60:.1f} min", flush=True)
        del bank


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    chosen = list(em.TRAINED_METHODS) if args.methods == ["all"] else args.methods
    unknown = set(chosen) - set(em.TRAINED_METHODS)
    if unknown or set(args.seeds) - set(SEEDS):
        raise SystemExit(f"unknown method / seed: {unknown or args.seeds}")
    main(chosen, args.seeds)
