# src/train.py
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import ExperimentConfig
from src.data import NpzPairDataset, load_splits
from src.masking import checkerboard_mask, masked_input, masked_recon_loss
from src.metrics import LpipsMetric, evaluate_model, save_per_slice_metrics
from src.ne2ne import ne2ne_loss
from src.unet import UNet

# Epoch-level resume. An uninterrupted run gives the same training_log.csv / best_model.pth / metrics.csv.
# Path fields and num_workers do not affect the result, so they are not part of the resume check.
RESUME_NAME: str = "last.pth"
RESUME_IGNORED_FIELDS: tuple[str, ...] = ("data_root", "output_dir", "num_workers")


def _resume_fingerprint(config: ExperimentConfig) -> dict[str, object]:
    """Configuration fields that must match when a run is resumed."""
    return {key: value for key, value in asdict(config).items() if key not in RESUME_IGNORED_FIELDS}


def _atomic_save(payload: dict, path: Path) -> None:
    """Write to a temporary file, then replace the target, so an interrupted write keeps the old file."""
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _best_from_history(history: list[dict]) -> tuple[float, int | None]:
    """Replay the selection rule (`val_loss < best`) over the training history."""
    best, best_epoch = float("inf"), None
    for row in history:
        if row["val_selection_loss"] < best:
            best, best_epoch = row["val_selection_loss"], row["epoch"]
    return best, best_epoch


def jinv_loss(model: nn.Module, noisy: torch.Tensor, phase: int) -> torch.Tensor:
    """Noise2Self loss: masked MSE on the pixels hidden by one checkerboard phase."""
    mask = checkerboard_mask(noisy.size(-2), noisy.size(-1), phase, noisy.device)
    prediction = model(masked_input(noisy, mask))
    return masked_recon_loss(prediction, noisy, mask)


def selection_loss(model: nn.Module, loader: DataLoader, config: ExperimentConfig, device: torch.device) -> float:
    """Mean validation loss of the run's method, used to select the best epoch."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for noisy, clean, _ in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            if config.method == "supervised":
                total += F.mse_loss(model(noisy), clean).item()
            elif config.method == "ne2ne":
                total += ne2ne_loss(model, noisy, gamma=2.0).item()
            else:
                total += sum(jinv_loss(model, noisy, phase).item() for phase in (0, 1)) / 2
    return total / len(loader)


def train_run(config: ExperimentConfig) -> pd.DataFrame:
    """Train one run with epoch-level resume, keep the best epoch and return its per-slice test metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(config.data_root, config.held_out_fraction, config.split_seed)

    def loader(name: str, shuffle: bool) -> DataLoader:
        """DataLoader of one split."""
        dataset = NpzPairDataset(splits[name], config.input_key, config.target_key)
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
        )

    train_loader, val_loader = loader("train_noisy", True), loader("val", False)
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    uses_clean = config.method == "supervised"
    best, history = float("inf"), []
    # Resume from last.pth when it exists.
    start_epoch = 0
    resume_path = config.run_dir / RESUME_NAME
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["config"] != _resume_fingerprint(config):
            changed = {
                key: (state["config"].get(key), value)
                for key, value in _resume_fingerprint(config).items()
                if state["config"].get(key) != value
            }
            raise RuntimeError(f"{resume_path} belongs to another configuration (saved, current): {changed}")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        start_epoch = len(history)
        best, best_epoch = _best_from_history(history)
        saved_epoch = (
            torch.load(config.checkpoint_path, map_location="cpu")["epoch"] if config.checkpoint_path.exists() else None
        )
        if saved_epoch != best_epoch:
            if best_epoch != start_epoch - 1:
                raise RuntimeError(
                    f"{config.checkpoint_path} epoch {saved_epoch}, history says best epoch "
                    f"{best_epoch}: best_model.pth and last.pth are inconsistent"
                )
            # last.pth was written but best_model.pth was not: the loaded model is the best one.
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "val_selection_loss": best,
                },
                config.checkpoint_path,
            )
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"[resume] {config.run_name}: epoch {start_epoch}/{config.epochs}, resuming (best epoch {best_epoch})")
    for epoch in range(start_epoch, config.epochs):
        model.train()
        total = 0.0
        for step, (noisy, clean, _) in enumerate(
            tqdm(train_loader, desc=f"{config.run_name} epoch {epoch}", leave=False)
        ):
            noisy = noisy.to(device)
            clean = clean.to(device) if uses_clean else None
            optimizer.zero_grad(set_to_none=True)
            if config.method == "supervised":
                rec = F.mse_loss(model(noisy), clean)
            elif config.method == "ne2ne":
                rec = ne2ne_loss(model, noisy, gamma=2.0 * epoch / config.epochs)
            else:
                rec = jinv_loss(model, noisy, step % 2)
            rec.backward()
            optimizer.step()
            total += rec.item()
        val_loss = selection_loss(model, val_loader, config, device)
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_rec": total / len(train_loader),
                "val_selection_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        # Log and last.pth are written every epoch, before best_model.pth; if the run stops in between,
        # resume rewrites the best weights from last.pth.
        config.run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(config.run_dir / "training_log.csv", index=False)
        _atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "config": _resume_fingerprint(config),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            resume_path,
        )
        if val_loss < best:
            best = val_loss
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_selection_loss": val_loss,
                },
                config.checkpoint_path,
            )
    pd.DataFrame(history).to_csv(config.run_dir / "training_log.csv", index=False)
    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device)["model"])
    per_slice = evaluate_model(
        model,
        loader("test", False),
        device,
        LpipsMetric(device),
        config.run_name,
        config.seed,
        j_invariant=config.method == "jinv",
    )
    return save_per_slice_metrics(per_slice, config.run_dir / "metrics.csv")
