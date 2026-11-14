# scripts/expert_outputs.py
"""Shared read / write path for expert outputs.

Each expert's validation (182) and test (220) outputs are stored as `<R2D_RESULTS>/experts/outputs/<key>/{val,test}.npy`
(float16, (N, 512, 512), [0, 1]) + `ids_{val,test}.json` (slice order).

Noisy validation / test inputs are read from the stage-1 cache (`<R2D_RESULTS>/infer_n2s_n2n/seed<s>/cache/`); the validation cache holds no GT.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from scripts.infer_n2s_n2n import RankRecipeConfig, load_cache
from src.data import load_splits
from src.paths import RESULTS_ROOT

OUT_ROOT = RESULTS_ROOT / "experts"
OUTPUTS = OUT_ROOT / "outputs"
SPLITS = ("val", "test")
EXPECTED = {"val": 182, "test": 220}


def output_dir(key: str) -> Path:
    """Output folder of an expert."""
    return OUTPUTS / key


def output_done(key: str) -> bool:
    """True when the outputs and slice orders of every split exist."""
    folder = output_dir(key)
    return all((folder / f"{split}.npy").is_file() and (folder / f"ids_{split}.json").is_file() for split in SPLITS)


def save_output(key: str, split: str, images: np.ndarray, ids: list[str]) -> None:
    """Atomic write; shape and value range are checked."""
    if images.shape != (EXPECTED[split], 512, 512) or len(ids) != EXPECTED[split]:
        raise RuntimeError(
            f"{key}/{split}: shape {images.shape}, {len(ids)} ids, expected ({EXPECTED[split]}, 512, 512)"
        )
    if not np.isfinite(images).all() or images.min() < 0.0 or images.max() > 1.0:
        raise RuntimeError(f"{key}/{split}: not finite or outside [0, 1].")
    folder = output_dir(key)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / f"{split}.tmp.npy"
    np.save(temporary, images.astype(np.float16))
    os.replace(temporary, folder / f"{split}.npy")
    (folder / f"ids_{split}.json").write_text(json.dumps(ids), encoding="utf-8")


def load_output(key: str, split: str, ids: list[str]) -> np.ndarray:
    """Outputs of one split as float64; raises if the slice order differs."""
    folder = output_dir(key)
    saved = json.loads((folder / f"ids_{split}.json").read_text(encoding="utf-8"))
    if saved != ids:
        raise RuntimeError(f"{folder}/ids_{split}.json: slice order does not match.")
    return np.load(folder / f"{split}.npy").astype(np.float64)


def base_config(seed: int = 42) -> RankRecipeConfig:
    """Default configuration of the given seed."""
    return RankRecipeConfig(seed=seed)


def split_files() -> dict[str, list[Path]]:
    """Data split of the default configuration."""
    base = base_config()
    return load_splits(base.data_root, base.held_out_fraction, base.split_seed)


def noisy_inputs(split: str, splits: dict[str, list[Path]]) -> tuple[list[str], np.ndarray]:
    """(slice ids, (N, 512, 512) float32 noisy input) from the seed-42 stage-1 cache (the input does not depend on the seed)."""
    data = load_cache(split, splits[split], base_config())
    ids = list(data)
    return ids, np.stack([data[slice_id]["noisy"] for slice_id in ids]).astype(np.float32)
