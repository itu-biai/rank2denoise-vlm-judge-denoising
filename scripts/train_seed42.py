# scripts/train_seed42.py
"""Train the seed-42 U-Nets: N2S (Noise2Self loss), N2N (Neighbor2Neighbor loss) and the supervised reference (MSE).

Same U-Net and setup for all three (`src/train.py train_run`): AdamW lr 1e-4, batch 8, 30 epochs, best checkpoint =
lowest validation selection loss, epoch-level resume. Only the supervised run reads GT (its training target).
Each run is started in its own process, one call per run. Output folder: `<R2D_RESULTS>/train/<run_name>/`.

Usage (cwd = repository root):
  python -m scripts.train_seed42 n2s_reference_seed42 jinv
  python -m scripts.train_seed42 ne2ne_seed42 ne2ne
  python -m scripts.train_seed42 supervised_seed42 supervised
"""

import sys
from dataclasses import replace

import torch

from src.config import BASE
from src.train import train_run

RUNS: tuple[tuple[str, str], ...] = (
    ("n2s_reference_seed42", "jinv"),
    ("ne2ne_seed42", "ne2ne"),
    ("supervised_seed42", "supervised"),
)


def main(run_name: str, method: str) -> None:
    """Train one of the seed-42 runs (N2S, N2N or supervised)."""
    if (run_name, method) not in RUNS:
        raise SystemExit(f"unknown run {run_name} / {method}; expected one of {RUNS}")
    config = replace(BASE, run_name=run_name, method=method)
    torch.manual_seed(config.seed)
    train_run(config)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
