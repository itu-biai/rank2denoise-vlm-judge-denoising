# scripts/train_n2s_seeds.py
"""Train the N2S U-Net (Noise2Self loss) with seed 123 or 456.

The data split is fixed (`split_seed` = 42); the seed changes only the weight initialisation and the batch order.
Output folder: `<R2D_RESULTS>/train/n2s_reference_seed<seed>/`.

Usage (cwd = repository root):  python -m scripts.train_n2s_seeds 123
"""

import random
import sys
from dataclasses import replace

import numpy as np
import torch

from src.config import BASE
from src.train import train_run


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(seed: int) -> None:
    """Train the N2S U-Net with the given seed."""
    set_seed(seed)
    train_run(replace(BASE, run_name=f"n2s_reference_seed{seed}", seed=seed))


if __name__ == "__main__":
    main(int(sys.argv[1]))
