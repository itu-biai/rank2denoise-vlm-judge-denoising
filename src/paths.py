# src/paths.py
"""Data and results folders of every script.

Set them with environment variables before a run (defaults in brackets):
  R2D_DATA     folder with `train/`, `val/`, `test/` (*.npz)   [data]
  R2D_RESULTS  folder for every output of the pipeline         [results]
Each script writes to its own sub-folder `<R2D_RESULTS>/<script name>/`; later scripts read from there.
"""

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("R2D_DATA", "data"))
RESULTS_ROOT = Path(os.environ.get("R2D_RESULTS", "results"))

TRAIN_ROOT = RESULTS_ROOT / "train"  # U-Net trainings (`src/train.py`): <run_name>/best_model.pth
NOISE_FIELD_ROOT = RESULTS_ROOT / "noise_fields"  # N2S noise fields of the noisy training slices, one folder per seed


def n2s_checkpoint(seed: int) -> Path:
    """N2S U-Net of one seed (`scripts/train_seed42.py`, `scripts/train_n2s_seeds.py`)."""
    return TRAIN_ROOT / f"n2s_reference_seed{seed}" / "best_model.pth"


N2N_CHECKPOINT = TRAIN_ROOT / "ne2ne_seed42" / "best_model.pth"


def noise_field_dir(seed: int) -> Path:
    """Chunks `chunk_<start>.pt` of the N2S noise fields of one seed."""
    return NOISE_FIELD_ROOT / f"n2s_seed{seed}"
