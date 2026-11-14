# src/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.paths import DATA_ROOT, TRAIN_ROOT

MethodName = Literal["jinv", "supervised", "ne2ne"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Data, model and training settings of one N2S run."""

    run_name: str
    data_root: Path
    output_dir: Path = TRAIN_ROOT
    input_key: str = "noisy_clipped_100k"
    target_key: str = "gt_clipped"
    held_out_fraction: float = 0.3
    split_seed: int = 42
    seed: int = 42
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 30
    use_amp: bool = False
    num_workers: int = 2
    method: MethodName = "jinv"
    vlm_name: str = "medsiglip"

    @property
    def run_dir(self) -> Path:
        """Folder of this run under the output directory."""
        return self.output_dir / self.run_name

    @property
    def checkpoint_path(self) -> Path:
        """Path of the best checkpoint of this run."""
        return self.run_dir / "best_model.pth"


# Base configuration of every N2S / N2N / supervised training run (seed and run name are replaced per run).
BASE = ExperimentConfig(run_name="base", data_root=DATA_ROOT, vlm_name="medsiglip")
