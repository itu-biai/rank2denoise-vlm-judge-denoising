# src/data.py
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

# Transient read errors on network-mounted storage can surface as BadZipFile, which is not an OSError subclass,
# so both are caught. Retry count and delay below.
NPZ_MAX_RETRIES: int = 5
NPZ_RETRY_DELAY: float = 2.0


def _require_files(name: str, files: list[Path], folder: Path) -> None:
    """Fail early when a split folder is empty."""
    if not files:
        raise FileNotFoundError(f"{name} split is empty: {folder} has no *.npz files")


def _load_npz_arrays(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Load arrays from an .npz file, retrying on transient read errors (network-mounted storage)."""
    for attempt in range(NPZ_MAX_RETRIES):
        try:
            with np.load(path) as sample:
                return {key: sample[key] for key in keys}
        except (OSError, zipfile.BadZipFile) as exc:
            if attempt == NPZ_MAX_RETRIES - 1:
                raise OSError(f"Could not read file: {path}") from exc
            time.sleep(NPZ_RETRY_DELAY)
    raise AssertionError("unreachable")


def patient_id_from_path(path: Path) -> str:
    """Patient ID, the part of the file name before the first underscore."""
    return path.stem.split("_")[0]


def assert_no_patient_leakage(splits: dict[str, list[Path]]) -> None:
    """Raise if any patient appears in more than one split."""
    patients = {name: {patient_id_from_path(p) for p in files} for name, files in splits.items()}
    names = list(patients)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared = patients[first] & patients[second]
            assert not shared, f"patients in both {first} and {second}: {sorted(shared)[:5]}"


def load_splits(data_root: Path, held_out_fraction: float, split_seed: int) -> dict[str, list[Path]]:
    """Train slices split by patient into train_noisy and held_out, plus the val and test folders."""
    train = sorted((data_root / "train").glob("*.npz"))
    _require_files("train", train, data_root / "train")
    groups = np.array([patient_id_from_path(p) for p in train])
    splitter = GroupShuffleSplit(n_splits=1, test_size=held_out_fraction, random_state=split_seed)
    noisy_index, pool_index = next(splitter.split(np.arange(len(train)), groups=groups))
    splits = {
        "train_noisy": [train[i] for i in noisy_index],
        "held_out": [train[i] for i in pool_index],
        "val": sorted((data_root / "val").glob("*.npz")),
        "test": sorted((data_root / "test").glob("*.npz")),
    }
    _require_files("val", splits["val"], data_root / "val")
    _require_files("test", splits["test"], data_root / "test")
    assert_no_patient_leakage(splits)
    return splits


class NpzPairDataset(Dataset):
    """Dataset of (input, target, slice name) tensors read from .npz files."""

    def __init__(self, files: list[Path], input_key: str, target_key: str) -> None:
        self.files = sorted(files)
        self.input_key = input_key
        self.target_key = target_key

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = _load_npz_arrays(self.files[index], (self.input_key, self.target_key))
        noisy = torch.from_numpy(sample[self.input_key].astype(np.float32)).unsqueeze(0)
        clean = torch.from_numpy(sample[self.target_key].astype(np.float32)).unsqueeze(0)
        return noisy, clean, self.files[index].stem


def load_image(path: Path, key: str) -> torch.Tensor:
    """One array of an .npz file as a float32 tensor of shape (1, H, W)."""
    sample = _load_npz_arrays(path, (key,))
    return torch.from_numpy(sample[key].astype(np.float32)).unsqueeze(0)
