from pathlib import Path

import numpy as np
import pytest

from src.data import assert_no_patient_leakage, load_splits, patient_id_from_path


def _write_split(root: Path, split: str, patients: list[str], slices: int) -> None:
    folder = root / split
    folder.mkdir(parents=True)
    for patient in patients:
        for k in range(slices):
            np.savez(folder / f"{patient}_01_01_{k:03d}.npz", x=np.zeros(1))


def test_patient_id_is_the_first_part_of_the_name() -> None:
    """The patient ID is the file name part before the first underscore."""
    assert patient_id_from_path(Path("000123_02_01_045.npz")) == "000123"


def test_splits_keep_each_patient_in_one_split(tmp_path: Path) -> None:
    """The noisy-training and held-out sets share no patient."""
    _write_split(tmp_path, "train", [f"{i:06d}" for i in range(10)], 3)
    _write_split(tmp_path, "val", ["100000", "100001"], 2)
    _write_split(tmp_path, "test", ["200000", "200001"], 2)
    splits = load_splits(tmp_path, held_out_fraction=0.3, split_seed=42)
    assert len(splits["train_noisy"]) + len(splits["held_out"]) == 30
    assert len(splits["val"]) == len(splits["test"]) == 4
    noisy = {patient_id_from_path(p) for p in splits["train_noisy"]}
    held_out = {patient_id_from_path(p) for p in splits["held_out"]}
    assert noisy and held_out and not noisy & held_out


def test_empty_split_raises(tmp_path: Path) -> None:
    """An empty split folder raises FileNotFoundError."""
    _write_split(tmp_path, "train", ["000001", "000002"], 1)
    _write_split(tmp_path, "val", ["100000"], 1)
    (tmp_path / "test").mkdir()
    with pytest.raises(FileNotFoundError):
        load_splits(tmp_path, held_out_fraction=0.5, split_seed=42)


def test_leakage_check_raises_on_shared_patient() -> None:
    """A patient in two splits raises AssertionError."""
    splits = {"val": [Path("000001_a.npz")], "test": [Path("000001_b.npz")]}
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(splits)
