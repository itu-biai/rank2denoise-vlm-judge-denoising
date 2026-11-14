# scripts/noise_fields_seed42.py
"""N2S noise fields of the noisy training slices for N2S seed 42 (no training, no GT).

  n_i = y_i - x_hat_i with the N2S model (seed 42), for every noisy training slice (`src.renoise.noise_fields`).

- Output: `<R2D_RESULTS>/noise_fields/n2s_seed42/chunk_*.pt`, computed in chunks of `FIELD_CHUNK` slices; slices are
  independent, so the chunks together equal a single call. Existing chunks are not recomputed (resume).

Usage (cwd = repository root):  python -m scripts.noise_fields_seed42
"""

import os
from pathlib import Path

import torch

from src.config import BASE, ExperimentConfig
from src.data import load_splits
from src.paths import n2s_checkpoint, noise_field_dir
from src.renoise import noise_fields
from src.unet import UNet

N2S_CHECKPOINT = n2s_checkpoint(42)  # written by `scripts/train_seed42.py`, read only here
FIELD_DIR = noise_field_dir(42)
FIELD_CHUNK = 50  # slices per chunk (647 slices -> 13 chunks)


def _atomic_save(payload: object, target: Path) -> None:
    """`.tmp` -> `os.replace`: no partial file is left under the target name."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_or_compute_fields(config: ExperimentConfig, files: list[Path], device: torch.device) -> torch.Tensor:
    """Compute `noise_fields` chunk by chunk; existing chunks are not recomputed.

    Each chunk stores its slice names, input key and checkpoint size; a mismatch raises an error
    instead of reusing a stale chunk.

    """
    folder = FIELD_DIR
    checkpoint_bytes = N2S_CHECKPOINT.stat().st_size
    model = None
    chunks = []
    for start in range(0, len(files), FIELD_CHUNK):
        part_files = files[start : start + FIELD_CHUNK]
        stems = [path.stem for path in part_files]
        part = folder / f"chunk_{start:04d}.pt"
        if part.exists():
            saved = torch.load(part, map_location="cpu")
            if (
                saved["stems"] != stems
                or saved["key"] != config.input_key
                or saved["checkpoint_bytes"] != checkpoint_bytes
            ):
                raise RuntimeError(
                    f"{part} comes from another run (slices / key / checkpoint differ). "
                    f"{folder}: delete this folder and run again."
                )
            chunks.append(saved["fields"])
            print(
                f"noise field {start + len(part_files)}/{len(files)} (from disk)",
                flush=True,
            )
            continue
        if model is None:
            model = UNet().to(device)
            model.load_state_dict(torch.load(N2S_CHECKPOINT, map_location=device)["model"])
        fields = noise_fields(model, part_files, config.input_key, device)
        _atomic_save(
            {
                "stems": stems,
                "key": config.input_key,
                "checkpoint_bytes": checkpoint_bytes,
                "fields": fields,
            },
            part,
        )
        chunks.append(fields)
        print(f"noise field {start + len(part_files)}/{len(files)} (computed)", flush=True)
    return torch.cat(chunks)


def main(config: ExperimentConfig = BASE) -> Path:
    """Compute the seed-42 noise fields (resume)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not N2S_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing N2S seed-42 checkpoint: {N2S_CHECKPOINT} (training output)")
    splits = load_splits(config.data_root, config.held_out_fraction, config.split_seed)
    files = splits["train_noisy"]
    load_or_compute_fields(config, files, device)
    print(f"noise fields written: {FIELD_DIR} · {len(files)} slices", flush=True)
    return FIELD_DIR


if __name__ == "__main__":
    main()
