# scripts/noise_fields_seeds.py
"""N2S noise fields of the noisy training slices for N2S seeds 123 and 456, and shared tools for building w. No training.

Noise field (`src/renoise.py`): n_i = y_i - x_J(y_i) (no clipping, N2S J-invariant two passes), for the 647 noisy
training slices. Output: `<R2D_RESULTS>/noise_fields/n2s_seed<s>/chunk_*.pt` (resume: existing chunks are skipped),
the same format as `scripts/noise_fields_seed42.py`.
Check: for seed 42, the first chunk is recomputed here and compared with the chunk of `scripts/noise_fields_seed42.py`
(<= 1e-4; `parity_fields.csv`), so both code paths give the same fields.

Shared tools: `rank_direction_cached` builds w = sum(phi(y) - phi(y^+)) / ||.|| and the margin
mean <phi(y) - phi(y^+), w> with y^+ = clip(y + pi(n_k), 0, 1), from pre-computed phi(y).

GT: not used.

Usage (cwd = repository root):  python -m scripts.noise_fields_seeds
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.infer_n2s_n2n import (
    CKPT_SHA256,
    RankRecipeConfig,
    _atomic_csv,
    _set_tf32,
    load_unet,
)
from src.data import _load_npz_arrays, load_splits
from src.masking import j_invariant_inference
from src.paths import NOISE_FIELD_ROOT, noise_field_dir
from src.renoise import renoised
from src.vlm import FrozenVlm

SEED42_FIELDS = noise_field_dir(42)  # seed-42 noise-field chunks of `scripts/noise_fields_seed42.py` (read only)
FIELD_CHUNK = 50  # same as scripts/noise_fields_seed42.py FIELD_CHUNK (647 slices -> 13 chunks)
OTHER_SEEDS: tuple[int, ...] = (123, 456)
RENOISE_SEED = 42  # default of src/renoise.rank_direction
RANK_BATCH = 16  # default of src/renoise.rank_direction
FIELDS_PARITY = 1e-4  # noise field, seed-42 chunk vs recomputed (GPU difference ~1e-6)


def _atomic_save(payload: dict, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _atomic_npy(array: np.ndarray, target: Path) -> None:
    temporary = target.with_name(target.stem + ".tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, target)


def w_name(n2s_seed: int, gen_seed: int) -> str:
    """File name of the w of an N2S seed and a generator seed."""
    return f"n2s_run{n2s_seed}_renoise{gen_seed}"


# ----------------------------------------------------------------------------- noisy training slices (once)


def load_train_noisy(files: list[Path], key: str, threads: int) -> torch.Tensor:
    """(647, 1, 512, 512) float32, same conversion as `src.data.load_image`, parallel reads."""
    with ThreadPoolExecutor(threads) as pool:
        arrays = list(
            pool.map(
                lambda path: _load_npz_arrays(path, (key,))[key].astype(np.float32),
                files,
            )
        )
    return torch.from_numpy(np.stack(arrays)).unsqueeze(1)


# ----------------------------------------------------------------------------- noise fields


@torch.no_grad()
def compute_fields(model: torch.nn.Module, noisy: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Same computation as `src.renoise.noise_fields` (slice by slice, no clipping); input from memory."""
    model.eval()
    return torch.cat(
        [
            (
                noisy[index : index + 1].to(device) - j_invariant_inference(model, noisy[index : index + 1].to(device))
            ).cpu()
            for index in range(len(noisy))
        ]
    )


def checkpoint_path(seed: int) -> Path:
    """N2S checkpoint of the seed."""
    return RankRecipeConfig(seed=seed).n2s_checkpoint


def fields_for_seed(seed: int, noisy: torch.Tensor, stems: list[str], key: str, device: torch.device) -> torch.Tensor:
    """Seeds 123 / 456: compute and save chunk by chunk (resume). Same format as `scripts/noise_fields_seed42.py`."""
    folder = noise_field_dir(seed)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_path(seed)
    checkpoint_bytes, model, chunks = checkpoint.stat().st_size, None, []
    for start in range(0, len(stems), FIELD_CHUNK):
        part, part_stems = (
            folder / f"chunk_{start:04d}.pt",
            stems[start : start + FIELD_CHUNK],
        )
        if part.exists():
            saved = torch.load(part, map_location="cpu")
            if saved["stems"] != part_stems or saved["key"] != key or saved["checkpoint_bytes"] != checkpoint_bytes:
                raise RuntimeError(f"{part} comes from another run (slices / key / checkpoint differ).")
            chunks.append(saved["fields"])
            continue
        if model is None:
            model = load_unet(checkpoint, CKPT_SHA256[seed], device)
        fields = compute_fields(model, noisy[start : start + FIELD_CHUNK], device)
        _atomic_save(
            {
                "stems": part_stems,
                "key": key,
                "checkpoint_bytes": checkpoint_bytes,
                "fields": fields,
            },
            part,
        )
        chunks.append(fields)
        print(
            f"  noise field seed {seed}: {start + len(part_stems)}/{len(stems)}",
            flush=True,
        )
    return torch.cat(chunks)


def seed42_fields(stems: list[str], key: str) -> torch.Tensor:
    """Seed-42 noise-field chunks (read only). Slice order and key are checked."""
    chunks = []
    for start in range(0, len(stems), FIELD_CHUNK):
        saved = torch.load(SEED42_FIELDS / f"chunk_{start:04d}.pt", map_location="cpu")
        if saved["stems"] != stems[start : start + FIELD_CHUNK] or saved["key"] != key:
            raise RuntimeError(f"{SEED42_FIELDS}/chunk_{start:04d}.pt: slice order / key do not match.")
        chunks.append(saved["fields"])
    return torch.cat(chunks)


def fields_parity(noisy: torch.Tensor, reference: torch.Tensor, device: torch.device) -> float:
    """Seed 42, first chunk: recomputed field against the chunk of `scripts/noise_fields_seed42.py`."""
    model = load_unet(checkpoint_path(42), CKPT_SHA256[42], device)
    difference = float((compute_fields(model, noisy[:FIELD_CHUNK], device) - reference[:FIELD_CHUNK]).abs().max())
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "check": "fields_n2s_seed42_chunk0_recomputed",
                    "value": difference,
                    "tolerance": FIELDS_PARITY,
                }
            ]
        ),
        NOISE_FIELD_ROOT / "parity_fields.csv",
    )
    print(
        f"noise-field check (seed 42, first {FIELD_CHUNK} slices): largest difference {difference:.2e}",
        flush=True,
    )
    if difference > FIELDS_PARITY:
        raise RuntimeError(f"noise field does not match the seed-42 chunk ({difference:.2e} > {FIELDS_PARITY}).")
    return difference


# ----------------------------------------------------------------------------- building w


@torch.no_grad()
def rank_direction_cached(
    vlm: FrozenVlm,
    noisy: torch.Tensor,
    phi_noisy: torch.Tensor,
    fields: torch.Tensor,
    gen_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Same as `src.renoise.rank_direction`: same batch (16), same random-generator order, same formula.
    Difference: phi(y) is pre-computed (`phi_noisy`); y^+ is built with `renoised` and embedded per batch.
    """
    generator = torch.Generator().manual_seed(gen_seed)
    differences = []
    for start in range(0, len(noisy), RANK_BATCH):
        copies = renoised(noisy[start : start + RANK_BATCH], fields, generator)
        differences.append(phi_noisy[start : start + RANK_BATCH] - vlm.encode_image(copies.to(device)).cpu())
    differences = torch.cat(differences)
    direction = differences.sum(0) / differences.sum(0).norm()
    return direction, float((differences @ direction).mean())


# ----------------------------------------------------------------------------- main


def main() -> None:
    """Compute the noise fields of seeds 123 and 456 and run the seed-42 parity check."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = RankRecipeConfig()
    splits = load_splits(base.data_root, base.held_out_fraction, base.split_seed)
    stems = [path.stem for path in splits["train_noisy"]]
    required = [
        *(checkpoint_path(seed) for seed in (42, *OTHER_SEEDS)),
        *(SEED42_FIELDS / f"chunk_{start:04d}.pt" for start in range(0, len(stems), FIELD_CHUNK)),
    ]
    missing = [str(path) for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs; nothing was run:\n  " + "\n  ".join(missing[:20]))
    _set_tf32(False)
    print(f"reading noisy training slices: {len(stems)} slices", flush=True)
    noisy = load_train_noisy(splits["train_noisy"], base.input_key, base.read_threads)
    fields_parity(noisy, seed42_fields(stems, base.input_key), device)
    for seed in OTHER_SEEDS:
        fields_for_seed(seed, noisy, stems, base.input_key, device)
    print(f"noise fields written: {NOISE_FIELD_ROOT}", flush=True)


if __name__ == "__main__":
    main()
