# scripts/stage1_search.py
"""Judge direction w (corner-masked VLM) and stage-1 recipe, three seeds. No network training, no GT for w or recipe.

Before every image enters the VLM, the four CORNER_PX x CORNER_PX corner squares are set to zero
(`CornerMaskedVlm`), for every seed and every image.

Pipeline:
 1. w (`noise_fields_seeds.rank_direction_cached`): 647 noisy training slices, each N2S seed with its own noise
    fields (`<R2D_RESULTS>/noise_fields/n2s_seed<s>/`), random seed 42. TF32 off.
 2. Stage-1 recipe (`infer_n2s_n2n.fit_recipe_vlm`) on the 182 noisy validation slices, from the cache of
    `scripts/infer_n2s_n2n.py`. TF32 on.
 3. Test evaluation (test GT only here): N2S and the stage-1 recipe, PSNR + SSIM (`data_range=1.0`).

Outputs (`<R2D_RESULTS>/stage1_search/`): `train_noisy_embeddings_masked.npy`, `w/n2s_run<s>_renoise42_masked.pt`,
`recipe_seed<s>/{steps,recipes}/all_slices.*`, `recipe_seed<s>/test_per_slice.csv`, `test_summary.csv`,
`recipes.csv`, `cost.csv`. Resume: every output is its own file; existing files are not recomputed.

Usage (cwd = repository root):  python -m scripts.stage1_search
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from scripts.infer_n2s_n2n import (
    RankRecipeConfig,
    RankScorer,
    _atomic_csv,
    _parallel,
    _set_tf32,
    apply_moves,
    fit_recipe_vlm,
    load_cache,
)
from scripts.noise_fields_seeds import (
    FIELD_CHUNK,
    RANK_BATCH,
    RENOISE_SEED,
    SEED42_FIELDS,
    _atomic_npy,
    _atomic_save,
    checkpoint_path,
    load_train_noisy,
    rank_direction_cached,
    seed42_fields,
    w_name,
)
from src.data import load_splits
from src.moves import psnr
from src.paths import RESULTS_ROOT, noise_field_dir
from src.vlm import FrozenVlm, load_vlm

OUT_ROOT = RESULTS_ROOT / "stage1_search"
CORNER_PX = 8  # corner square size in pixels
SEEDS: tuple[int, ...] = (42, 123, 456)
SEED_FIELDS = {seed: noise_field_dir(seed) for seed in (123, 456)}  # written by `scripts/noise_fields_seeds.py`
RECIPE_NAME = "all_slices"  # one recipe for all validation slices
METHODS = ("n2s", "stage1_vlm")


class CornerMaskedVlm:
    """VLM wrapper: the four CORNER_PX x CORNER_PX corner squares of the (N, 1, H, W) input are set to zero before
    embedding. `RankScorer` and `rank_direction_cached` call only `encode_image`, so both are used unchanged.
    """

    def __init__(self, vlm: FrozenVlm) -> None:
        self.vlm = vlm

    @staticmethod
    def mask(images: torch.Tensor) -> torch.Tensor:
        """Copy of the images with the four corner squares set to zero."""
        masked = images.clone()
        c = CORNER_PX
        masked[..., :c, :c] = 0.0
        masked[..., :c, -c:] = 0.0
        masked[..., -c:, :c] = 0.0
        masked[..., -c:, -c:] = 0.0
        return masked

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """VLM embedding of the corner-masked images."""
        return self.vlm.encode_image(self.mask(images))


@dataclass(frozen=True)
class MaskedRecipeConfig(RankRecipeConfig):
    """Same settings as `RankRecipeConfig` (start_batch 512, halves on OOM); separate output folder."""

    @property
    def out_dir(self) -> Path:
        """Output folder of this seed."""
        return OUT_ROOT / f"recipe_seed{self.seed}"


def mask_self_test() -> None:
    """Check that the mask changes only the corner squares (CPU, no VLM)."""
    images = torch.rand(2, 1, 512, 512) + 0.1
    masked = CornerMaskedVlm.mask(images)
    changed = (masked != images)[0, 0]
    if int(changed.sum()) != 4 * CORNER_PX * CORNER_PX or float(masked[..., :CORNER_PX, :CORNER_PX].abs().max()) != 0.0:
        raise RuntimeError(
            f"corner mask does not change the expected {4 * CORNER_PX**2} pixels ({int(changed.sum())})."
        )
    if not torch.equal(images[..., CORNER_PX:-CORNER_PX, :], masked[..., CORNER_PX:-CORNER_PX, :]):
        raise RuntimeError("corner mask changes pixels outside the corners.")


def seed_fields(seed: int, stems: list[str], key: str) -> torch.Tensor:
    """Noise fields of seeds 123 / 456 (read only; not recomputed when missing)."""
    folder, checkpoint_bytes, chunks = (
        SEED_FIELDS[seed],
        checkpoint_path(seed).stat().st_size,
        [],
    )
    for start in range(0, len(stems), FIELD_CHUNK):
        saved = torch.load(folder / f"chunk_{start:04d}.pt", map_location="cpu")
        if (
            saved["stems"] != stems[start : start + FIELD_CHUNK]
            or saved["key"] != key
            or saved["checkpoint_bytes"] != checkpoint_bytes
        ):
            raise RuntimeError(f"{folder}/chunk_{start:04d}.pt: slices / key / checkpoint do not match.")
        chunks.append(saved["fields"])
    return torch.cat(chunks)


def masked_noisy_embeddings(
    noisy: torch.Tensor, stems: list[str], vlm: CornerMaskedVlm, device: torch.device
) -> torch.Tensor:
    """phi(mask(y)), same batch (16) and order as `src.renoise.rank_direction`. Resume: read when the file exists."""
    target, ids_path = (
        OUT_ROOT / "train_noisy_embeddings_masked.npy",
        OUT_ROOT / "train_noisy_slice_ids.json",
    )
    if target.exists():
        if json.loads(ids_path.read_text(encoding="utf-8")) != stems:
            raise RuntimeError(f"{ids_path}: slice order does not match; {OUT_ROOT} comes from another run.")
        return torch.from_numpy(np.load(target))
    with torch.no_grad():
        embeddings = torch.cat(
            [
                vlm.encode_image(noisy[start : start + RANK_BATCH].to(device)).cpu()
                for start in range(0, len(noisy), RANK_BATCH)
            ]
        )
    ids_path.write_text(json.dumps(stems), encoding="utf-8")
    _atomic_npy(embeddings.numpy(), target)
    return embeddings


def build_masked_w(
    seed: int,
    vlm: CornerMaskedVlm,
    noisy: torch.Tensor,
    phi_noisy: torch.Tensor,
    fields: torch.Tensor,
    device: torch.device,
) -> dict:
    """Judge direction w of the seed with the corner-masked VLM; a saved w is reused."""
    target = OUT_ROOT / "w" / f"{w_name(seed, RENOISE_SEED)}_masked.pt"
    if target.exists():
        return torch.load(target, map_location="cpu")
    began = time.time()
    direction, margin = rank_direction_cached(vlm, noisy, phi_noisy, fields, RENOISE_SEED, device)
    if not torch.isfinite(direction).all() or abs(float(direction.norm()) - 1.0) > 1e-4 or not np.isfinite(margin):
        raise RuntimeError(f"seed {seed}: invalid masked w / margin (||w|| {float(direction.norm())}, margin {margin})")
    payload = {
        "direction": direction,
        "margin": margin,
        "n2s_seed": seed,
        "gen_seed": RENOISE_SEED,
        "corner_mask_px": CORNER_PX,
    }
    _atomic_save(payload, target)
    print(
        f"  masked w seed {seed}: margin {margin:.6f} · {time.time() - began:.0f} s",
        flush=True,
    )
    return payload


def required_inputs(splits: dict[str, list[Path]], stems: list[str], base: RankRecipeConfig) -> list[Path]:
    """Input files that must exist before the run starts."""
    paths: list[Path] = [*(checkpoint_path(seed) for seed in SEED_FIELDS)]
    paths += [SEED42_FIELDS / f"chunk_{start:04d}.pt" for start in range(0, len(stems), FIELD_CHUNK)]
    paths += [
        folder / f"chunk_{start:04d}.pt"
        for folder in SEED_FIELDS.values()
        for start in range(0, len(stems), FIELD_CHUNK)
    ]
    for seed in SEEDS:
        cache = RankRecipeConfig(seed=seed).out_dir / "cache"
        paths += [
            cache / f"{split}_{start:04d}.npz"
            for split in ("val", "test")
            for start in range(0, len(splits[split]), base.chunk_files)
        ]
    return paths


def w_stage(
    splits: dict[str, list[Path]],
    stems: list[str],
    base: RankRecipeConfig,
    vlm: CornerMaskedVlm,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    """Judge direction w of every seed; returns the directions by seed."""
    _set_tf32(False)
    targets = {seed: OUT_ROOT / "w" / f"{w_name(seed, RENOISE_SEED)}_masked.pt" for seed in SEEDS}
    if not all(path.exists() for path in targets.values()):
        print(f"reading noisy training slices: {len(stems)} slices", flush=True)
        noisy = load_train_noisy(splits["train_noisy"], base.input_key, base.read_threads)
        phi_noisy = masked_noisy_embeddings(noisy, stems, vlm, device)
        for seed in SEEDS:
            if targets[seed].exists():
                continue
            fields = seed42_fields(stems, base.input_key) if seed == 42 else seed_fields(seed, stems, base.input_key)
            build_masked_w(seed, vlm, noisy, phi_noisy, fields, device)
            del fields
        del noisy, phi_noisy
    return {seed: torch.load(targets[seed], map_location="cpu")["direction"].float() for seed in SEEDS}


def evaluate_seed(
    seed: int,
    test: dict[str, dict[str, np.ndarray]],
    recipe: list[str],
    config: MaskedRecipeConfig,
) -> pd.DataFrame:
    """Test PSNR / SSIM of N2S and of the stage-1 recipe (test GT only here)."""

    def one(slice_id: str) -> dict:
        """Test metrics of one slice for N2S and the stage-1 recipe."""
        data = test[slice_id]
        images = {"n2s": data["n2s"], "stage1_vlm": apply_moves(data, recipe)}
        row = {"seed": seed, "slice_id": slice_id}
        for method, image in images.items():
            row[f"psnr_{method}"] = psnr(image, data["clean"])
            row[f"ssim_{method}"] = float(structural_similarity(data["clean"], image, data_range=1.0))
        return row

    per_slice = pd.DataFrame(_parallel(one, list(test), config.move_threads))
    _atomic_csv(per_slice, config.out_dir / "test_per_slice.csv")
    return per_slice


def summarise(per_slice: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics and gain over N2S per seed and method."""
    rows = []
    for seed, block in per_slice.groupby("seed"):
        for method in METHODS:
            gain = block[f"psnr_{method}"] - block["psnr_n2s"]
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "n_slices": len(block),
                    "psnr": block[f"psnr_{method}"].mean(),
                    "ssim": block[f"ssim_{method}"].mean(),
                    "gain_psnr_vs_n2s": gain.mean(),
                    "slices_better_than_n2s": int((gain > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Build w, search the stage-1 recipe and evaluate it on the test slices for every seed."""
    run_began = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (OUT_ROOT / "w").mkdir(parents=True, exist_ok=True)
    mask_self_test()
    base = RankRecipeConfig()
    splits = load_splits(base.data_root, base.held_out_fraction, base.split_seed)
    stems = [path.stem for path in splits["train_noisy"]]
    missing = [str(path) for path in required_inputs(splits, stems, base) if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs; nothing was run:\n  " + "\n  ".join(missing[:20]))
    vlm = CornerMaskedVlm(load_vlm(base.vlm_name, device))

    # 1. masked w (three seeds)
    directions = w_stage(splits, stems, base, vlm, device)

    # 2-3. stage-1 recipe (masked VLM, masked w) + test, seed by seed
    _set_tf32(True)
    all_rows, recipe_rows, embedded, vlm_seconds = [], [], 0, 0.0
    for seed in SEEDS:
        config = MaskedRecipeConfig(seed=seed)
        config.out_dir.mkdir(parents=True, exist_ok=True)
        read_config = RankRecipeConfig(seed=seed)  # stage-1 cache (read only)
        val = load_cache("val", splits["val"], read_config)
        scorer = RankScorer(vlm, directions[seed], config, device)
        print(
            f"seed {seed} · stage-1 recipe selection: {RECIPE_NAME} ({len(val)} slices)",
            flush=True,
        )
        recipe = fit_recipe_vlm(RECIPE_NAME, list(val), val, scorer, config)
        print(f"  {RECIPE_NAME}: {' '.join(recipe) or '(empty: N2S)'}", flush=True)
        recipe_rows.append(
            {
                "seed": seed,
                "form": RECIPE_NAME,
                "n_steps": len(recipe),
                "recipe": " ".join(recipe),
            }
        )
        embedded, vlm_seconds = embedded + scorer.embedded, vlm_seconds + scorer.seconds
        del val
        target = config.out_dir / "test_per_slice.csv"
        if target.exists():
            all_rows.append(pd.read_csv(target))
        else:
            test = load_cache("test", splits["test"], read_config)
            all_rows.append(evaluate_seed(seed, test, recipe, config))
            del test
        _atomic_csv(pd.DataFrame(recipe_rows), OUT_ROOT / "recipes.csv")
        _atomic_csv(summarise(pd.concat(all_rows)), OUT_ROOT / "test_summary.csv")  # written after every seed

    summary = summarise(pd.concat(all_rows))
    _atomic_csv(summary, OUT_ROOT / "test_summary.csv")
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "recipe_embedded_images": embedded,
                    "recipe_vlm_seconds": round(vlm_seconds, 1),
                    "images_per_second": embedded / max(vlm_seconds, 1e-9),
                    "run_seconds": round(time.time() - run_began, 1),
                }
            ]
        ),
        OUT_ROOT / "cost.csv",
    )
    print(pd.DataFrame(recipe_rows).to_markdown(index=False), flush=True)
    print(
        summary.pivot(index="method", columns="seed", values="psnr").loc[list(METHODS)].round(3).to_markdown(),
        flush=True,
    )


if __name__ == "__main__":
    main()
