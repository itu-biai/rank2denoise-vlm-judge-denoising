# scripts/stage2_region_rule.py
"""Stage-2 band recipe with the region rule (edge + background mask), 3 seeds (no training).

Region rule: no move changes the background or the organ boundaries.
- m = max(edge mask, background mask); the 15 expert band moves are x + 0.1 * (1 - m) * band (as in
  `stage2_boundary_mask.EdgeMaskedFilters`, with the larger mask); the five stage-1 moves are masked too: x + (1 - m) * (move(x) - x).
- Background mask: air | border | corner (`body_regions.region_masks`, from the N2S output, no GT) = 1, Gaussian sigma 2,
  [0, 1]. Edge mask as in `stage2_boundary_mask.py` (body_edge, dilate 4 px, Gaussian sigma 2).
  `air` = N2S output < 0.02 inside the border strip.
Search rule as in `stage2_band_moves.greedy` (20 moves, alpha 0.1, <= 30 steps, masked per-seed w), x0 = stage-1 recipe(N2S).
Consistency checks: mean m >= 0.9 on validation edge pixels and on validation pixels outside the body.
Test (220; GT only here): N2S, equal blend, stage-1, zero-mean band, region-masked band (Rank2Denoise).
  Checks: stage-1 vs `stage1_search`, zero-mean band vs `stage2_band_moves` (<= 0.02 dB).

Usage (cwd = repository root):  python -m scripts.stage2_region_rule
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from scripts.body_regions import region_masks
from scripts.expert_outputs import split_files
from scripts.infer_n2s_n2n import _atomic_csv, _set_tf32
from scripts.region_maps import region_stack
from scripts.stage1_search import OUT_ROOT as STAGE1_ROOT
from scripts.stage1_search import CornerMaskedVlm
from scripts.stage2_band_moves import OUT as BAND_MOVES_ROOT
from scripts.stage2_band_moves import (
    ZeroMeanLowFilters,
    apply_recipe,
    greedy,
    read_recipe,
)
from scripts.stage2_boundary_mask import (
    MASK_SIGMA,
    EdgeMaskedFilters,
    edge_mask,
    slice_metrics,
)
from scripts.stage2_moves import (
    STAGE1_MOVES,
    Filters,
    GpuMonitor,
    load_split,
    move_names,
    scorer_for,
)
from src.metrics import LpipsMetric
from src.paths import RESULTS_ROOT
from src.vlm import load_vlm

OUT = RESULTS_ROOT / "stage2_region_rule"
SEEDS = (42, 123, 456)
MASK_MIN = 0.9
SCORE_TOL = 1e-4
PARITY_DB = 0.02
OUTSIDE = ("air", "border", "corner")
METHODS = (
    "n2s",
    "equal_n2s_n2n",
    "stage1_only",
    "stage2_band_moves",
    "stage2_region_rule",
)


def outside_body(n2s: np.ndarray) -> np.ndarray:
    """(N, H, W) N2S output -> outside the body (air | border | corner), bool."""
    return np.stack([np.logical_or.reduce([region_masks(image)[name] for name in OUTSIDE]) for image in n2s])


def air_edge_mask(n2s: np.ndarray) -> np.ndarray:
    """m = max(edge mask, Gauss_sigma2(outside body)) in [0, 1]."""
    outside = outside_body(n2s).astype(np.float64)
    air = np.clip(np.stack([gaussian_filter(image, MASK_SIGMA) for image in outside]), 0.0, 1.0)
    return np.maximum(edge_mask(n2s), air)


class AllMaskedFilters(EdgeMaskedFilters):
    """Expert band moves as in `EdgeMaskedFilters`; the five stage-1 moves are masked too: x + (1 - m) * (move(x) - x).
    Background and boundaries are not changed by any move. Needs base["keep"] = 1 - m.
    """

    def apply(self, move: str, x: torch.Tensor, base: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply a move; every move is weighted by base['keep'] = 1 - m."""
        if move in STAGE1_MOVES:  # x and move(x) are in [0, 1]; the convex combination stays in [0, 1]
            return x + base["keep"] * (Filters.apply(self, move, x, base) - x)
        return super().apply(move, x, base)


def with_mask(base: dict[str, torch.Tensor], device: torch.device) -> np.ndarray:
    """Region mask m (edge and background) of every slice; stores 1 - m in base['keep'] and returns m."""
    m = air_edge_mask(base["n2s"].cpu().numpy())
    base["keep"] = torch.from_numpy(1.0 - m).to(device)
    return m


def run(
    seed: int,
    splits: dict[str, list[Path]],
    vlm: CornerMaskedVlm,
    lpips_metric: LpipsMetric,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict]]:
    """Searches and test evaluation for one seed; returns the per-slice test table and the check rows. Saved results are reused."""
    folder = OUT / f"seed{seed}"
    target = folder / "test_per_slice.csv"
    if target.exists():
        return pd.read_csv(target), pd.read_csv(folder / "checks.csv").to_dict("records")
    stage1_filters, band_filters, masked_filters = (
        Filters(device),
        ZeroMeanLowFilters(device),
        AllMaskedFilters(device),
    )
    moves = move_names(seed)
    stage1_only_record = read_recipe(STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json")
    stage2_band_moves = read_recipe(BAND_MOVES_ROOT / f"seed{seed}" / "recipes" / "greedy_recipe.json")["recipe"]
    # validation: mask + x0 = stage-1 recipe(N2S), checks, search (no GT)
    _, base = load_split(seed, "val", splits, device)
    m = with_mask(base, device)
    n2s_val = base["n2s"].cpu().numpy()
    edge = np.stack([region_masks(image)["body_edge"] for image in n2s_val])
    outside = outside_body(n2s_val)
    scorer = scorer_for(seed, vlm, device)
    start = apply_recipe(stage1_filters, stage1_only_record["recipe"], base["n2s"], base)
    start_score = float(scorer(start).mean())
    on_edge, on_outside = float(m[edge].mean()), float(m[outside].mean())
    checks = [
        {
            "seed": seed,
            "check": "start_score_vs_stage1_only",
            "value": abs(start_score - stage1_only_record["final_mean_score"]),
            "tolerance": SCORE_TOL,
            "passed": abs(start_score - stage1_only_record["final_mean_score"]) <= SCORE_TOL,
        },
        {
            "seed": seed,
            "check": "val_mask_mean_on_edge_min",
            "value": on_edge,
            "tolerance": MASK_MIN,
            "passed": on_edge >= MASK_MIN,
        },
        {
            "seed": seed,
            "check": "val_mask_mean_outside_body_min",
            "value": on_outside,
            "tolerance": MASK_MIN,
            "passed": on_outside >= MASK_MIN,
        },
    ]
    print(
        pd.DataFrame(checks).to_markdown(index=False),
        f"\n  share of validation pixels with m >= 0.5: {(m >= 0.5).mean():.3f}",
        flush=True,
    )
    if not all(row["passed"] for row in checks):
        raise RuntimeError(f"seed {seed}: start check failed; search not started.")
    print(
        f"seed {seed}: {len(moves)} moves, x0 = stage-1 recipe ({len(stage1_only_record['recipe'])} steps), edge + background mask",
        flush=True,
    )
    recipe = greedy(folder, moves, start, base, masked_filters, scorer)
    val_final = float(scorer(apply_recipe(masked_filters, recipe, start, base)).mean())
    val_mask_share = float((m >= 0.5).mean())
    del base, start, m
    torch.cuda.empty_cache()
    # test: evaluation (GT only here)
    ids, base = load_split(seed, "test", splits, device)
    with_mask(base, device)
    stage1_only_test = apply_recipe(stage1_filters, stage1_only_record["recipe"], base["n2s"], base)
    band_moves_test = apply_recipe(band_filters, stage2_band_moves, stage1_only_test, base)
    outputs = {
        "n2s": base["n2s"],
        "equal_n2s_n2n": (0.5 * base["n2s"] + 0.5 * base["ne2ne"]).clamp(0.0, 1.0),
        "stage1_only": stage1_only_test,
        "stage2_band_moves": band_moves_test,
        "stage2_region_rule": apply_recipe(masked_filters, recipe, stage1_only_test, base),
    }
    clean = base["clean"].cpu().numpy()
    images = {name: value.cpu().numpy() for name, value in outputs.items()}
    del base, outputs
    torch.cuda.empty_cache()
    masks = region_stack(images["n2s"])
    frame = pd.concat(
        [
            pd.DataFrame({"seed": seed, "slice_id": ids}),
            *(slice_metrics(name, images[name], images["n2s"], clean, masks, lpips_metric) for name in METHODS),
        ],
        axis=1,
    )
    for name, path, column in (
        (
            "stage1_only",
            STAGE1_ROOT / f"recipe_seed{seed}" / "test_per_slice.csv",
            "psnr_stage1_vlm",
        ),
        (
            "stage2_band_moves",
            BAND_MOVES_ROOT / f"seed{seed}" / "test_per_slice.csv",
            "psnr_stage2_band_moves",
        ),
    ):
        reference = pd.read_csv(path).set_index("slice_id")[column].loc[ids].to_numpy()
        worst = float(np.abs(frame[f"psnr_{name}"].to_numpy() - reference).max())
        checks.append(
            {
                "seed": seed,
                "check": f"test_psnr_{name}_parity",
                "value": worst,
                "tolerance": PARITY_DB,
                "passed": worst <= PARITY_DB,
            }
        )
    if not all(row["passed"] for row in checks):
        raise RuntimeError(
            f"seed {seed}: test check failed; do not use these results.\n"
            + pd.DataFrame(checks).to_markdown(index=False)
        )
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(checks), folder / "checks.csv")
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "n_steps": len(recipe),
                    "val_vlm_stage1_only": start_score,
                    "val_vlm_final": val_final,
                    "val_mask_share_ge_half": val_mask_share,
                    "recipe": " ".join(recipe),
                }
            ]
        ),
        folder / "recipe.csv",
    )
    _atomic_csv(frame, target)  # written last: its presence marks the run as done (resume)
    return frame, checks


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per seed and method, with the PSNR difference to stage 1 only."""
    rows = []
    for seed, block in frame.groupby("seed"):
        for method in METHODS:
            row = {"seed": seed, "method": method}
            for metric in ("psnr", "ssim", "lpips", "rh", "shift_air"):
                row[metric] = block[f"{metric}_{method}"].mean()
            diff = block[f"psnr_{method}"] - block["psnr_stage1_only"]
            row["gain_psnr_vs_stage1_only"] = diff.mean()
            row["slices_better_than_stage1_only"] = int((diff > 0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Run all seeds and write the per-slice, summary and check tables."""
    began = time.time()
    device = torch.device("cuda")
    splits = split_files()
    _set_tf32(True)  # VLM with TF32 on
    vlm = CornerMaskedVlm(load_vlm("medsiglip", device))
    lpips_metric = LpipsMetric(device)
    frames, checks = [], []
    with GpuMonitor() as monitor:
        for seed in SEEDS:
            frame, seed_checks = run(seed, splits, vlm, lpips_metric, device)
            frames.append(frame)
            checks.extend(seed_checks)
            _atomic_csv(summarise(pd.concat(frames)), OUT / "summary.csv")  # after every seed
            _atomic_csv(pd.DataFrame(checks), OUT / "checks.csv")
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "run_seconds": round(time.time() - began, 1),
                    "max_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    **monitor.summary(),
                }
            ]
        ),
        OUT / "cost.csv",
    )
    _atomic_csv(pd.concat(frames), OUT / "test_per_slice_all.csv")
    view = summarise(pd.concat(frames)).pivot_table(index="method", columns="seed", values="psnr").loc[list(METHODS)]
    view["mean"] = view.mean(axis=1)
    print(view.round(3).to_markdown(), flush=True)


if __name__ == "__main__":
    main()
