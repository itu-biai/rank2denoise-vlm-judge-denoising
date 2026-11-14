# scripts/stage2_band_moves.py
"""Stage 2 with band moves, no region mask (Table 2, "+ band moves"), 3 seeds (no training).

- 20 moves: the five stage-1 moves + 5 experts x 3 Gaussian bands (`scripts/stage2_moves.py`), alpha 0.1.
- Mean-adjusted low band: the low-band move removes the per-slice mean of the band,
      to_<e>_low(x) = clip(x + 0.1 * (G4 r - mean(G4 r)), 0, 1), r = e - x.
  The mean is removed before clipping; a small shift can remain after clipping (measured as `shift_air`).
- Search: x0 = stage-1 recipe(N2S); greedy rule (`greedy`: mean score of every move at each step, stop when the best
  does not beat the current image, ties go to the first move, <= 30 steps); masked per-seed w, `CornerMaskedVlm`.
  Validation GT is never read.
Consistency checks (the run stops if one fails): `stage2_moves.consistency_checks` and `band_self_test` before the search; per seed,
  validation score of x0 vs the stage-1 `final_mean_score` (<= 1e-4) and the mean-adjusted move keeps the slice mean
  before clipping (<= 1e-12, 4 validation slices); test PSNR of the stage-1 recipe vs `stage1_search` (<= 0.02 dB).
Test (220; GT only here): N2S, equal N2S + N2N blend, stage-1 recipe, this recipe;
  PSNR / SSIM / LPIPS / R_H + background shift (relative to N2S).

Usage (cwd = repository root):  python -m scripts.stage2_band_moves
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from skimage.metrics import structural_similarity

from scripts.expert_outputs import split_files
from scripts.infer_n2s_n2n import _atomic_csv, _atomic_json, _set_tf32
from scripts.region_maps import region_stack
from scripts.stage1_search import OUT_ROOT as STAGE1_ROOT
from scripts.stage1_search import CornerMaskedVlm
from scripts.stage2_moves import (
    Filters,
    GpuMonitor,
    TensorScorer,
    band_self_test,
    consistency_checks,
    load_split,
    move_names,
    scorer_for,
)
from src.metrics import LpipsMetric, high_band_ratio
from src.moves import BLEND, MAX_STEPS
from src.paths import RESULTS_ROOT
from src.vlm import load_vlm

OUT = RESULTS_ROOT / "stage2_band_moves"
SEEDS = (42, 123, 456)
SCORE_TOL = 1e-4
ZERO_MEAN_TOL = 1e-12
PARITY_DB = 0.02
METHODS = ("n2s", "equal_n2s_n2n", "stage1_only", "stage2_band_moves")


class ZeroMeanLowFilters(Filters):
    """Expert low-band moves are zero-mean per slice. The other 19 moves are the same as in `Filters`."""

    def apply(self, move: str, x: torch.Tensor, base: dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply a move; the expert low-band moves use the zero-mean low band."""
        if not (move.startswith("to_") and move.endswith("_low")):
            return super().apply(move, x, base)
        return (x + BLEND * self.low_step(move, x, base)).clamp(0.0, 1.0)

    def low_step(self, move: str, x: torch.Tensor, base: dict[str, torch.Tensor]) -> torch.Tensor:
        """Low band of (expert - x) with its per-slice mean removed."""
        key = move.removeprefix("to_").removesuffix("_low")
        low = self.band(base[key] - x, "low")
        return low - low.mean(dim=(1, 2), keepdim=True)


def apply_recipe(
    filters: Filters,
    steps: list[str],
    start: torch.Tensor,
    base: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply the moves of a recipe in order, starting from `start`."""
    x = start
    for move in steps:
        x = filters.apply(move, x, base)
    return x


def read_recipe(path: Path) -> dict:
    """Read a recipe JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- search (with start point x0)


def greedy(
    folder: Path,
    moves: list[str],
    start: torch.Tensor,
    base: dict[str, torch.Tensor],
    filters: Filters,
    scorer: TensorScorer,
) -> list[str]:
    """Greedy rule: mean score of every move per step; stop when the best does not beat the current image; ties go to
    the first move; <= 30 steps; x0 = `start`. Every step is written to CSV (resume).
    """
    recipe_path, steps_path = (
        folder / "recipes" / "greedy_recipe.json",
        folder / "steps" / "greedy_recipe.csv",
    )
    if recipe_path.exists():
        return read_recipe(recipe_path)["recipe"]
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(steps_path, float_precision="round_trip").to_dict("records") if steps_path.exists() else []
    recipe = [row["best_move"] for row in rows if row["accepted"]]
    current = apply_recipe(filters, recipe, start, base)
    if rows and not rows[-1]["accepted"]:
        _atomic_json(
            {"recipe": recipe, "final_mean_score": float(rows[-1]["score_before"])},
            recipe_path,
        )
        return recipe
    score = float(rows[-1]["score_after"]) if rows else float(scorer(current).mean())
    while len(recipe) < MAX_STEPS:
        means, best_images, best_mean = {}, None, -np.inf
        for move in moves:
            candidates = filters.apply(move, current, base)
            means[move] = float(scorer(candidates).mean())
            if means[move] > best_mean:
                best_mean, best_images = means[move], candidates
            del candidates
        best = max(moves, key=lambda move: (means[move], -moves.index(move)))
        accepted = means[best] > score
        ordered = sorted(means.values())
        rows.append(
            {
                "step": len(recipe) + 1,
                **{f"mean_{move}": means[move] for move in moves},
                "score_before": score,
                "best_move": best,
                "accepted": accepted,
                "score_after": means[best] if accepted else score,
                "gap_best_second": ordered[-1] - ordered[-2],
            }
        )
        _atomic_csv(pd.DataFrame(rows), steps_path)
        if not accepted:
            break
        current, score = best_images, means[best]
        recipe.append(best)
        print(f"  seed step {len(recipe)}: {best} → {score:.6f}", flush=True)
    _atomic_json({"recipe": recipe, "final_mean_score": score}, recipe_path)
    return recipe


# ----------------------------------------------------------------------------- run


def start_checks(
    seed: int,
    start: torch.Tensor,
    base: dict[str, torch.Tensor],
    filters: ZeroMeanLowFilters,
    scorer: TensorScorer,
    moves: list[str],
) -> tuple[list[dict], float]:
    """(1) score of x0 vs stored stage-1 score; (2) zero-mean moves (before clipping)."""
    recorded = read_recipe(STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json")["final_mean_score"]
    start_score = float(scorer(start).mean())
    worst_mean = max(
        float(filters.low_step(move, start[:4], {k: v[:4] for k, v in base.items()}).mean(dim=(1, 2)).abs().max())
        for move in moves
        if move.startswith("to_") and move.endswith("_low")
    )
    return [
        {
            "seed": seed,
            "check": "start_score_vs_stage1_only",
            "value": abs(start_score - recorded),
            "tolerance": SCORE_TOL,
            "passed": abs(start_score - recorded) <= SCORE_TOL,
        },
        {
            "seed": seed,
            "check": "low_step_zero_mean",
            "value": worst_mean,
            "tolerance": ZERO_MEAN_TOL,
            "passed": worst_mean <= ZERO_MEAN_TOL,
        },
    ], start_score


def slice_metrics(
    name: str,
    x: np.ndarray,
    n2s: np.ndarray,
    clean: np.ndarray,
    masks: dict[str, np.ndarray],
    lpips_metric: LpipsMetric,
) -> pd.DataFrame:
    """Per-slice test metrics of one method."""
    squared = (x - clean) ** 2
    shift = x - n2s
    rows = []
    for k in range(len(x)):
        row = {
            f"psnr_{name}": float(10.0 * np.log10(1.0 / squared[k].mean())),
            f"ssim_{name}": float(structural_similarity(clean[k], x[k], data_range=1.0)),
            f"lpips_{name}": lpips_metric(x[k].astype(np.float32), clean[k].astype(np.float32)),
            f"rh_{name}": high_band_ratio(x[k], clean[k]),
            f"shift_air_{name}": float(shift[k][masks["air"][k]].mean()) if masks["air"][k].any() else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


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
    stage1_filters, band_filters = Filters(device), ZeroMeanLowFilters(device)
    moves = move_names(seed)
    stage1_only = read_recipe(STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json")["recipe"]
    # validation: x0 = stage-1 recipe(N2S), checks, search (no GT)
    _, base = load_split(seed, "val", splits, device)
    scorer = scorer_for(seed, vlm, device)
    start = apply_recipe(stage1_filters, stage1_only, base["n2s"], base)
    checks, start_score = start_checks(seed, start, base, band_filters, scorer, moves)
    print(pd.DataFrame(checks).to_markdown(index=False), flush=True)
    if not all(row["passed"] for row in checks):
        raise RuntimeError(f"seed {seed}: start check failed; search not started.")
    print(
        f"seed {seed}: {len(moves)} moves, x0 = stage-1 recipe ({len(stage1_only)} steps)",
        flush=True,
    )
    recipe = greedy(folder, moves, start, base, band_filters, scorer)
    val_final = float(scorer(apply_recipe(band_filters, recipe, start, base)).mean())
    del base, start
    torch.cuda.empty_cache()
    # test: evaluation (GT only here)
    ids, base = load_split(seed, "test", splits, device)
    stage1_only_test = apply_recipe(stage1_filters, stage1_only, base["n2s"], base)
    outputs = {
        "n2s": base["n2s"],
        "equal_n2s_n2n": (0.5 * base["n2s"] + 0.5 * base["ne2ne"]).clamp(0.0, 1.0),
        "stage1_only": stage1_only_test,
        "stage2_band_moves": apply_recipe(band_filters, recipe, stage1_only_test, base),
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
                    "val_vlm_start": start_score,
                    "val_vlm_final": val_final,
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
        band_self_test(Filters(device), device)
        consistency_checks(splits, vlm, Filters(device), device)
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
