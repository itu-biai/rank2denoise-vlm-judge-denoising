# scripts/judge_residual_variance.py
"""Residual-variance judge on unclipped blocks (no training, no VLM).

sigma is estimated only from blocks with no clipped pixel, and the residual term uses the same blocks. Block = 2x2 Haar block; unclipped = 0 < y < 1 for all 4 pixels.
  s(x) = -| mean_B((y - x)^2) - sigma_B^2 |,  sigma_B = median_B(|HH_1(y)|) / 0.6745.
Same two-stage search as the other judges: stage 1 = the five stage-1 moves (x0 = N2S, `Filters`), stage 2 = 20 moves
(`AllMaskedFilters`, mask of the method), 3 seeds; test 220 slices.
Checks (the run stops if one fails): inputs; sigma_B torch vs numpy (3 slices, <= 1e-12); test PSNR of N2S / equal /
  stage 1 / Rank2Denoise vs `compare_judges` (per slice, <= 0.02 dB).
The block mask is computed once from the noisy input (fixed while candidates change); the run stops when a slice has no
  unclipped block. TF32 on.

Usage (cwd = repository root):  python -m scripts.judge_residual_variance
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.compare_judges import OUT as JUDGES_ROOT
from scripts.expert_outputs import split_files
from scripts.infer_n2s_n2n import _atomic_csv, _set_tf32
from scripts.region_maps import region_stack
from scripts.stage1_search import OUT_ROOT as STAGE1_ROOT
from scripts.stage2_band_moves import apply_recipe, greedy, read_recipe
from scripts.stage2_boundary_mask import slice_metrics
from scripts.stage2_moves import (
    STAGE1_MOVES,
    Filters,
    GpuMonitor,
    load_split,
    move_names,
)
from scripts.stage2_region_rule import OUT as REGION_RULE_ROOT
from scripts.stage2_region_rule import AllMaskedFilters, with_mask
from src.metrics import LpipsMetric
from src.paths import RESULTS_ROOT

OUT = RESULTS_ROOT / "judge_residual_variance"
SEEDS = (42, 123, 456)
SIGMA_TOL = 1e-12
PARITY_DB = 0.02
MAD_SCALE = 0.6745
TEST_METHODS = ("n2s", "equal_n2s_n2n", "stage1_only", "ours", "resvar_sel")
PARITY_REFS = ("n2s", "equal_n2s_n2n", "stage1_only", "ours")


# ----------------------------------------------------------------------------- region judge


def haar_hh1(images: torch.Tensor) -> torch.Tensor:
    """(N, H, W) -> finest orthonormal Haar diagonal coefficients (a - b - c + d) / 2 per 2x2 block."""
    a, b = images[:, 0::2, 0::2], images[:, 0::2, 1::2]
    c, d = images[:, 1::2, 0::2], images[:, 1::2, 1::2]
    return (a - b - c + d) / 2.0


def unclipped_blocks(noisy: torch.Tensor) -> torch.Tensor:
    """(N, H, W) -> (N, H/2, W/2) bool: all 4 pixels of the 2x2 block satisfy 0 < y < 1 (unclipped)."""
    inside = (noisy > 0.0) & (noisy < 1.0)
    return inside[:, 0::2, 0::2] & inside[:, 0::2, 1::2] & inside[:, 1::2, 0::2] & inside[:, 1::2, 1::2]


class ResidualVarianceJudge:
    """s(x) = -|mean_B((y - x)^2) - sigma_B^2|; B = unclipped blocks, sigma_B by Donoho MAD on B only. No GT."""

    def __init__(self, noisy: torch.Tensor) -> None:
        self.noisy = noisy
        blocks = unclipped_blocks(noisy)
        counts = blocks.flatten(1).sum(dim=1)
        if int(counts.min()) == 0:
            raise RuntimeError(f"{int((counts == 0).sum())} slices have no unclipped block; sigma undefined.")
        coefficients = haar_hh1(noisy).abs()
        # different number of coefficients per slice -> loop; quantile(0.5) = numpy median
        self.sigma = (
            torch.stack([torch.quantile(coefficients[k][blocks[k]], 0.5) for k in range(len(noisy))]) / MAD_SCALE
        )
        self.variance = self.sigma**2
        self.pixels = blocks.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).to(noisy.dtype)
        self.count = self.pixels.sum(dim=(1, 2))
        self.block_share = counts.double() / blocks[0].numel()

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        residual = (((self.noisy - images) ** 2) * self.pixels).sum(dim=(1, 2)) / self.count
        return -(residual - self.variance).abs()


def region_sigma_numpy(image: np.ndarray) -> float:
    """Check: separate implementation (block reshape + np.median)."""
    h, w = image.shape
    blocks = image.reshape(h // 2, 2, w // 2, 2)
    keep = ((blocks > 0.0) & (blocks < 1.0)).all(axis=(1, 3))
    hh = (blocks[:, 0, :, 0] - blocks[:, 0, :, 1] - blocks[:, 1, :, 0] + blocks[:, 1, :, 1]) / 2.0
    return float(np.median(np.abs(hh[keep])) / MAD_SCALE)


# ----------------------------------------------------------------------------- validation (per seed)


def val_stage(
    seed: int, splits: dict[str, list[Path]], device: torch.device
) -> tuple[dict[str, list[str]], list[dict]]:
    """Validation searches and checks for one seed; returns the recipes and the check rows. Saved results are reused."""
    folder = OUT / f"seed{seed}"
    done = folder / "val_done.csv"
    stage1_dir, stage2_dir = folder / "resvar" / "stage1", folder / "resvar" / "stage2"
    if done.exists():
        return (
            {
                "resvar_stage1": read_recipe(stage1_dir / "recipes" / "greedy_recipe.json")["recipe"],
                "resvar_stage2": read_recipe(stage2_dir / "recipes" / "greedy_recipe.json")["recipe"],
            },
            pd.read_csv(folder / "val_checks.csv").to_dict("records"),
        )
    stage1_filters, all_filters = Filters(device), AllMaskedFilters(device)
    moves = move_names(seed)
    ids, base = load_split(seed, "val", splits, device)
    with_mask(base, device)  # base["keep"] = mask of the method
    checks = []
    # check: sigma_B torch vs numpy
    judge = ResidualVarianceJudge(base["noisy"])
    sample = [0, len(ids) // 2, len(ids) - 1]
    numpy_sigma = np.array([region_sigma_numpy(base["noisy"][k].cpu().numpy()) for k in sample])
    gap = float(np.max(np.abs(judge.sigma[sample].cpu().numpy() - numpy_sigma)))
    checks.append(
        {
            "seed": seed,
            "check": "region_sigma_torch_vs_numpy",
            "value": gap,
            "tolerance": SIGMA_TOL,
        }
    )
    for row in checks:
        row["passed"] = row["value"] <= row["tolerance"]
    if not all(row["passed"] for row in checks):
        raise RuntimeError(
            f"seed {seed}: validation check failed; do not use these results.\n"
            + pd.DataFrame(checks).to_markdown(index=False)
        )
    # search (no GT)
    resvar_stage1 = greedy(stage1_dir, list(STAGE1_MOVES), base["n2s"], base, stage1_filters, judge)
    start = apply_recipe(stage1_filters, resvar_stage1, base["n2s"], base)
    resvar_stage2 = greedy(stage2_dir, moves, start, base, all_filters, judge)
    del start, base, judge
    torch.cuda.empty_cache()
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "recipe": "resvar_stage1",
                    "n": len(resvar_stage1),
                    "steps": " ".join(resvar_stage1),
                },
                {
                    "seed": seed,
                    "recipe": "resvar_stage2",
                    "n": len(resvar_stage2),
                    "steps": " ".join(resvar_stage2),
                },
            ]
        ),
        folder / "recipes.csv",
    )
    _atomic_csv(pd.DataFrame(checks), folder / "val_checks.csv")
    _atomic_csv(pd.DataFrame([{"seed": seed}]), done)  # written last (resume)
    return {"resvar_stage1": resvar_stage1, "resvar_stage2": resvar_stage2}, checks


# ----------------------------------------------------------------------------- test (GT for evaluation only)


def test_stage(
    seed: int,
    splits: dict[str, list[Path]],
    recipes: dict[str, list[str]],
    lpips_metric: LpipsMetric,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict]]:
    """Test evaluation of the recipes for one seed; returns the per-slice test table and the check rows. Saved results are reused."""
    folder = OUT / f"seed{seed}"
    target = folder / "test_per_slice.csv"
    if target.exists():
        return pd.read_csv(target), pd.read_csv(folder / "test_checks.csv").to_dict("records")
    stage1_filters, all_filters = Filters(device), AllMaskedFilters(device)
    stage1_only = read_recipe(STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json")["recipe"]
    ours = read_recipe(REGION_RULE_ROOT / f"seed{seed}" / "recipes" / "greedy_recipe.json")["recipe"]
    ids, base = load_split(seed, "test", splits, device)
    with_mask(base, device)
    stage1_only_test = apply_recipe(stage1_filters, stage1_only, base["n2s"], base)
    resvar_stage1 = apply_recipe(stage1_filters, recipes["resvar_stage1"], base["n2s"], base)
    outputs = {
        "n2s": base["n2s"],
        "equal_n2s_n2n": (0.5 * base["n2s"] + 0.5 * base["ne2ne"]).clamp(0.0, 1.0),
        "stage1_only": stage1_only_test,
        "ours": apply_recipe(all_filters, ours, stage1_only_test, base),
        "resvar_stage1": resvar_stage1,
        "resvar_sel": apply_recipe(all_filters, recipes["resvar_stage2"], resvar_stage1, base),
    }
    clean = base["clean"].cpu().numpy()
    images = {name: value.cpu().numpy() for name, value in outputs.items()}
    del base, outputs
    torch.cuda.empty_cache()
    masks = region_stack(images["n2s"])
    frame = pd.concat(
        [
            pd.DataFrame({"seed": seed, "slice_id": ids}),
            *(slice_metrics(name, images[name], images["n2s"], clean, masks, lpips_metric) for name in TEST_METHODS),
        ],
        axis=1,
    )
    checks = []
    for names, path in ((PARITY_REFS, JUDGES_ROOT / f"seed{seed}" / "test_per_slice.csv"),):
        reference = pd.read_csv(path).set_index("slice_id").loc[ids]
        for name in names:
            worst = float(np.abs(frame[f"psnr_{name}"].to_numpy() - reference[f"psnr_{name}"].to_numpy()).max())
            checks.append(
                {
                    "seed": seed,
                    "check": f"test_psnr_{name}_vs_{path.parts[-3]}",
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
    _atomic_csv(pd.DataFrame(checks), folder / "test_checks.csv")
    _atomic_csv(frame, target)  # written last (resume)
    return frame, checks


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per seed and method."""
    rows = []
    for seed, block in frame.groupby("seed"):
        for method in TEST_METHODS:
            row = {"seed": seed, "method": method}
            for metric in ("psnr", "ssim", "lpips", "shift_air"):
                row[metric] = block[f"{metric}_{method}"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def required_inputs() -> list[Path]:
    """Input files that must exist before the run starts."""
    paths = []
    for seed in SEEDS:
        paths += [
            STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json",
            REGION_RULE_ROOT / f"seed{seed}" / "recipes" / "greedy_recipe.json",
            JUDGES_ROOT / f"seed{seed}" / "test_per_slice.csv",
        ]
    return paths


def main() -> None:
    """Run all seeds and write the per-slice, summary and check tables."""
    began = time.time()
    missing = [str(path) for path in required_inputs() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs; nothing was run:\n  " + "\n  ".join(missing[:20]))
    device = torch.device("cuda")
    splits = split_files()
    lpips_metric = LpipsMetric(device)
    OUT.mkdir(parents=True, exist_ok=True)
    _set_tf32(True)  # TF32 on
    frames, checks, recipe_rows = [], [], []
    with GpuMonitor() as monitor:
        for seed in SEEDS:
            recipes, val_checks = val_stage(seed, splits, device)
            frame, test_checks = test_stage(seed, splits, recipes, lpips_metric, device)
            frames.append(frame)
            checks += val_checks + test_checks
            recipe_rows.append(pd.read_csv(OUT / f"seed{seed}" / "recipes.csv"))
            _atomic_csv(summarise(pd.concat(frames)), OUT / "summary.csv")  # after every seed
            _atomic_csv(pd.DataFrame(checks), OUT / "checks.csv")
            _atomic_csv(pd.concat(recipe_rows), OUT / "recipes.csv")
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
    view = (
        summarise(pd.concat(frames)).pivot_table(index="method", columns="seed", values="psnr").loc[list(TEST_METHODS)]
    )
    view["mean"] = view.mean(axis=1)
    print(view.round(3).to_markdown(), flush=True)


if __name__ == "__main__":
    main()
