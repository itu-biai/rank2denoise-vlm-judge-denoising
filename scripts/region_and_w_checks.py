# scripts/region_and_w_checks.py
"""Region-rule signs, path checks, w controls and the search without S2S (GPU, no training).

- Region-rule signs on validation slices (GT is not read), 5 variants: stage 1, unmasked band (`stage2_band_moves`), edge mask
  only (`stage2_boundary_mask`), background only (background-only mask of `single_experts_and_checks`), Rank2Denoise (`stage2_region_rule`):
  background shift = mean(x - N2S) on air pixels; edge sharpness = mean |grad G_1.5 x| on boundary pixels / that of N2S.
- Path checks and w controls: (a) at every step of the stage-1 and stage-2 paths, the validation PSNR of all candidates
  (validation GT for measurement only): does the chosen move raise the mean validation PSNR, its rank among the
  candidates, share of improved slices. (b) w is rebuilt with two controls: the slice's own N2S field (donor = the slice
  itself) and white Gaussian noise of the same strength (std of the donor); cosine with the original w.
- Search without S2S: stage 2 from the stage-1 output with 17 moves (the 3 S2S band moves removed), same mask and w, 3 seeds.
Details: separate generator for white noise (WHITE_GEN_SEED); donor / flip draws use the same generator order as the
  original w; 'boundary' = region_masks body_edge (top 10% of the body gradient), 'air' = air region (border strip /
  corners excluded; same as shift_air on test); also written: abs_shift_air_mean, frac_slices_edge_blurred,
  best_by_val_psnr.
Checks (the run stops if one fails): input files (including N2S checkpoints); rebuilt w vs stored w (cosine >= 0.9999,
  margin difference <= 1e-3); validation scores of the variants vs stored final_mean_score (<= 1e-4); recipe JSON vs step
  CSV; x0 score of the search without S2S vs stage 1; test PSNR of N2S / equal / stage 1 / Rank2Denoise vs `compare_judges`
  (per slice, <= 0.02 dB).

Usage (cwd = repository root):  python -m scripts.region_and_w_checks
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from scripts.compare_judges import OUT as JUDGES_ROOT
from scripts.compare_judges import val_clean
from scripts.expert_outputs import split_files
from scripts.infer_n2s_n2n import RankRecipeConfig, _atomic_csv, _set_tf32
from scripts.noise_fields_seeds import (
    RANK_BATCH,
    RENOISE_SEED,
    checkpoint_path,
    load_train_noisy,
    seed42_fields,
)
from scripts.region_maps import region_stack
from scripts.single_experts_and_checks import OUT as SINGLES_ROOT
from scripts.single_experts_and_checks import bg_only_mask
from scripts.stage1_search import OUT_ROOT as STAGE1_ROOT
from scripts.stage1_search import CornerMaskedVlm, masked_noisy_embeddings, seed_fields
from scripts.stage2_band_moves import OUT as BAND_MOVES_ROOT
from scripts.stage2_band_moves import (
    ZeroMeanLowFilters,
    apply_recipe,
    greedy,
    read_recipe,
)
from scripts.stage2_boundary_mask import OUT as BOUNDARY_MASK_ROOT
from scripts.stage2_boundary_mask import EdgeMaskedFilters, edge_mask, slice_metrics
from scripts.stage2_moves import (
    STAGE1_MOVES,
    Filters,
    GpuMonitor,
    load_split,
    move_names,
    scorer_for,
)
from scripts.stage2_region_rule import OUT as REGION_RULE_ROOT
from scripts.stage2_region_rule import AllMaskedFilters, with_mask
from src.metrics import LpipsMetric
from src.paths import RESULTS_ROOT
from src.vlm import load_vlm

OUT = RESULTS_ROOT / "region_and_w_checks"
SEEDS = (42, 123, 456)
SCORE_TOL = 1e-4
PARITY_DB = 0.02
W_COSINE_MIN = 0.9999
MARGIN_TOL = 1e-3  # margin tolerance
EDGE_SIGMA = 1.5  # same smoothing as the boundary gradient in region_masks
WHITE_GEN_SEED = 7  # white Gaussian (w control)
TEST_METHODS = ("n2s", "equal_n2s_n2n", "stage1_only", "ours", "no_s2s")
PARITY = ("n2s", "equal_n2s_n2n", "stage1_only", "ours")


# ----------------------------------------------------------------------------- region-rule signs on validation (no GT)


def edge_sharpness(x: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Mean |grad G_1.5 x| on boundary pixels, per slice."""
    out = np.empty(len(x))
    for k in range(len(x)):
        gy, gx = np.gradient(gaussian_filter(x[k], EDGE_SIGMA))
        out[k] = np.hypot(gx, gy)[edge[k]].mean() if edge[k].any() else np.nan
    return out


def region_sign_rows(seed: int, variants: dict[str, np.ndarray], n2s: np.ndarray) -> pd.DataFrame:
    """Background shift and edge sharpness of each variant relative to N2S."""
    masks = region_stack(n2s)
    base_sharp = edge_sharpness(n2s, masks["body_edge"])
    rows = []
    for name, x in variants.items():
        shift = x - n2s
        air = np.array([shift[k][masks["air"][k]].mean() if masks["air"][k].any() else np.nan for k in range(len(x))])
        ratio = edge_sharpness(x, masks["body_edge"]) / base_sharp
        rows.append(
            {
                "seed": seed,
                "variant": name,
                "shift_air_mean": float(np.nanmean(air)),
                "abs_shift_air_mean": float(np.nanmean(np.abs(air))),
                "edge_sharpness_ratio_mean": float(np.nanmean(ratio)),
                "frac_slices_edge_blurred": float(np.nanmean(ratio < 1.0)),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- path checks (validation GT for measurement only)


def psnr_per_slice(images: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """PSNR of every slice with data range 1."""
    return 10.0 * torch.log10(1.0 / ((images - clean) ** 2).mean(dim=(1, 2)))


def path_rows(
    seed: int,
    stage: str,
    recipe: list[str],
    moves: list[str],
    start: torch.Tensor,
    base: dict[str, torch.Tensor],
    filters: Filters,
    clean: torch.Tensor,
    recorded: pd.DataFrame | None,
) -> tuple[pd.DataFrame, float]:
    """Replay the stored path; validation PSNR of all candidates at every step. Returns step rows and the number of mismatches with the record."""
    rows, current, mismatches = [], start, 0
    for step, chosen in enumerate(recipe, 1):
        before = psnr_per_slice(current, clean)
        means, chosen_slices = {}, None
        for move in moves:
            values = psnr_per_slice(filters.apply(move, current, base), clean)
            means[move] = float(values.mean())
            if move == chosen:
                chosen_slices = values
        order = sorted(moves, key=lambda move: -means[move])
        if recorded is not None and str(recorded.iloc[step - 1]["best_move"]) != chosen:
            mismatches += 1
        rows.append(
            {
                "seed": seed,
                "stage": stage,
                "step": step,
                "chosen": chosen,
                "val_psnr_before": float(before.mean()),
                "val_psnr_after": means[chosen],
                "chosen_raises_val_psnr": means[chosen] > float(before.mean()),
                "chosen_rank_by_val_psnr": order.index(chosen) + 1,
                "n_candidates": len(moves),
                "best_by_val_psnr": order[0],
                "frac_slices_improved": float((chosen_slices > before).double().mean()),
            }
        )
        current = filters.apply(chosen, current, base)
    return pd.DataFrame(rows), mismatches


# ----------------------------------------------------------------------------- w controls


def control_direction(
    kind: str,
    noisy: torch.Tensor,
    phi_noisy: torch.Tensor,
    fields: torch.Tensor,
    vlm: CornerMaskedVlm,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Same batch and generator order as `noise_fields_seeds.rank_direction_cached` (donor + flip draws); only the added field changes.
    kind: "original" (donor field, check), "same" (the slice's own field, not flipped), "white" (donor std x white Gaussian;
    flipped). The flip draw is consumed in every branch, so the generator order stays the same.
    """
    generator = torch.Generator().manual_seed(RENOISE_SEED)
    white = torch.Generator().manual_seed(WHITE_GEN_SEED)
    differences = []
    with torch.no_grad():
        for start in range(0, len(noisy), RANK_BATCH):
            batch = noisy[start : start + RANK_BATCH]
            donors = torch.randint(0, len(fields), (len(batch),), generator=generator)  # same order as `renoised`
            flip = torch.rand(len(batch), generator=generator) < 0.5
            if kind == "original":
                added = fields[donors].clone()
            elif kind == "same":
                added = fields[start : start + len(batch)].clone()
            else:
                std = fields[donors].flatten(1).std(dim=1).view(-1, *[1] * (fields.dim() - 1))
                added = std * torch.randn(fields[donors].shape, generator=white, dtype=fields.dtype)
            if kind != "same":  # the own field is not flipped, so it stays aligned
                added[flip] = added[flip].flip(-1)
            copies = (batch + added).clamp(0.0, 1.0)
            differences.append(phi_noisy[start : start + RANK_BATCH] - vlm.encode_image(copies.to(device)).cpu())
    differences = torch.cat(differences).double()
    direction = differences.sum(0) / differences.sum(0).norm()
    return direction, float((differences @ direction).mean())


def w_controls(
    seed: int,
    noisy: torch.Tensor,
    phi_noisy: torch.Tensor,
    stems: list[str],
    vlm: CornerMaskedVlm,
    device: torch.device,
) -> list[dict]:
    """Rebuild w with the two control noise fields of the seed and compare it with the stored w."""
    target = OUT / f"seed{seed}" / "w_controls.csv"
    if target.exists():
        return pd.read_csv(target).to_dict("records")
    key = RankRecipeConfig().input_key
    fields = seed42_fields(stems, key) if seed == 42 else seed_fields(seed, stems, key)
    payload = torch.load(STAGE1_ROOT / "w" / f"n2s_run{seed}_renoise42_masked.pt", map_location="cpu")
    saved = payload["direction"].double()
    rows = []
    for kind in ("original", "same", "white"):
        direction, margin = control_direction(kind, noisy, phi_noisy, fields, vlm, device)
        rows.append(
            {
                "seed": seed,
                "kind": kind,
                "cosine_to_w": float(direction @ saved),
                "margin": margin,
            }
        )
        print(
            f"  seed {seed} w control {kind}: cosine {rows[-1]['cosine_to_w']:.4f}, margin {margin:.5f}",
            flush=True,
        )
    margin_gap = abs(rows[0]["margin"] - float(payload["margin"]))
    if rows[0]["cosine_to_w"] < W_COSINE_MIN or margin_gap > MARGIN_TOL:
        raise RuntimeError(
            f"seed {seed}: rebuilt w does not match the stored w (cosine {rows[0]['cosine_to_w']:.6f}, "
            f"margin difference {margin_gap:.2e})."
        )
    (OUT / f"seed{seed}").mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(rows), target)
    return rows


# ----------------------------------------------------------------------------- validation stage (per seed)


def recipe_of(root: Path, seed: int, sub: str = "") -> dict:
    """Saved greedy recipe of a seed."""
    return read_recipe(root / f"seed{seed}" / sub / "recipes" / "greedy_recipe.json")


def val_stage(
    seed: int, splits: dict[str, list[Path]], vlm: CornerMaskedVlm, device: torch.device
) -> tuple[dict[str, list[str]], list[dict]]:
    """Validation searches and checks for one seed; returns the recipes and the check rows. Saved results are reused."""
    folder = OUT / f"seed{seed}"
    done = folder / "val_done.csv"
    if done.exists():
        return (
            {"no_s2s": read_recipe(folder / "no_s2s" / "recipes" / "greedy_recipe.json")["recipe"]},
            pd.read_csv(folder / "val_checks.csv").to_dict("records"),
        )
    stage1_filters, band_filters, edge_filters, all_filters = (
        Filters(device),
        ZeroMeanLowFilters(device),
        EdgeMaskedFilters(device),
        AllMaskedFilters(device),
    )
    moves = move_names(seed)
    stage1_only = read_recipe(STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json")
    records = {
        "stage2_band_moves": recipe_of(BAND_MOVES_ROOT, seed),
        "stage2_boundary_mask": recipe_of(BOUNDARY_MASK_ROOT, seed),
        "bg_only": recipe_of(SINGLES_ROOT, seed, "bg_only"),
        "ours": recipe_of(REGION_RULE_ROOT, seed),
    }
    ids, base = load_split(seed, "val", splits, device)
    n2s = base["n2s"].cpu().numpy()
    keep_all = 1.0 - with_mask(base, device)  # mask of the method
    keep = {
        "stage2_band_moves": None,
        "stage2_boundary_mask": 1.0 - edge_mask(n2s),
        "bg_only": 1.0 - bg_only_mask(n2s),
        "ours": keep_all,
    }
    filters = {
        "stage2_band_moves": band_filters,
        "stage2_boundary_mask": edge_filters,
        "bg_only": all_filters,
        "ours": all_filters,
    }
    scorer = scorer_for(seed, vlm, device)
    start = apply_recipe(stage1_filters, stage1_only["recipe"], base["n2s"], base)
    checks = [
        {
            "seed": seed,
            "check": "val_score_stage1_only",
            "value": abs(float(scorer(start).mean()) - stage1_only["final_mean_score"]),
            "tolerance": SCORE_TOL,
        }
    ]
    # region-rule variants (no GT)
    variants = {"stage1_only": start.cpu().numpy()}
    for name, record in records.items():
        if keep[name] is not None:
            base["keep"] = torch.from_numpy(keep[name]).to(device)
        image = apply_recipe(filters[name], record["recipe"], start, base)
        checks.append(
            {
                "seed": seed,
                "check": f"val_score_{name}",
                "value": abs(float(scorer(image).mean()) - record["final_mean_score"]),
                "tolerance": SCORE_TOL,
            }
        )
        variants[name] = image.cpu().numpy()
        del image
    base["keep"] = torch.from_numpy(keep_all).to(device)
    signs = region_sign_rows(seed, variants, n2s)
    del variants
    for row in checks:
        row["passed"] = row["value"] <= row["tolerance"]
    if not all(row["passed"] for row in checks):
        raise RuntimeError(
            f"seed {seed}: validation check failed; do not use these results.\n"
            + pd.DataFrame(checks).to_markdown(index=False)
        )
    # stage 2 without S2S (no GT)
    no_s2s_moves = [move for move in moves if not move.startswith("to_s2s_")]
    if len(no_s2s_moves) != len(moves) - 3:
        raise RuntimeError("number of S2S moves is not 3.")
    no_s2s = greedy(folder / "no_s2s", no_s2s_moves, start, base, all_filters, scorer)
    # path checks (validation GT only here, for measurement)
    clean = val_clean(splits, ids, device)
    stage1_steps = (
        pd.read_csv(STAGE1_ROOT / f"recipe_seed{seed}" / "steps" / "all_slices.csv")
        if (STAGE1_ROOT / f"recipe_seed{seed}" / "steps" / "all_slices.csv").exists()
        else None
    )
    stage2_steps = pd.read_csv(REGION_RULE_ROOT / f"seed{seed}" / "steps" / "greedy_recipe.csv")
    path1, miss1 = path_rows(
        seed,
        "stage1",
        stage1_only["recipe"],
        list(STAGE1_MOVES),
        base["n2s"],
        base,
        stage1_filters,
        clean,
        stage1_steps,
    )
    path2, miss2 = path_rows(
        seed,
        "stage2",
        records["ours"]["recipe"],
        moves,
        start,
        base,
        all_filters,
        clean,
        stage2_steps,
    )
    # Only checks that recipe JSON and step CSV come from the same run; the final score checks verify the path itself
    checks += [
        {
            "seed": seed,
            "check": "stage1_recipe_json_vs_steps_csv",
            "value": float(miss1),
            "tolerance": 0.0,
            "passed": miss1 == 0 or stage1_steps is None,
        },
        {
            "seed": seed,
            "check": "stage2_recipe_json_vs_steps_csv",
            "value": float(miss2),
            "tolerance": 0.0,
            "passed": miss2 == 0,
        },
    ]
    if not all(row["passed"] for row in checks):
        raise RuntimeError(
            f"seed {seed}: path record does not match; do not use these results.\n"
            + pd.DataFrame(checks).to_markdown(index=False)
        )
    del clean, base, start
    torch.cuda.empty_cache()
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_csv(signs, folder / "region_signs.csv")
    _atomic_csv(pd.concat([path1, path2]), folder / "path_checks.csv")
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "recipe": "no_s2s",
                    "n": len(no_s2s),
                    "steps": " ".join(no_s2s),
                }
            ]
        ),
        folder / "recipes.csv",
    )
    _atomic_csv(pd.DataFrame(checks), folder / "val_checks.csv")
    _atomic_csv(pd.DataFrame([{"seed": seed}]), done)  # written last (resume)
    print(signs.round(5).to_markdown(index=False), flush=True)
    return {"no_s2s": no_s2s}, checks


# ----------------------------------------------------------------------------- test stage (GT for evaluation only)


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
    ours = recipe_of(REGION_RULE_ROOT, seed)["recipe"]
    ids, base = load_split(seed, "test", splits, device)
    with_mask(base, device)
    stage1_only_test = apply_recipe(stage1_filters, stage1_only, base["n2s"], base)
    outputs = {
        "n2s": base["n2s"],
        "equal_n2s_n2n": (0.5 * base["n2s"] + 0.5 * base["ne2ne"]).clamp(0.0, 1.0),
        "stage1_only": stage1_only_test,
        "ours": apply_recipe(all_filters, ours, stage1_only_test, base),
        "no_s2s": apply_recipe(all_filters, recipes["no_s2s"], stage1_only_test, base),
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
    reference = pd.read_csv(JUDGES_ROOT / f"seed{seed}" / "test_per_slice.csv").set_index("slice_id").loc[ids]
    checks = []
    for name in PARITY:
        worst = float(np.abs(frame[f"psnr_{name}"].to_numpy() - reference[f"psnr_{name}"].to_numpy()).max())
        checks.append(
            {
                "seed": seed,
                "check": f"test_psnr_{name}_vs_compare_judges",
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
    """Input check inside the script, including the N2S checkpoints."""
    paths = [
        STAGE1_ROOT / "train_noisy_embeddings_masked.npy",
        *(checkpoint_path(seed) for seed in (123, 456)),
    ]
    for seed in SEEDS:
        paths += [
            STAGE1_ROOT / "w" / f"n2s_run{seed}_renoise42_masked.pt",
            STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json",
            STAGE1_ROOT / f"recipe_seed{seed}" / "steps" / "all_slices.csv",
            *(
                root / f"seed{seed}" / "recipes" / "greedy_recipe.json"
                for root in (BAND_MOVES_ROOT, BOUNDARY_MASK_ROOT, REGION_RULE_ROOT)
            ),
            REGION_RULE_ROOT / f"seed{seed}" / "steps" / "greedy_recipe.csv",
            SINGLES_ROOT / f"seed{seed}" / "bg_only" / "recipes" / "greedy_recipe.json",
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
    vlm = CornerMaskedVlm(load_vlm("medsiglip", device))
    lpips_metric = LpipsMetric(device)
    OUT.mkdir(parents=True, exist_ok=True)
    frames, checks, controls = [], [], []
    with GpuMonitor() as monitor:
        # w controls: same setting as the w computation (TF32 off)
        if not all((OUT / f"seed{seed}" / "w_controls.csv").exists() for seed in SEEDS):
            _set_tf32(False)
            config = RankRecipeConfig()
            stems = [path.stem for path in splits["train_noisy"]]
            noisy = load_train_noisy(splits["train_noisy"], config.input_key, config.read_threads)
            phi_noisy = masked_noisy_embeddings(noisy, stems, vlm, device)
            for seed in SEEDS:
                controls += w_controls(seed, noisy, phi_noisy, stems, vlm, device)
            del noisy, phi_noisy
        else:
            controls = [
                row for seed in SEEDS for row in pd.read_csv(OUT / f"seed{seed}" / "w_controls.csv").to_dict("records")
            ]
        _atomic_csv(pd.DataFrame(controls), OUT / "w_controls.csv")
        _set_tf32(True)  # search / score: TF32 on
        for seed in SEEDS:
            recipes, val_checks = val_stage(seed, splits, vlm, device)
            frame, test_checks = test_stage(seed, splits, recipes, lpips_metric, device)
            frames.append(frame)
            checks += val_checks + test_checks
            _atomic_csv(summarise(pd.concat(frames)), OUT / "summary.csv")  # after every seed
            _atomic_csv(pd.DataFrame(checks), OUT / "checks.csv")
            for name in ("region_signs", "path_checks", "recipes"):
                _atomic_csv(
                    pd.concat([pd.read_csv(OUT / f"seed{s}" / f"{name}.csv") for s in SEEDS[: SEEDS.index(seed) + 1]]),
                    OUT / f"{name}.csv",
                )
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
