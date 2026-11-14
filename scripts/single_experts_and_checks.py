# scripts/single_experts_and_checks.py
"""Single-expert rows, judge-free band moves, score checks and the background-only mask (GPU, no training).

- Single experts and blends: the 7 experts one by one (N2S, N2N, wavelet, N2V, R2R, S2S, B2U) + equal blend of all 7 +
  judge-free band moves: x0 = 0.5 N2S + 0.5 N2N, each of the 15 expert band moves applied once without a judge, same mask
  as the method (edge + outside body).
- Score checks: (a) cosine of each phi(y_i) - phi(y_i^+) difference with w (w is rebuilt with the same generator order and
  checked against the stored w); (b) per-slice Spearman(s, PSNR) on validation slices, at method level (11 candidates:
  7 experts + equal blend (0.5 N2S + 0.5 N2N) + equal blend of 7 + stage 1 + Rank2Denoise) and at search level (the 20
  candidate moves of the first stage-2 step). Validation GT is read only for this measurement; it is never used for selection.
- Background-only mask: m = outside body only (Gaussian sigma 2), no edge mask; stage 2 is searched again from the same
  stage-1 output (`stage2_band_moves.greedy`, `stage2_region_rule.AllMaskedFilters`, masked per-seed w), 3 seeds.
Definitions: judge-free band moves follow the `move_names` order. Spearman ties use average ranks.
Checks (the run stops if one fails): rebuilt w vs stored masked w (cosine >= 0.9999, margin <= 1e-3); mean validation
  score of the 20 moves vs the first-step record of `stage2_region_rule` (<= 1e-4); Rank2Denoise validation score vs `stage2_region_rule`
  `val_vlm_final` (<= 1e-4); background-only x0 score vs stage 1 (<= 1e-4); test PSNR of N2S / equal / stage 1 /
  Rank2Denoise / val.-PSNR judge vs `compare_judges` (per slice, <= 0.02 dB).

Usage (cwd = repository root):  python -m scripts.single_experts_and_checks
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from scripts.compare_judges import OUT as JUDGES_ROOT
from scripts.compare_judges import val_clean
from scripts.expert_outputs import split_files
from scripts.infer_n2s_n2n import RankRecipeConfig, _atomic_csv, _set_tf32
from scripts.noise_fields_seeds import (
    FIELD_CHUNK,
    RANK_BATCH,
    RENOISE_SEED,
    SEED42_FIELDS,
    checkpoint_path,
    load_train_noisy,
    seed42_fields,
)
from scripts.region_maps import region_stack
from scripts.stage1_search import OUT_ROOT as STAGE1_ROOT
from scripts.stage1_search import CornerMaskedVlm, masked_noisy_embeddings, seed_fields
from scripts.stage2_band_moves import apply_recipe, greedy, read_recipe
from scripts.stage2_boundary_mask import MASK_SIGMA, slice_metrics
from scripts.stage2_moves import (
    BANDS,
    STAGE1_MOVES,
    Filters,
    GpuMonitor,
    expert_keys,
    load_split,
    move_names,
    scorer_for,
)
from scripts.stage2_region_rule import OUT as REGION_RULE_ROOT
from scripts.stage2_region_rule import AllMaskedFilters, outside_body, with_mask
from src.metrics import LpipsMetric
from src.paths import RESULTS_ROOT, noise_field_dir
from src.renoise import renoised
from src.vlm import load_vlm

OUT = RESULTS_ROOT / "single_experts_and_checks"
SEEDS = (42, 123, 456)
SCORE_TOL = 1e-4
PARITY_DB = 0.02
W_COSINE_MIN = 0.9999
MARGIN_TOL = 1e-3
MASK_MIN = 0.9
PARITY = (
    ("n2s", "n2s"),
    ("equal_n2s_n2n", "equal_n2s_n2n"),
    ("stage1_only", "stage1_only"),
    ("ours", "ours"),
    ("psnr_sel", "psnr_sel"),
)


def short(key: str) -> str:
    """'n2v_seed42' -> 'n2v' (column names do not depend on the seed)."""
    return key.split("_seed")[0]


def expert_pool(seed: int) -> list[str]:
    """The 7 experts of the paper: N2S (n2s), N2N (ne2ne) + the 5 experts of the band moves."""
    return ["n2s", "ne2ne", *expert_keys(seed)]


def stage2_moves(seed: int) -> list[str]:
    """Moves of stage 2 that are not stage-1 moves."""
    return [move for move in move_names(seed) if move not in STAGE1_MOVES]


def bg_only_mask(n2s: np.ndarray) -> np.ndarray:
    """m = clip(Gauss_sigma2(outside body)): the background part of `stage2_region_rule.air_edge_mask`, without the edge part."""
    outside = outside_body(n2s).astype(np.float64)
    return np.clip(np.stack([gaussian_filter(image, MASK_SIGMA) for image in outside]), 0.0, 1.0)


def stage_recipes(seed: int) -> dict[str, list[str]]:
    """Stored recipes of the seed: stage 1 only, Rank2Denoise, and the two stages of the validation-PSNR judge."""
    psnr_row = pd.read_csv(JUDGES_ROOT / f"seed{seed}" / "recipes.csv").set_index("judge").loc["psnr"]

    def steps(value: object) -> list[str]:  # an empty recipe is NaN in the CSV
        """Moves of a recipe cell; an empty cell gives an empty list."""
        return [] if pd.isna(value) else str(value).split()

    return {
        "stage1_only": read_recipe(STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json")["recipe"],
        "ours": read_recipe(REGION_RULE_ROOT / f"seed{seed}" / "recipes" / "greedy_recipe.json")["recipe"],
        "psnr_stage1": steps(psnr_row["stage1"]),
        "psnr_stage2": steps(psnr_row["stage2"]),
    }


def required_inputs() -> list[Path]:
    """Input files that must exist before the run starts."""
    stems_count = 647
    paths = [
        STAGE1_ROOT / "train_noisy_embeddings_masked.npy",
        STAGE1_ROOT / "train_noisy_slice_ids.json",
        *(checkpoint_path(seed) for seed in (123, 456)),
        *(SEED42_FIELDS / f"chunk_{start:04d}.pt" for start in range(0, stems_count, FIELD_CHUNK)),
        *(
            noise_field_dir(seed) / f"chunk_{start:04d}.pt"
            for seed in (123, 456)
            for start in range(0, stems_count, FIELD_CHUNK)
        ),
    ]
    for seed in SEEDS:
        paths += [
            STAGE1_ROOT / "w" / f"n2s_run{seed}_renoise42_masked.pt",
            STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json",
            REGION_RULE_ROOT / f"seed{seed}" / "recipes" / "greedy_recipe.json",
            REGION_RULE_ROOT / f"seed{seed}" / "steps" / "greedy_recipe.csv",
            REGION_RULE_ROOT / f"seed{seed}" / "recipe.csv",
            JUDGES_ROOT / f"seed{seed}" / "recipes.csv",
            JUDGES_ROOT / f"seed{seed}" / "test_per_slice.csv",
        ]
    return paths


# ----------------------------------------------------------------------------- cosine of the differences with w


def w_alignment(
    seed: int,
    noisy: torch.Tensor,
    phi_noisy: torch.Tensor,
    stems: list[str],
    vlm: CornerMaskedVlm,
    device: torch.device,
) -> dict:
    """Same as `noise_fields_seeds.rank_direction_cached` (batch 16, generator RENOISE_SEED, same fields); the differences are kept."""
    target = OUT / f"seed{seed}" / "w_alignment_pairs.csv"
    if target.exists():
        return pd.read_csv(OUT / f"seed{seed}" / "w_alignment_summary.csv").iloc[0].to_dict()
    key = RankRecipeConfig().input_key
    fields = seed42_fields(stems, key) if seed == 42 else seed_fields(seed, stems, key)
    generator = torch.Generator().manual_seed(RENOISE_SEED)
    differences = []
    with torch.no_grad():
        for start in range(0, len(noisy), RANK_BATCH):
            copies = renoised(noisy[start : start + RANK_BATCH], fields, generator)
            differences.append(phi_noisy[start : start + RANK_BATCH] - vlm.encode_image(copies.to(device)).cpu())
    differences = torch.cat(differences).double()
    direction = differences.sum(0) / differences.sum(0).norm()
    saved = torch.load(STAGE1_ROOT / "w" / f"n2s_run{seed}_renoise42_masked.pt", map_location="cpu")
    cosine_w = float(direction @ saved["direction"].double())
    margin = float((differences @ direction).mean())
    if cosine_w < W_COSINE_MIN or abs(margin - float(saved["margin"])) > MARGIN_TOL:
        raise RuntimeError(
            f"seed {seed}: rebuilt w does not match the stored w (cosine {cosine_w:.6f}, margin "
            f"{margin:.6f} ↔ {float(saved['margin']):.6f}); do not use these results."
        )
    w = saved["direction"].double()
    projection = differences @ w
    cosine = projection / differences.norm(dim=1)
    pairs = pd.DataFrame(
        {
            "slice_id": stems,
            "projection_on_w": projection.numpy(),
            "cosine_to_w": cosine.numpy(),
        }
    )
    summary = {
        "seed": seed,
        "n_pairs": len(stems),
        "noise_fields_seeds_cosine": cosine_w,
        "margin": margin,
        "frac_projection_positive": float((projection > 0).double().mean()),
        "cosine_mean": float(cosine.mean()),
    }
    (OUT / f"seed{seed}").mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame([summary]), OUT / f"seed{seed}" / "w_alignment_summary.csv")
    _atomic_csv(pairs, target)
    print(f"seed {seed} cosine of differences with w: {summary}", flush=True)
    return summary


# ----------------------------------------------------------------------------- score checks + background-only search (validation)


def psnr_per_slice(images: torch.Tensor, clean: torch.Tensor) -> np.ndarray:
    """PSNR of every slice with data range 1."""
    return (10.0 * torch.log10(1.0 / ((images - clean) ** 2).mean(dim=(1, 2)))).cpu().numpy()


def spearman_rows(
    scores: np.ndarray,
    psnrs: np.ndarray,
    level: str,
    seed: int,
    ids: list[str],
    names: list[str],
) -> tuple[pd.DataFrame, dict]:
    """scores, psnrs: (candidate, slice). Per-slice Spearman over candidates; summary = median over slices."""
    rows = []
    for k, slice_id in enumerate(ids):
        rho = spearmanr(scores[:, k], psnrs[:, k]).statistic
        rows.append({"seed": seed, "level": level, "slice_id": slice_id, "spearman": float(rho)})
    frame = pd.DataFrame(rows)
    summary = {
        "seed": seed,
        "level": level,
        "n_candidates": len(names),
        "n_slices": len(ids),
        "spearman_median": float(frame.spearman.median()),
    }
    return frame, summary


def val_stage(
    seed: int, splits: dict[str, list[Path]], vlm: CornerMaskedVlm, device: torch.device
) -> tuple[list[str], list[dict]]:
    """Score checks (method and search level) and the background-only search. Returns the recipe and the check rows."""
    folder = OUT / f"seed{seed}"
    done = folder / "score_vs_psnr_summary.csv"
    recipe_file = folder / "bg_only" / "recipes" / "greedy_recipe.json"
    if done.exists() and recipe_file.exists():
        return read_recipe(recipe_file)["recipe"], pd.read_csv(folder / "val_checks.csv").to_dict("records")
    recipes = stage_recipes(seed)
    stage1_filters, masked_filters = Filters(device), AllMaskedFilters(device)
    ids, base = load_split(seed, "val", splits, device)
    edge_air = with_mask(base, device)  # mask of the method (edge + outside body)
    scorer = scorer_for(seed, vlm, device)
    clean = val_clean(splits, ids, device)  # validation GT: measurement only (Spearman); never used for selection
    start = apply_recipe(stage1_filters, recipes["stage1_only"], base["n2s"], base)
    checks = []
    # search level: the 20 candidates of the first stage-2 step (x0 = stage-1 output)
    moves = move_names(seed)
    recorded = pd.read_csv(REGION_RULE_ROOT / f"seed{seed}" / "steps" / "greedy_recipe.csv").iloc[0]
    search_scores, search_psnr, worst = [], [], 0.0
    for move in moves:
        candidate = masked_filters.apply(move, start, base)
        values = scorer(candidate).cpu().numpy()
        search_scores.append(values)
        search_psnr.append(psnr_per_slice(candidate, clean))
        worst = max(worst, abs(float(values.mean()) - float(recorded[f"mean_{move}"])))
        del candidate
    checks.append(
        {
            "seed": seed,
            "check": "score_vs_psnr_step1_means_vs_stage2_region_rule",
            "value": worst,
            "tolerance": SCORE_TOL,
            "passed": worst <= SCORE_TOL,
        }
    )
    # method level: 7 experts + equal + stage 1 + Rank2Denoise
    ours = apply_recipe(masked_filters, recipes["ours"], start, base)
    method_images = {
        **{short(key): base[key] for key in expert_pool(seed)},
        "equal_n2s_n2n": (0.5 * base["n2s"] + 0.5 * base["ne2ne"]).clamp(0.0, 1.0),
        "equal7": torch.stack([base[key] for key in expert_pool(seed)]).mean(0),
        "stage1_only": start,
        "ours": ours,
    }
    method_scores = {name: scorer(image).cpu().numpy() for name, image in method_images.items()}
    method_psnr = {name: psnr_per_slice(image, clean) for name, image in method_images.items()}
    ours_gap = abs(
        float(method_scores["ours"].mean())
        - float(pd.read_csv(REGION_RULE_ROOT / f"seed{seed}" / "recipe.csv").iloc[0]["val_vlm_final"])
    )
    checks.append(
        {
            "seed": seed,
            "check": "ours_val_score_vs_stage2_region_rule",
            "value": ours_gap,
            "tolerance": SCORE_TOL,
            "passed": ours_gap <= SCORE_TOL,
        }
    )
    if not all(row["passed"] for row in checks):
        raise RuntimeError(
            f"seed {seed}: validation check failed; do not use these results.\n"
            + pd.DataFrame(checks).to_markdown(index=False)
        )
    names = list(method_images)
    _, sum_m = spearman_rows(
        np.stack([method_scores[n] for n in names]),
        np.stack([method_psnr[n] for n in names]),
        "method",
        seed,
        ids,
        names,
    )
    _, sum_s = spearman_rows(np.stack(search_scores), np.stack(search_psnr), "search_step1", seed, ids, moves)
    del clean, method_images, ours
    torch.cuda.empty_cache()
    # background-only masked stage-2 search (no GT)
    bg = bg_only_mask(base["n2s"].cpu().numpy())
    base["keep"] = torch.from_numpy(1.0 - bg).to(device)
    n2s_val = base["n2s"].cpu().numpy()
    edge = region_stack(n2s_val)["body_edge"]
    outside = outside_body(n2s_val)
    start_score = float(scorer(start).mean())
    stage1_only_score = read_recipe(STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json")[
        "final_mean_score"
    ]
    checks += [
        {
            "seed": seed,
            "check": "bgmask_start_score_vs_stage1_only",
            "value": abs(start_score - stage1_only_score),
            "tolerance": SCORE_TOL,
            "passed": abs(start_score - stage1_only_score) <= SCORE_TOL,
        },
        {
            "seed": seed,
            "check": "bgmask_mask_mean_outside_min",
            "value": float(bg[outside].mean()),
            "tolerance": MASK_MIN,
            "passed": float(bg[outside].mean()) >= MASK_MIN,
        },
    ]
    print(
        pd.DataFrame(checks).to_markdown(index=False),
        f"\n  info: mean background-only mask on edge pixels {float(bg[edge].mean()):.3f} "
        f"(method mask {float(edge_air[edge].mean()):.3f})",
        flush=True,
    )
    if not all(row["passed"] for row in checks):
        raise RuntimeError(f"seed {seed}: background-only check failed; search not started.")
    recipe = greedy(folder / "bg_only", moves, start, base, masked_filters, scorer)
    final_score = float(scorer(apply_recipe(masked_filters, recipe, start, base)).mean())
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "n_steps": len(recipe),
                    "val_vlm_start": start_score,
                    "val_vlm_final": final_score,
                    "recipe": " ".join(recipe),
                }
            ]
        ),
        folder / "bg_only_recipe.csv",
    )
    _atomic_csv(pd.DataFrame(checks), folder / "val_checks.csv")
    _atomic_csv(pd.DataFrame([sum_m, sum_s]), done)  # written last (resume)
    print(
        f"seed {seed} score checks: {sum_m}\n  {sum_s}\n  background-only recipe {len(recipe)} steps",
        flush=True,
    )
    del base, start
    torch.cuda.empty_cache()
    return recipe, checks


# ----------------------------------------------------------------------------- single experts + background-only mask (test)


def test_stage(
    seed: int,
    splits: dict[str, list[Path]],
    bg_recipe: list[str],
    lpips_metric: LpipsMetric,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict]]:
    """Test evaluation of the recipes for one seed; returns the per-slice test table and the check rows. Saved results are reused."""
    folder = OUT / f"seed{seed}"
    target = folder / "test_per_slice.csv"
    if target.exists():
        return pd.read_csv(target), pd.read_csv(folder / "test_checks.csv").to_dict("records")
    recipes = stage_recipes(seed)
    stage1_filters, masked_filters = Filters(device), AllMaskedFilters(device)
    ids, base = load_split(seed, "test", splits, device)
    with_mask(base, device)  # mask of the method
    equal = (0.5 * base["n2s"] + 0.5 * base["ne2ne"]).clamp(0.0, 1.0)
    stage1_only = apply_recipe(stage1_filters, recipes["stage1_only"], base["n2s"], base)
    psnr_stage1 = apply_recipe(stage1_filters, recipes["psnr_stage1"], base["n2s"], base)
    outputs = {
        "noisy": base["noisy"],
        **{short(key): base[key] for key in expert_pool(seed)},
        "equal_n2s_n2n": equal,
        "equal7": torch.stack([base[key] for key in expert_pool(seed)]).mean(0),
        "judgefree_band": apply_recipe(masked_filters, stage2_moves(seed), equal, base),
        "stage1_only": stage1_only,
        "ours": apply_recipe(masked_filters, recipes["ours"], stage1_only, base),
        "psnr_sel": apply_recipe(masked_filters, recipes["psnr_stage2"], psnr_stage1, base),
    }
    base["keep"] = torch.from_numpy(1.0 - bg_only_mask(base["n2s"].cpu().numpy())).to(device)  # background-only mask
    outputs["bg_only"] = apply_recipe(masked_filters, bg_recipe, stage1_only, base)
    clean = base["clean"].cpu().numpy()
    images = {name: value.cpu().numpy() for name, value in outputs.items()}
    del base, outputs, equal, stage1_only, psnr_stage1
    torch.cuda.empty_cache()
    masks = region_stack(images["n2s"])
    frame = pd.concat(
        [
            pd.DataFrame({"seed": seed, "slice_id": ids}),
            *(slice_metrics(name, images[name], images["n2s"], clean, masks, lpips_metric) for name in images),
        ],
        axis=1,
    )
    reference = pd.read_csv(JUDGES_ROOT / f"seed{seed}" / "test_per_slice.csv").set_index("slice_id").loc[ids]
    checks = []
    for name, column in PARITY:
        worst = float(np.abs(frame[f"psnr_{name}"].to_numpy() - reference[f"psnr_{column}"].to_numpy()).max())
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
    _atomic_csv(frame, target)  # written last: its presence marks the run as done (resume)
    return frame, checks


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per seed and method."""
    methods = [c[len("psnr_") :] for c in frame.columns if c.startswith("psnr_")]
    rows = []
    for seed, block in frame.groupby("seed"):
        for method in methods:
            row = {"seed": seed, "method": method}
            for metric in ("psnr", "ssim", "lpips", "rh", "shift_air"):
                column = f"{metric}_{method}"
                row[metric] = block[column].mean() if column in block else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Run all seeds and write the per-slice, summary and check tables."""
    began = time.time()
    device = torch.device("cuda")
    missing = [str(path) for path in required_inputs() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs; nothing was run:\n  " + "\n  ".join(missing[:20]))
    for band in BANDS:  # judge-free band moves: 15 = 5 experts x 3 bands
        if sum(move.endswith(f"_{band}") for move in stage2_moves(42)) != 5:
            raise RuntimeError("unexpected number of band moves.")
    splits = split_files()
    vlm = CornerMaskedVlm(load_vlm("medsiglip", device))
    lpips_metric = LpipsMetric(device)
    OUT.mkdir(parents=True, exist_ok=True)
    frames, checks, w_alignment_rows = [], [], []
    with GpuMonitor() as monitor:
        # w rebuild with the same setting (TF32 off)
        if not all((OUT / f"seed{seed}" / "w_alignment_pairs.csv").exists() for seed in SEEDS):
            _set_tf32(False)
            config = RankRecipeConfig()
            stems = [path.stem for path in splits["train_noisy"]]
            noisy = load_train_noisy(splits["train_noisy"], config.input_key, config.read_threads)
            phi_noisy = masked_noisy_embeddings(noisy, stems, vlm, device)
            for seed in SEEDS:
                w_alignment_rows.append(w_alignment(seed, noisy, phi_noisy, stems, vlm, device))
            del noisy, phi_noisy
        else:
            w_alignment_rows = [
                pd.read_csv(OUT / f"seed{seed}" / "w_alignment_summary.csv").iloc[0].to_dict() for seed in SEEDS
            ]
        _atomic_csv(pd.DataFrame(w_alignment_rows), OUT / "w_alignment_summary.csv")
        _set_tf32(True)  # search / score: TF32 on
        score_vs_psnr = []
        for seed in SEEDS:
            recipe, val_checks = val_stage(seed, splits, vlm, device)
            frame, test_checks = test_stage(seed, splits, recipe, lpips_metric, device)
            frames.append(frame)
            checks += val_checks + test_checks
            score_vs_psnr.append(pd.read_csv(OUT / f"seed{seed}" / "score_vs_psnr_summary.csv"))
            _atomic_csv(summarise(pd.concat(frames)), OUT / "summary.csv")  # after every seed
            _atomic_csv(pd.DataFrame(checks), OUT / "checks.csv")
            _atomic_csv(pd.concat(score_vs_psnr), OUT / "score_vs_psnr_summary.csv")
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
    view = summarise(pd.concat(frames)).pivot_table(index="method", columns="seed", values="psnr")
    view["mean"] = view.mean(axis=1)
    print(view.round(3).to_markdown(), flush=True)


if __name__ == "__main__":
    main()
