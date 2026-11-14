"""Numbers and tables of the paper. No training, no GPU; CSV files on disk and one checkpoint (parameter count) are read.

Inputs (under `<results_root>`, the R2D_RESULTS folder of the runs):
  compare_judges/  -- main methods and judges (`scripts/compare_judges.py`)
  train/supervised_seed42/metrics.csv -- supervised reference row (`scripts/train_seed42.py`)
  stage2_region_rule/, stage2_band_moves/, stage2_boundary_mask/ -- ablation
  single_experts_and_checks/   -- single experts, equal blend of 7, judge-free band moves, score checks,
                                background-only mask (`scripts/single_experts_and_checks.py`)
  judge_resnet50/    -- ResNet-50 direction (`scripts/judge_resnet50.py`)
  region_and_w_checks/    -- region-rule signs, path and w checks, search without S2S (`scripts/region_and_w_checks.py`)
  judge_residual_variance/ -- residual-variance judge (`scripts/judge_residual_variance.py`)
  stage-1 and stage-2 step CSVs (move limit), stage-2 recipes of Rank2Denoise (moves chosen), cost.csv of
  stage1_search and stage2_region_rule (run time), train/n2s_reference_seed42/best_model.pth (U-Net parameters)
Outputs (under `<output_dir>`): data/paper_numbers.csv, figures/table_main.tex, figures/table_ablation.tex.
Statistics: (1) patient-level exact Wilcoxon signed-rank test (`method="exact"`, 31 patients; differences are averaged
over seeds per slice, then per patient); (2) patient-cluster bootstrap 95% CI, 5000 draws, RNG 0.
Usage: python paper/make_tables.py <results_root> <output_dir>
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

results, art = (Path(p) for p in sys.argv[1:3])
for sub in ("data", "figures"):
    (art / sub).mkdir(parents=True, exist_ok=True)
refs = results / "compare_judges"
singles = results / "single_experts_and_checks"
resnet_dir = results / "judge_resnet50"
summary = pd.read_csv(refs / "summary.csv")
per_slice = pd.read_csv(refs / "test_per_slice_all.csv")
singles_summary = pd.read_csv(singles / "summary.csv")
singles_slices = pd.read_csv(singles / "test_per_slice_all.csv")
resnet_dir_summary = pd.read_csv(resnet_dir / "summary.csv")
air = pd.read_csv(results / "stage2_region_rule" / "summary.csv")
band_moves = pd.read_csv(results / "stage2_band_moves" / "summary.csv")
edge = pd.read_csv(results / "stage2_boundary_mask" / "summary.csv")
region_w = results / "region_and_w_checks"
resvar = results / "judge_residual_variance"
region_w_summary, resvar_summary = (
    pd.read_csv(region_w / "summary.csv"),
    pd.read_csv(resvar / "summary.csv"),
)
region_w_slices, resvar_slices = (
    pd.read_csv(region_w / "test_per_slice_all.csv"),
    pd.read_csv(resvar / "test_per_slice_all.csv"),
)
rows: list[dict] = []


def seed_table(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    """Per-seed rows of one method (indexed by seed)."""
    part = frame[frame.method == method].set_index("seed").sort_index()
    if len(part) != 3:
        raise SystemExit(f"{method}: three seeds missing")
    return part


def fmt(value: float, digits: int) -> str:
    """Number with a fixed count of decimals."""
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------- consistency checks
a = seed_table(air, "stage2_region_rule")
o = seed_table(summary, "ours")
worst = float(max((a[m] - o[m]).abs().max() for m in ("psnr", "ssim", "lpips")))
if worst > 1e-9:
    raise SystemExit(f"ours does not match the stage2_region_rule run: {worst}")
rows.append({"key": "parity_ours_vs_stage2_region_rule_run", "value": worst})
# Single-expert tables give the same three metrics per seed as compare_judges for shared methods.
for frame, name in (
    (singles_summary, "single_experts_and_checks"),
    (resnet_dir_summary, "judge_resnet50"),
    (region_w_summary, "region_and_w_checks"),
    (resvar_summary, "judge_residual_variance"),
):
    for method in ("n2s", "equal_n2s_n2n", "stage1_only", "ours"):
        gap = float(
            max(
                (seed_table(frame, method)[m] - seed_table(summary, method)[m]).abs().max()
                for m in ("psnr", "ssim", "lpips")
            )
        )
        if gap > 1e-6:
            raise SystemExit(f"{name} {method} does not match compare_judges: {gap}")
        rows.append({"key": f"parity_{name}_{method}", "value": gap})
same = singles_slices.set_index(["seed", "slice_id"]).psnr_ours - per_slice.set_index(["seed", "slice_id"]).psnr_ours
if float(same.abs().max()) > 1e-9:
    raise SystemExit("single_experts_and_checks per-slice PSNR does not match compare_judges")
# Per-slice table: single experts + search without S2S + residual-variance judge; the Rank2Denoise columns must match.
key = ["seed", "slice_id"]
slices = singles_slices.set_index(key)
for frame, method in ((region_w_slices, "no_s2s"), (resvar_slices, "resvar_sel")):
    other = frame.set_index(key)
    if float((other.psnr_ours - slices.psnr_ours).abs().max()) > 1e-9:
        raise SystemExit(f"{method}: per-slice PSNR of ours does not match single_experts_and_checks")
    for metric in ("psnr", "ssim", "lpips", "shift_air"):
        slices[f"{metric}_{method}"] = other[f"{metric}_{method}"]
slices = slices.reset_index()

# All methods in one table (seed, method) -> metrics.
pool = pd.concat(
    [
        summary[["seed", "method", "psnr", "ssim", "lpips", "shift_air"]],
        singles_summary[~singles_summary.method.isin(summary.method)][
            ["seed", "method", "psnr", "ssim", "lpips", "shift_air"]
        ],
        resnet_dir_summary[resnet_dir_summary.method == "resnet_sel"][
            ["seed", "method", "psnr", "ssim", "lpips", "shift_air"]
        ],
        region_w_summary[region_w_summary.method == "no_s2s"][["seed", "method", "psnr", "ssim", "lpips", "shift_air"]],
        resvar_summary[resvar_summary.method == "resvar_sel"][["seed", "method", "psnr", "ssim", "lpips", "shift_air"]],
    ]
)

# ---------------------------------------------------------------- main table
ONE_RUN = {
    "ne2ne",
    "s2s",
    "wavelet",
}  # single run / no training: the same output for every seed
BLOCKS = [
    [
        ("n2s", "N2S", "no"),
        ("ne2ne", "N2N$^{1}$", "no"),
        ("n2v", "N2V", "no"),
        ("r2r", "R2R", "no"),
        ("b2u", "B2U", "no"),
        ("s2s", "S2S$^{1}$", "no"),
        ("wavelet", "Wavelet$^{1}$", "no"),
    ],
    [
        ("equal_n2s_n2n", "Equal blend", "no"),
        ("equal7", "Equal blend, 7 experts", "no"),
        ("judgefree_band", "Band moves, no judge", "no"),
        ("stage1_only", "Stage 1 only", "no"),
        ("ours", r"\textbf{Rank2Denoise (Ours)}", "no"),
        ("no_s2s", "Ours without S2S moves", "no"),
    ],
    [
        ("random_sel", "Random direction", "no"),
        ("resnet_sel", "ResNet-50 direction", "no"),
        ("q_sel", r"$Q$ index", "no"),
        ("resvar_sel", "Residual-variance judge", "no"),
    ],
    [("psnr_sel", "Val.-PSNR judge", "val")],
]
# Noisy input: the noisy test slices themselves (same for every seed); supervised: test metrics written by
# `src/train.py` at the end of the supervised run (`scripts/train_seed42.py`).
noisy_row = seed_table(singles_summary, "noisy").mean(numeric_only=True)
supervised = pd.read_csv(results / "train" / "supervised_seed42" / "metrics.csv")
if len(supervised) != 220:
    raise SystemExit(f"supervised metrics: {len(supervised)} slices, expected 220")
lines = [
    r"\begin{tabular}{@{}lccccc@{}}",
    r"\toprule",
    r"Method & GT & PSNR\,$\uparrow$ & Gain & SSIM\,$\uparrow$ & LPIPS\,$\downarrow$ \\",
    r"\midrule",
    f"Noisy input & -- & {fmt(noisy_row.psnr, 2)} & -- & {fmt(noisy_row.ssim, 3)} & {fmt(noisy_row.lpips, 3)} \\\\",
    (
        f"Supervised$^{{1}}$ & train & {fmt(supervised.psnr.mean(), 2)} & -- & {fmt(supervised.ssim.mean(), 3)} & "
        f"{fmt(supervised.lpips.mean(), 3)} \\\\"
    ),
]
equal = seed_table(pool, "equal_n2s_n2n")
for block in BLOCKS:
    lines.append(r"\midrule")
    for method, label, gt in block:
        t = seed_table(pool, method)
        delta = t.psnr - equal.psnr
        psnr = fmt(t.psnr.mean(), 2)
        if method not in ONE_RUN:
            psnr += f"$\\pm${fmt(t.psnr.std(ddof=1), 2)}"
        d = "0" if method == "equal_n2s_n2n" else f"${delta.mean():+.2f}$"
        ssim, lp = fmt(t.ssim.mean(), 3), fmt(t.lpips.mean(), 3)
        if method == "ours":
            psnr, d, lp = rf"\textbf{{{psnr}}}", rf"\textbf{{{d}}}", rf"\textbf{{{lp}}}"
        lines.append(f"{label} & {gt} & {psnr} & {d} & {ssim} & {lp} \\\\")
        for metric in ("psnr", "ssim", "lpips"):
            rows.append({"key": f"mean_{metric}_{method}", "value": float(t[metric].mean())})
            rows.append({"key": f"std_{metric}_{method}", "value": float(t[metric].std(ddof=1))})
        for seed, value in t.psnr.items():
            rows.append({"key": f"psnr_{method}_seed{seed}", "value": float(value)})
lines += [r"\bottomrule", r"\end{tabular}"]
(art / "figures" / "table_main.tex").write_text("\n".join(lines) + "\n")

ours = seed_table(pool, "ours")
psnr_sel = seed_table(pool, "psnr_sel")
share = (ours.psnr - equal.psnr) / (psnr_sel.psnr - equal.psnr)
# One value per seed; the paper gives the average of the three seeds (73%).
rows.append({"key": "share_of_psnr_sel_gain_over_equal", "value": float(share.mean())})
for seed, value in share.items():
    rows.append({"key": f"share_of_psnr_sel_gain_over_equal_seed{seed}", "value": float(value)})
# PSNR gain of Rank2Denoise over the methods named in the text (three-seed mean)
for method in ("n2s", "equal_n2s_n2n", "s2s", "judgefree_band", "resvar_sel"):
    rows.append(
        {
            "key": f"diff_psnr_ours_minus_{method}",
            "value": float((ours.psnr - seed_table(pool, method).psnr).mean()),
        }
    )
# number of seeds with a lower LPIPS than the equal blend
rows.append(
    {
        "key": "seeds_ours_better_lpips_equal_n2s_n2n",
        "value": float((ours.lpips - equal.lpips < 0).sum()),
    }
)

# ---------------------------------------------------------------- patient-level tests


def patient_groups(metric: str, other: str, first: str = "ours") -> tuple[pd.Series, list[np.ndarray]]:
    """Per-slice difference of a metric between two methods and the same differences grouped by patient."""
    diff = (slices[f"{metric}_{first}"] - slices[f"{metric}_{other}"]).groupby(slices.slice_id).mean()
    patient = diff.index.str[:6]
    return diff, [diff.values[patient == p] for p in np.unique(patient)]


rng = np.random.default_rng(0)  # fixed order: CIs vs N2S and vs the equal blend
for other in ("n2s", "equal_n2s_n2n"):
    _, groups = patient_groups("psnr", other)
    draws = np.empty(5000)
    for b in range(5000):
        pick = rng.integers(0, len(groups), len(groups))
        draws[b] = np.concatenate([groups[i] for i in pick]).mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    rows += [
        {"key": f"ci_lo_psnr_ours_minus_{other}", "value": float(lo)},
        {"key": f"ci_hi_psnr_ours_minus_{other}", "value": float(hi)},
    ]
PAIRS = [("ours", o) for o in ("n2s", "equal_n2s_n2n", "psnr_sel", "s2s", "judgefree_band", "resvar_sel")]
PAIRS += [("no_s2s", "equal_n2s_n2n")]
for first, other in PAIRS:
    for metric in ("psnr", "ssim", "lpips"):
        diff, groups = patient_groups(metric, other, first)
        means = np.array([g.mean() for g in groups])
        better = (means < 0) if metric == "lpips" else (means > 0)
        slice_better = (diff < 0) if metric == "lpips" else (diff > 0)
        tag = "ours" if first == "ours" else first
        rows += [
            {
                "key": f"wilcoxon_p_{metric}_{tag}_vs_{other}",
                "value": float(stats.wilcoxon(means, method="exact").pvalue),
            },
            {
                "key": f"patients_{tag}_better_{metric}_{other}",
                "value": float(better.sum()),
            },
            {
                "key": f"slices_{tag}_better_{metric}_{other}",
                "value": float(slice_better.sum()),
            },
        ]
# LPIPS CI (equal blend), separate RNG
lp_rng = np.random.default_rng(0)
_, groups = patient_groups("lpips", "equal_n2s_n2n")
draws = np.array(
    [np.concatenate([groups[i] for i in lp_rng.integers(0, len(groups), len(groups))]).mean() for _ in range(5000)]
)
rows += [
    {
        "key": "ci_lo_lpips_ours_minus_equal_n2s_n2n",
        "value": float(np.percentile(draws, 2.5)),
    },
    {
        "key": "ci_hi_lpips_ours_minus_equal_n2s_n2n",
        "value": float(np.percentile(draws, 97.5)),
    },
    {"key": "n_patients", "value": float(len(groups))},
]

# ---------------------------------------------------------------- score checks
w_alignment = pd.read_csv(singles / "w_alignment_summary.csv")
score_vs_psnr = pd.read_csv(singles / "score_vs_psnr_summary.csv")
rows += [
    {
        "key": "w_alignment_frac_projection_positive_min",
        "value": float(w_alignment.frac_projection_positive.min()),
    },
    {
        "key": "w_alignment_cosine_mean_min",
        "value": float(w_alignment.cosine_mean.min()),
    },
    {
        "key": "w_alignment_cosine_mean_max",
        "value": float(w_alignment.cosine_mean.max()),
    },
    {"key": "w_alignment_n_pairs", "value": float(w_alignment.n_pairs.iloc[0])},
]
for level in ("method", "search_step1"):
    part = score_vs_psnr[score_vs_psnr.level == level]
    for col in ("spearman_median",):
        rows += [
            {
                "key": f"score_vs_psnr_{level}_{col}_min",
                "value": float(part[col].min()),
            },
            {
                "key": f"score_vs_psnr_{level}_{col}_max",
                "value": float(part[col].max()),
            },
        ]
    rows.append(
        {
            "key": f"score_vs_psnr_{level}_n_candidates",
            "value": float(part.n_candidates.iloc[0]),
        }
    )

# ---------------------------------------------------------------- ablation
ABL = [
    (summary, "stage1_only", "Stage 1 only"),
    (band_moves, "stage2_band_moves", "+ band moves"),
    (edge, "stage2_boundary_mask", "+ boundary mask"),
    (summary, "ours", r"+ background (\textbf{Rank2Denoise})"),
    (singles_summary, "bg_only", "Background mask only"),
]
lines = [
    r"\begin{tabular}{@{}lcccc@{}}",
    r"\toprule",
    r"Variant & PSNR\,$\uparrow$ & SSIM\,$\uparrow$ & LPIPS\,$\downarrow$ & Shift\,$\downarrow$ \\",
    r"\midrule",
]
for frame, method, label in ABL:
    t = seed_table(frame, method)
    shift = 1e3 * t.shift_air.abs().mean()
    if method == "bg_only":
        lines.append(r"\midrule")
    lines.append(
        f"{label} & {fmt(t.psnr.mean(), 2)} & {fmt(t.ssim.mean(), 3)} & {fmt(t.lpips.mean(), 3)} & {fmt(shift, 2)} \\\\"
    )
    for metric in ("psnr", "ssim", "lpips"):
        rows.append({"key": f"abl_{metric}_{method}", "value": float(t[metric].mean())})
    rows.append({"key": f"abl_abs_shift_air_{method}", "value": float(t.shift_air.abs().mean())})
lines += [r"\bottomrule", r"\end{tabular}"]
(art / "figures" / "table_ablation.tex").write_text("\n".join(lines) + "\n")

# ---------------------------------------------------------------- region signs, path checks, w controls
m1 = pd.read_csv(region_w / "region_signs.csv")
for variant, part in m1.groupby("variant"):
    rows += [
        {
            "key": f"region_signs_abs_shift_air_{variant}_min",
            "value": float(part.abs_shift_air_mean.min()),
        },
        {
            "key": f"region_signs_abs_shift_air_{variant}_max",
            "value": float(part.abs_shift_air_mean.max()),
        },
        {
            "key": f"region_signs_frac_blurred_{variant}_min",
            "value": float(part.frac_slices_edge_blurred.min()),
        },
        {
            "key": f"region_signs_frac_blurred_{variant}_max",
            "value": float(part.frac_slices_edge_blurred.max()),
        },
    ]
path = pd.read_csv(region_w / "path_checks.csv")
for (seed, stage), part in path.groupby(["seed", "stage"]):
    rows += [
        {
            "key": f"path_check_{stage}_raise_frac_seed{seed}",
            "value": float(part.chosen_raises_val_psnr.mean()),
        },
        {
            "key": f"path_check_{stage}_best_is_gauss_frac_seed{seed}",
            "value": float((part.best_by_val_psnr == "gauss").mean()),
        },
        {
            "key": f"path_check_{stage}_chosen_gauss_count_seed{seed}",
            "value": float((part.chosen == "gauss").sum()),
        },
        {
            "key": f"path_check_{stage}_n_candidates",
            "value": float(part.n_candidates.iloc[0]),
        },
    ]
w = pd.read_csv(region_w / "w_controls.csv")
for kind, part in w.groupby("kind"):
    rows += [
        {"key": f"path_check_w_cos_{kind}_min", "value": float(part.cosine_to_w.min())},
        {"key": f"path_check_w_cos_{kind}_max", "value": float(part.cosine_to_w.max())},
    ]
# move limit (30): how many of the six searches reached it
capped = 0
for seed in (42, 123, 456):
    for stage, steps in (
        (
            "stage1",
            results / "stage1_search" / f"recipe_seed{seed}" / "steps" / "all_slices.csv",
        ),
        (
            "stage2",
            results / "stage2_region_rule" / f"seed{seed}" / "steps" / "greedy_recipe.csv",
        ),
    ):
        capped += bool(pd.read_csv(steps).accepted.iloc[-1])
rows += [
    {"key": "move_limit_n_capped", "value": float(capped)},
    {"key": "move_limit_n_searches", "value": 6.0},
]
# R2R: background shift and high-band ratio
rows.append(
    {
        "key": "mean_shift_air_r2r",
        "value": float(seed_table(singles_summary, "r2r").shift_air.mean()),
    }
)
for method in ("ours", "psnr_sel"):  # high-band ratio (GT = 1)
    rows.append(
        {
            "key": f"mean_rh_summary_{method}",
            "value": float(seed_table(singles_summary, method).rh.mean()),
        }
    )
rows.append({"key": "mean_rh_r2r", "value": float(singles_slices.rh_r2r.mean())})

# moves chosen in the stage-2 recipes of Rank2Denoise (three seeds), per move and per expert
usage: dict[str, int] = {}
for seed in (42, 123, 456):
    recipe = pd.read_csv(results / "stage2_region_rule" / f"seed{seed}" / "recipe.csv").recipe.fillna("").iloc[0]
    for move in recipe.split():
        name = re.sub(r"_seed\d+", "", move)  # expert keys carry the seed (to_n2v_seed42_high -> to_n2v_high)
        usage[name] = usage.get(name, 0) + 1
rows += [{"key": f"stage2_chosen_{name}", "value": float(count)} for name, count in sorted(usage.items())]
for expert in ("wavelet", "n2v", "r2r", "s2s", "b2u"):
    rows.append(
        {
            "key": f"stage2_chosen_expert_{expert}",
            "value": float(sum(count for name, count in usage.items() if name.startswith(f"to_{expert}_"))),
        }
    )
# U-Net parameters (no BatchNorm, so the state dict holds only the trainable parameters)
state = torch.load(results / "train" / "n2s_reference_seed42" / "best_model.pth", map_location="cpu")["model"]
rows.append(
    {
        "key": "unet_parameters",
        "value": float(sum(value.numel() for value in state.values())),
    }
)
# run time of the w + stage-1 search and of the stage-2 search (with test evaluation), per seed
cost = {name: pd.read_csv(results / name / "cost.csv").iloc[0] for name in ("stage1_search", "stage2_region_rule")}
rows += [
    {"key": f"run_hours_per_seed_{name}", "value": float(row.run_seconds) / 3 / 3600} for name, row in cost.items()
]
rows.append(
    {
        "key": "run_hours_per_seed_stage1_plus_stage2",
        "value": sum(float(row.run_seconds) for row in cost.values()) / 3 / 3600,
    }
)

out = pd.DataFrame(rows)
out = out.drop_duplicates("key")  # n_patients once
out.to_csv(art / "data" / "paper_numbers.csv", index=False)
print("written:", art / "data" / "paper_numbers.csv")
