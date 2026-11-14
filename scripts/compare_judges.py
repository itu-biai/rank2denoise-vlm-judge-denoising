# scripts/compare_judges.py
"""Comparison rows of the paper: the same two-stage method, only the judge changes (no training).

- Val.-PSNR judge (GT reference): PSNR against the validation GT. Validation GT is read only inside this judge
  (reference row, not part of the method).
- Q judge (Zhu-Milanfar Q). Q runs on the GPU and is checked against a numpy version.
- Random direction: same method with a random unit vector instead of w.
For every judge both stages are searched with that judge.
- Fig. 2: seed 42, two fixed slices (`paper/fig2_selection.csv`); full-slice images are saved.
Method (the same for every judge): stage 1 = five stage-1 moves, unmasked `Filters`, x0 = N2S output; stage 2 =
  `stage2_region_rule.AllMaskedFilters` (m = max(edge, outside body), zero-mean low band), x0 = stage-1 output; `stage2_band_moves.greedy`
  rule (<= 30 steps, mean score, stop when it does not rise, ties go to the first move).
Random directions: three draws from RNG 2027, by seed order 42 -> 0, 123 -> 1, 456 -> 2. The test row q_stage1 is used
  by the paper-run check below; the figure file includes the noisy input.
Checks (the run stops if one fails): Q GPU vs numpy (<= 1e-9 relative); Rank2Denoise recipe re-applied vs
  `stage2_region_rule` `psnr_stage2_region_rule` and stage 1 vs `stage1_search` (per slice, <= 0.02 dB).
Paper-run checks (warning only, `src/paper_run.py`): stage-1 recipes of the Val.-PSNR and Q judges and the mean test
  PSNR of the Q stage-1 recipe vs the paper run.

Usage (cwd = repository root):  python -m scripts.compare_judges
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.expert_outputs import split_files
from scripts.infer_n2s_n2n import RankRecipeConfig, _atomic_csv, _set_tf32
from scripts.region_maps import region_stack
from scripts.stage1_search import OUT_ROOT as STAGE1_ROOT
from scripts.stage1_search import CornerMaskedVlm, MaskedRecipeConfig
from scripts.stage2_band_moves import apply_recipe, greedy, read_recipe
from scripts.stage2_boundary_mask import slice_metrics
from scripts.stage2_moves import (
    STAGE1_MOVES,
    Filters,
    GpuMonitor,
    TensorScorer,
    load_split,
    move_names,
)
from scripts.stage2_region_rule import OUT as REGION_RULE_ROOT
from scripts.stage2_region_rule import AllMaskedFilters, with_mask
from src import paper_run
from src.data import _load_npz_arrays
from src.metrics import LpipsMetric
from src.paths import RESULTS_ROOT
from src.vlm import load_vlm

OUT = RESULTS_ROOT / "compare_judges"
SEEDS = (42, 123, 456)
JUDGES = ("psnr", "q", "random")
JUDGE_LABELS = {"psnr": "Val.-PSNR", "q": "Q"}  # names in the paper-run notes
PATCH, ALPHA = 8, 0.001  # patch size, coherence threshold
RANDOM_SEED = 2027  # random seed
PARITY_DB = 0.02
Q_PARITY_REL = 1e-9
# stage-1 recipes of the Val.-PSNR and Q judges in the paper run (paper-run check only; the run searches its own)
PAPER_STAGE1 = {
    "psnr": {42: ["to_ne2ne"] * 5, 123: ["to_ne2ne"] * 5, 456: ["to_ne2ne"] * 5},
    "q": {
        42: ["to_ne2ne"] * 4 + ["to_n2s"],
        123: ["to_ne2ne"] * 3,
        456: ["to_ne2ne"] * 4 + ["to_n2s"],
    },
}
# mean test PSNR of the Q stage-1 recipe per seed in the paper run (220 slices; paper-run check only)
PAPER_Q_STAGE1_TEST_PSNR = {42: 29.019848, 123: 29.043387, 456: 29.073045}
FIGURE_SEED = 42
FIGURE_SLICES = (
    "001276_05_01_038.png",
    "000064_01_01_067.png",
)  # the slices of paper/fig2_selection.csv
METHODS = (
    "n2s",
    "equal_n2s_n2n",
    "stage1_only",
    "ours",
    "psnr_sel",
    "q_stage1",
    "q_sel",
    "random_sel",
)


# ----------------------------------------------------------------------------- judges


class PsnrJudge:
    """PSNR with validation GT (data_range 1). Reference row only; not used by the method."""

    def __init__(self, clean: torch.Tensor) -> None:
        self.clean = clean

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return 10.0 * torch.log10(1.0 / ((images - self.clean) ** 2).mean(dim=(1, 2)))


def _patch_svd(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Singular values (s1 >= s2) of the gradient matrix on non-overlapping 8x8 patches (torch)."""
    gy, gx = torch.gradient(images, dim=(1, 2))  # same as np.gradient(image): [row, column] order
    n = images.shape[1] // PATCH

    def block_sum(values: torch.Tensor) -> torch.Tensor:
        """Sum over non-overlapping PATCH x PATCH blocks."""
        return values[:, : n * PATCH, : n * PATCH].reshape(len(values), n, PATCH, n, PATCH).sum((2, 4))

    a, b, c = block_sum(gx * gx), block_sum(gx * gy), block_sum(gy * gy)
    half_trace, root = (a + c) / 2.0, torch.sqrt(((a - c) / 2.0) ** 2 + b**2)
    return torch.sqrt((half_trace + root).clamp_min(0.0)), torch.sqrt((half_trace - root).clamp_min(0.0))


class QJudge:
    """Zhu-Milanfar Q: the patch set is chosen once from the noisy input; Q = mean(s1 * R) over the selected patches."""

    def __init__(self, noisy: torch.Tensor) -> None:
        a = ALPHA ** (1.0 / (PATCH * PATCH - 1))
        tau = float(np.sqrt((1.0 - a) / (1.0 + a)))
        s1, s2 = _patch_svd(noisy)
        self.mask = ((s1 - s2) / (s1 + s2).clamp_min(1e-12) > tau).to(noisy.dtype)
        if bool((self.mask.sum((1, 2)) == 0).any()):
            raise RuntimeError("Q: a slice has no selected patch.")

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        s1, s2 = _patch_svd(images)
        value = s1 * (s1 - s2) / (s1 + s2).clamp_min(1e-12)
        return (value * self.mask).sum((1, 2)) / self.mask.sum((1, 2))


def q_numpy(image: np.ndarray, noisy: np.ndarray) -> float:
    """Numpy version of the coherence mask + Q metric; used only for the consistency check."""

    def patch_svd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Singular values (s1 >= s2) of the gradient matrix on non-overlapping patches (numpy)."""
        gy, gx = np.gradient(x.astype(np.float64))
        n = x.shape[0] // PATCH

        def block_sum(values: np.ndarray) -> np.ndarray:
            """Sum over non-overlapping PATCH x PATCH blocks."""
            return values[: n * PATCH, : n * PATCH].reshape(n, PATCH, n, PATCH).sum((1, 3))

        a, b, c = block_sum(gx * gx), block_sum(gx * gy), block_sum(gy * gy)
        half_trace, root = (a + c) / 2.0, np.sqrt(((a - c) / 2.0) ** 2 + b**2)
        return np.sqrt(np.maximum(half_trace + root, 0.0)), np.sqrt(np.maximum(half_trace - root, 0.0))

    a = ALPHA ** (1.0 / (PATCH * PATCH - 1))
    tau = np.sqrt((1.0 - a) / (1.0 + a))
    s1, s2 = patch_svd(noisy)
    mask = (s1 - s2) / np.maximum(s1 + s2, 1e-12) > tau
    s1, s2 = patch_svd(image)
    return float((s1 * (s1 - s2) / np.maximum(s1 + s2, 1e-12))[mask].mean())


def random_direction(seed: int, dim: int) -> torch.Tensor:
    """Random unit vector of the seed, from a fixed generator."""
    vectors = np.random.default_rng(RANDOM_SEED).standard_normal((len(SEEDS), dim))
    vector = vectors[SEEDS.index(seed)]
    return torch.from_numpy(vector / np.linalg.norm(vector)).float()


def val_clean(splits: dict[str, list[Path]], ids: list[str], device: torch.device) -> torch.Tensor:
    """Validation GT (gt_clipped), same dtype path as the cached test GT (float32 -> float64). Used only by PsnrJudge."""
    paths = {path.stem: path for path in splits["val"]}
    key = RankRecipeConfig().clean_key
    stack = np.stack([_load_npz_arrays(paths[i], (key,))[key].astype(np.float32).astype(np.float64) for i in ids])
    return torch.from_numpy(stack).to(device)


# ----------------------------------------------------------------------------- run


def judge_for(
    name: str,
    seed: int,
    base: dict[str, torch.Tensor],
    ids: list[str],
    splits: dict[str, list[Path]],
    vlm: CornerMaskedVlm,
    device: torch.device,
) -> PsnrJudge | QJudge | TensorScorer:
    """Judge of the given name: validation PSNR, Q or random direction."""
    if name == "psnr":
        return PsnrJudge(val_clean(splits, ids, device))
    if name == "q":
        return QJudge(base["noisy"])
    own = torch.load(STAGE1_ROOT / "w" / f"n2s_run{seed}_renoise42_masked.pt", map_location="cpu")["direction"]
    return TensorScorer(
        vlm,
        random_direction(seed, own.numel()).to(device),
        MaskedRecipeConfig(seed=seed).start_batch,
    )


def run(
    seed: int,
    splits: dict[str, list[Path]],
    vlm: CornerMaskedVlm,
    lpips_metric: LpipsMetric,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict]]:
    """Two-stage search with every judge and test evaluation for one seed; returns the per-slice test table and the check rows. Saved results are reused."""
    folder = OUT / f"seed{seed}"
    target = folder / "test_per_slice.csv"
    if target.exists():
        return pd.read_csv(target), pd.read_csv(folder / "checks.csv").to_dict("records")
    stage1_filters, masked_filters = Filters(device), AllMaskedFilters(device)
    moves = move_names(seed)
    stage1_only = read_recipe(STAGE1_ROOT / f"recipe_seed{seed}" / "recipes" / "all_slices.json")["recipe"]
    ours = read_recipe(REGION_RULE_ROOT / f"seed{seed}" / "recipes" / "greedy_recipe.json")["recipe"]
    checks, recipes = [], {}
    # validation: two-stage search for every judge; validation GT only inside PsnrJudge
    ids, base = load_split(seed, "val", splits, device)
    with_mask(base, device)
    sample = [0, len(ids) // 2, len(ids) - 1]
    q_gpu = QJudge(base["noisy"][sample])(base["n2s"][sample]).cpu().numpy()
    q_cpu = np.array([q_numpy(base["n2s"][k].cpu().numpy(), base["noisy"][k].cpu().numpy()) for k in sample])
    worst = float(np.max(np.abs(q_gpu - q_cpu) / np.abs(q_cpu)))
    checks.append(
        {
            "seed": seed,
            "check": "q_gpu_vs_numpy_rel",
            "value": worst,
            "tolerance": Q_PARITY_REL,
            "passed": worst <= Q_PARITY_REL,
        }
    )
    if not checks[-1]["passed"]:
        raise RuntimeError(f"seed {seed}: Q GPU vs numpy mismatch ({worst:.2e}).")
    val_rows = []
    for name in JUDGES:
        judge = judge_for(name, seed, base, ids, splits, vlm, device)
        stage_folder = folder / name
        stage1 = greedy(
            stage_folder / "stage1",
            list(STAGE1_MOVES),
            base["n2s"],
            base,
            stage1_filters,
            judge,
        )
        if name in PAPER_STAGE1:
            paper_run.recipe(
                f"the stage-1 recipe of the {JUDGE_LABELS[name]} judge",
                stage1,
                PAPER_STAGE1[name][seed],
                seed=seed,
            )
        start = apply_recipe(stage1_filters, stage1, base["n2s"], base)
        stage2 = greedy(stage_folder / "stage2", moves, start, base, masked_filters, judge)
        final = apply_recipe(masked_filters, stage2, start, base)
        recipes[name] = (stage1, stage2)
        val_rows.append(
            {
                "seed": seed,
                "judge": name,
                "n_stage1": len(stage1),
                "n_stage2": len(stage2),
                "val_judge_start": float(judge(start).mean()),
                "val_judge_final": float(judge(final).mean()),
                "stage1": " ".join(stage1),
                "stage2": " ".join(stage2),
            }
        )
        print(
            f"seed {seed} {name}: stage 1 {len(stage1)}, stage 2 {len(stage2)} steps",
            flush=True,
        )
        del judge, start, final
        torch.cuda.empty_cache()
    del base
    torch.cuda.empty_cache()
    # test: evaluation (GT only here)
    ids, base = load_split(seed, "test", splits, device)
    with_mask(base, device)
    stage1_only_test = apply_recipe(stage1_filters, stage1_only, base["n2s"], base)
    outputs = {
        "n2s": base["n2s"],
        "equal_n2s_n2n": (0.5 * base["n2s"] + 0.5 * base["ne2ne"]).clamp(0.0, 1.0),
        "stage1_only": stage1_only_test,
        "ours": apply_recipe(masked_filters, ours, stage1_only_test, base),
    }
    for name, (stage1, stage2) in recipes.items():
        outputs[f"{name}_stage1"] = apply_recipe(stage1_filters, stage1, base["n2s"], base)
        outputs[f"{name}_sel"] = apply_recipe(masked_filters, stage2, outputs[f"{name}_stage1"], base)
    clean = base["clean"].cpu().numpy()
    images = {name: value.cpu().numpy() for name, value in outputs.items()}
    if seed == FIGURE_SEED:
        index = [ids.index(i) for i in FIGURE_SLICES]
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "figure_slices.npz",
            slice_ids=np.array(FIGURE_SLICES),
            clean=clean[index],
            noisy=base["noisy"].cpu().numpy()[index],
            ne2ne=base["ne2ne"].cpu().numpy()[index],
            **{name: images[name][index] for name in METHODS},
        )
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
            "ours",
            REGION_RULE_ROOT / f"seed{seed}" / "test_per_slice.csv",
            "psnr_stage2_region_rule",
        ),
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
    paper_run.number(
        "the mean test PSNR of the Q stage-1 recipe",
        float(frame["psnr_q_stage1"].mean()),
        PAPER_Q_STAGE1_TEST_PSNR[seed],
        tolerance=PARITY_DB,
        unit=" dB",
        seed=seed,
    )
    if not all(row["passed"] for row in checks):
        raise RuntimeError(
            f"seed {seed}: check failed; do not use these results.\n" + pd.DataFrame(checks).to_markdown(index=False)
        )
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(checks), folder / "checks.csv")
    _atomic_csv(pd.DataFrame(val_rows), folder / "recipes.csv")
    _atomic_csv(frame, target)  # written last: its presence marks the run as done (resume)
    return frame, checks


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per seed and method."""
    rows = []
    for seed, block in frame.groupby("seed"):
        for method in METHODS:
            row = {"seed": seed, "method": method}
            for metric in ("psnr", "ssim", "lpips", "rh", "shift_air"):
                row[metric] = block[f"{metric}_{method}"].mean()
            diff = block["psnr_ours"] - block[f"psnr_{method}"]
            row["ours_minus_method_psnr"] = diff.mean()
            row["slices_ours_better"] = int((diff > 0).sum())
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
