# scripts/judge_resnet50.py
"""Same method with a non-medical embedding: the direction w is built in ImageNet ResNet-50 space (no training).

Same layout as the random-direction judge of `compare_judges.py`; only the embedding changes:
- w: `noise_fields_seeds.rank_direction_cached` (647 noisy training slices, batch 16, generator RENOISE_SEED, noise fields of
  the seed, corner mask `CornerMaskedVlm`, TF32 off), with ResNet-50 instead of MedSigLIP.
- Stage 1: the five stage-1 moves are searched with this judge (x0 = N2S output, no mask); stage 2: 20 moves,
  `stage2_region_rule.AllMaskedFilters` (edge + outside-body mask), x0 = stage-1 output; `stage2_band_moves.greedy` rule.
Embedding: torchvision `resnet50` with IMAGENET1K_V2 weights, last layer removed (2048-d global average pool), grey ->
  3-channel copy, ImageNet mean / std, 512 x 512 input (no resizing), L2 unit vector (as the MedSigLIP embedding), fp32.
Checks: test PSNR of N2S / equal / stage 1 / Rank2Denoise vs `compare_judges` (per slice, <= 0.02 dB); w is a unit vector,
  margin > 0 and finite.

Usage (cwd = repository root):  python -m scripts.judge_resnet50
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

from scripts.compare_judges import OUT as JUDGES_ROOT
from scripts.expert_outputs import split_files
from scripts.infer_n2s_n2n import RankRecipeConfig, _atomic_csv, _set_tf32
from scripts.noise_fields_seeds import (
    RANK_BATCH,
    RENOISE_SEED,
    load_train_noisy,
    rank_direction_cached,
    seed42_fields,
)
from scripts.region_maps import region_stack
from scripts.single_experts_and_checks import stage_recipes
from scripts.stage1_search import CornerMaskedVlm, seed_fields
from scripts.stage2_band_moves import apply_recipe, greedy
from scripts.stage2_boundary_mask import slice_metrics
from scripts.stage2_moves import (
    STAGE1_MOVES,
    Filters,
    GpuMonitor,
    TensorScorer,
    load_split,
    move_names,
)
from scripts.stage2_region_rule import AllMaskedFilters, with_mask
from src.metrics import LpipsMetric
from src.paths import RESULTS_ROOT

OUT = RESULTS_ROOT / "judge_resnet50"
SEEDS = (42, 123, 456)
PARITY_DB = 0.02
SCORE_BATCH = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METHODS = ("n2s", "equal_n2s_n2n", "stage1_only", "ours", "resnet_sel")
PARITY = ("n2s", "equal_n2s_n2n", "stage1_only", "ours")


class ResNetEmbedder:
    """Replaces `FrozenVlm.encode_image`: (N, 1, H, W) in [0, 1] -> (N, 2048) unit vector."""

    def __init__(self, device: torch.device) -> None:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        self.model = model.to(device).eval()
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """L2-normalised ResNet-50 embedding of (N, 1, H, W) images in [0, 1]."""
        x = images.to(self.device, torch.float32).expand(-1, 3, -1, -1)
        features = self.model((x - self.mean) / self.std)
        return torch.nn.functional.normalize(features, dim=1)


def build_w(
    seed: int,
    embedder: CornerMaskedVlm,
    noisy: torch.Tensor,
    phi_noisy: torch.Tensor,
    stems: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Judge direction w of the seed in ResNet-50 space; a saved w is reused."""
    target = OUT / "w" / f"resnet_n2s_run{seed}.pt"
    if target.exists():
        return torch.load(target, map_location="cpu")["direction"].float()
    key = RankRecipeConfig().input_key
    fields = seed42_fields(stems, key) if seed == 42 else seed_fields(seed, stems, key)
    direction, margin = rank_direction_cached(embedder, noisy, phi_noisy, fields, RENOISE_SEED, device)
    if not torch.isfinite(direction).all() or abs(float(direction.norm()) - 1.0) > 1e-4 or not margin > 0.0:
        raise RuntimeError(
            f"seed {seed}: invalid ResNet w / margin (||w|| {float(direction.norm())}, margin {margin})."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "direction": direction,
            "margin": margin,
            "n2s_seed": seed,
            "gen_seed": RENOISE_SEED,
            "encoder": "resnet50_v2",
        },
        target,
    )
    print(f"  ResNet w seed {seed}: margin {margin:.6f}", flush=True)
    return direction.float()


def run(
    seed: int,
    splits: dict[str, list[Path]],
    embedder: CornerMaskedVlm,
    direction: torch.Tensor,
    lpips_metric: LpipsMetric,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict]]:
    """Searches and test evaluation for one seed; returns the per-slice test table and the check rows. Saved results are reused."""
    folder = OUT / f"seed{seed}"
    target = folder / "test_per_slice.csv"
    if target.exists():
        return pd.read_csv(target), pd.read_csv(folder / "checks.csv").to_dict("records")
    recipes = stage_recipes(seed)
    stage1_filters, masked_filters = Filters(device), AllMaskedFilters(device)
    judge = TensorScorer(embedder, direction.to(device), SCORE_BATCH)
    # validation: two-stage search (no GT)
    _, base = load_split(seed, "val", splits, device)
    with_mask(base, device)
    stage1 = greedy(folder / "stage1", list(STAGE1_MOVES), base["n2s"], base, stage1_filters, judge)
    start = apply_recipe(stage1_filters, stage1, base["n2s"], base)
    stage2 = greedy(folder / "stage2", move_names(seed), start, base, masked_filters, judge)
    print(
        f"seed {seed} ResNet judge: stage 1 {len(stage1)}, stage 2 {len(stage2)} steps",
        flush=True,
    )
    del base, start
    torch.cuda.empty_cache()
    # test: evaluation (GT only here)
    ids, base = load_split(seed, "test", splits, device)
    with_mask(base, device)
    stage1_only = apply_recipe(stage1_filters, recipes["stage1_only"], base["n2s"], base)
    resnet_stage1 = apply_recipe(stage1_filters, stage1, base["n2s"], base)
    outputs = {
        "n2s": base["n2s"],
        "equal_n2s_n2n": (0.5 * base["n2s"] + 0.5 * base["ne2ne"]).clamp(0.0, 1.0),
        "stage1_only": stage1_only,
        "ours": apply_recipe(masked_filters, recipes["ours"], stage1_only, base),
        "resnet_stage1": resnet_stage1,
        "resnet_sel": apply_recipe(masked_filters, stage2, resnet_stage1, base),
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
    folder.mkdir(parents=True, exist_ok=True)
    _atomic_csv(pd.DataFrame(checks), folder / "checks.csv")
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "seed": seed,
                    "n_stage1": len(stage1),
                    "n_stage2": len(stage2),
                    "stage1": " ".join(stage1),
                    "stage2": " ".join(stage2),
                }
            ]
        ),
        folder / "recipes.csv",
    )
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
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Run all seeds and write the per-slice, summary and check tables."""
    began = time.time()
    device = torch.device("cuda")
    splits = split_files()
    embedder = CornerMaskedVlm(ResNetEmbedder(device))
    lpips_metric = LpipsMetric(device)
    frames, checks = [], []
    with GpuMonitor() as monitor:
        directions = {}
        if not all((OUT / "w" / f"resnet_n2s_run{seed}.pt").exists() for seed in SEEDS):
            _set_tf32(False)  # w computation: same setting as the other w vectors
            config = RankRecipeConfig()
            stems = [path.stem for path in splits["train_noisy"]]
            noisy = load_train_noisy(splits["train_noisy"], config.input_key, config.read_threads)
            with torch.no_grad():
                phi_noisy = torch.cat(
                    [
                        embedder.encode_image(noisy[start : start + RANK_BATCH].to(device)).cpu()
                        for start in range(0, len(noisy), RANK_BATCH)
                    ]
                )
            for seed in SEEDS:
                directions[seed] = build_w(seed, embedder, noisy, phi_noisy, stems, device)
            del noisy, phi_noisy
        else:
            directions = {seed: build_w(seed, embedder, None, None, [], device) for seed in SEEDS}
        _set_tf32(True)  # search: same as the other judges
        for seed in SEEDS:
            frame, seed_checks = run(seed, splits, embedder, directions[seed], lpips_metric, device)
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
