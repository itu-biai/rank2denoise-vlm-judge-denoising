# scripts/stage2_moves.py
"""Stage-2 moves on the GPU (torch, float64), the VLM scorer for tensors, data loading and consistency checks.

Experts of the band moves: wavelet, N2V, R2R, S2S, B2U.
Moves: the five stage-1 moves (same definition as `src.moves.directions`, ported to torch) + for every
expert e and band b:  to_<e>_<b>(x) = clip(x + 0.1 * band_b(e - x), 0, 1).
Gaussian bands: low = G4 r, mid = G1 r - G4 r, high = r - G1 r (scipy `gaussian_filter` defaults: truncate 4,
  mode "reflect"); low + mid + high = r.
Consistency checks (`consistency_checks`, run by `scripts/stage2_band_moves.py` before its search): (1) torch moves vs numpy
  `directions` and torch Gaussian filters vs scipy (val, 4 slices, <= 1e-9); (2) first-step mean scores of the five
  stage-1 moves vs the steps of `scripts/stage1_search.py` (<= 1e-4). `band_self_test`: the three bands sum to r.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter

from scripts.expert_outputs import load_output, output_done
from scripts.infer_n2s_n2n import (
    RankRecipeConfig,
    _atomic_csv,
    load_cache,
)
from scripts.stage1_search import OUT_ROOT as STAGE1_ROOT
from scripts.stage1_search import CornerMaskedVlm, MaskedRecipeConfig
from src.moves import BLEND, GAUSS_SIGMA, HIGH_BAND_MASK, directions
from src.paths import RESULTS_ROOT

OUT = RESULTS_ROOT / "stage2_move_checks"  # output of `consistency_checks`
SEEDS = (42, 123, 456)
BANDS = ("low", "mid", "high")
STAGE1_MOVES = (
    "to_noisy",
    "to_ne2ne",
    "to_n2s",
    "gauss",
    "high_band",
)  # same order as infer_n2s_n2n.MOVES
SIGMA_FINE, SIGMA_COARSE = 1.0, 4.0  # Gaussian sigmas for the fine / coarse smoothing
VLM_CHECK_TOL = 1e-4
TORCH_CHECK_TOL = 1e-9


def expert_keys(seed: int) -> list[str]:
    """Expert order: wavelet, N2V, R2R, S2S, B2U."""
    return ["wavelet", f"n2v_seed{seed}", f"r2r_seed{seed}", "s2s", f"b2u_seed{seed}"]


def move_names(seed: int) -> list[str]:
    """The five stage-1 moves followed by every expert band move of the seed."""
    return [
        *STAGE1_MOVES,
        *(f"to_{key}_{band}" for key in expert_keys(seed) for band in BANDS),
    ]


# ----------------------------------------------------------------------------- torch filters (float64)


def _gauss_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """scipy `gaussian_filter` kernel: radius int(4 sigma + 0.5), normalised."""
    radius = int(4.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
    weights = torch.exp(-0.5 * (x / sigma) ** 2)
    return weights / weights.sum()


def _symmetric_pad(x: torch.Tensor, radius: int, dim: int) -> torch.Tensor:
    """scipy mode "reflect" (d c b a | a b c d | d c b a)."""
    size = x.shape[dim]
    head = x.narrow(dim, 0, radius).flip(dim)
    tail = x.narrow(dim, size - radius, radius).flip(dim)
    return torch.cat([head, x, tail], dim=dim)


def gauss(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """(N, H, W) -> separable Gaussian, axis 0 (H) first, then axis 1 (W), as in scipy. The kernel is symmetric, so
    conv2d (correlation) = scipy correlate1d.
    """
    kernel = _gauss_kernel(sigma, x.device)
    radius = (len(kernel) - 1) // 2
    out = torch.nn.functional.conv2d(_symmetric_pad(x, radius, 1).unsqueeze(1), kernel.view(1, 1, -1, 1))
    out = torch.nn.functional.conv2d(_symmetric_pad(out, radius, 3), kernel.view(1, 1, 1, -1))
    return out.squeeze(1)


class Filters:
    """Constants on the device (high_band mask) + band functions."""

    def __init__(self, device: torch.device) -> None:
        self.mask = torch.from_numpy(HIGH_BAND_MASK).to(device)

    def high_band(self, x: torch.Tensor) -> torch.Tensor:
        """High-frequency band (FFT with the fixed high-band mask)."""
        return torch.fft.ifft2(torch.fft.fft2(x) * self.mask).real

    def smooth(self, r: torch.Tensor, which: str) -> torch.Tensor:
        """which = "fine" (Gaussian sigma 1) or "coarse" (Gaussian sigma 4)."""
        return gauss(r, SIGMA_FINE if which == "fine" else SIGMA_COARSE)

    def band(self, r: torch.Tensor, band: str) -> torch.Tensor:
        """low = coarse, mid = fine - coarse, high = r - fine (sum = r)."""
        if band == "low":
            return self.smooth(r, "coarse")
        if band == "mid":
            return self.smooth(r, "fine") - self.smooth(r, "coarse")
        return r - self.smooth(r, "fine")

    def apply(self, move: str, x: torch.Tensor, base: dict[str, torch.Tensor]) -> torch.Tensor:
        """One move on a batch (N, H, W): stage-1 moves + band moves. The result is clipped to [0, 1]."""
        if move == "to_noisy":
            out = x + BLEND * (base["noisy"] - x)
        elif move == "to_ne2ne":
            out = x + BLEND * (base["ne2ne"] - x)
        elif move == "to_n2s":
            out = x + BLEND * (base["n2s"] - x)
        elif move == "gauss":
            out = gauss(x, GAUSS_SIGMA)
        elif move == "high_band":
            out = x + BLEND * self.high_band(x)
        else:
            key, band = move.removeprefix("to_").rsplit("_", 1)
            out = x + BLEND * self.band(base[key] - x, band)
        return out.clamp(0.0, 1.0)


# ----------------------------------------------------------------------------- VLM score (torch input) + GPU monitor


class TensorScorer:
    """Same computation as `infer_n2s_n2n.RankScorer`; the input is already on the device (float64 -> float32). Batch halves on OOM."""

    def __init__(self, vlm: CornerMaskedVlm, direction: torch.Tensor, batch: int) -> None:
        self.vlm, self.batch = vlm, batch
        self.direction = direction
        self.embedded, self.seconds = 0, 0.0

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        began, scores, index = time.time(), [], 0
        while index < len(images):
            try:
                batch = images[index : index + self.batch].to(torch.float32)
                scores.append(self.vlm.encode_image(batch.unsqueeze(1)).float() @ self.direction)
                index += len(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch == 1:
                    raise
                self.batch //= 2
                print(f"  CUDA OOM → VLM batch {self.batch}", flush=True)
        self.embedded += len(images)
        self.seconds += time.time() - began
        return torch.cat(scores).double()


def scorer_for(seed: int, vlm: CornerMaskedVlm, device: torch.device) -> TensorScorer:
    """VLM scorer with the stored w of the seed."""
    payload = torch.load(STAGE1_ROOT / "w" / f"n2s_run{seed}_renoise42_masked.pt", map_location="cpu")
    return TensorScorer(
        vlm,
        payload["direction"].float().to(device),
        MaskedRecipeConfig(seed=seed).start_batch,
    )


class GpuMonitor:
    """Samples `torch.cuda.utilization()` once per second (needs pynvml; stays empty otherwise)."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.samples.append(float(torch.cuda.utilization()))
            except Exception:  # noqa: BLE001  # no pynvml -> no measurement; the run continues
                return

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()

    def summary(self) -> dict[str, float]:
        """Mean and median GPU utilisation and the number of samples."""
        values = np.array(self.samples) if self.samples else np.array([np.nan])
        return {
            "gpu_util_mean_pct": float(np.nanmean(values)),
            "gpu_util_median_pct": float(np.nanmedian(values)),
            "gpu_util_samples": len(self.samples),
        }


# ----------------------------------------------------------------------------- data


def load_split(
    seed: int, split: str, splits: dict[str, list[Path]], device: torch.device
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """{noisy, n2s, ne2ne[, clean]} + five experts, all on the device in float64."""
    data = load_cache(split, splits[split], RankRecipeConfig(seed=seed))  # validation: no GT (checked by load_cache)
    ids = list(data)
    base = {
        key: torch.from_numpy(np.stack([data[i][key] for i in ids])).to(device)
        for key in ("noisy", "n2s", "ne2ne", "clean")
        if key in data[ids[0]]
    }
    del data
    if split == "val" and "clean" in base:
        raise RuntimeError("validation data contain GT; they must not.")
    for key in expert_keys(seed):
        if not output_done(key):
            raise FileNotFoundError(f"expert output missing: experts/outputs/{key}; run step 4 of the README first.")
        base[key] = torch.from_numpy(load_output(key, split, ids)).to(device)
    return ids, base


# ----------------------------------------------------------------------------- consistency checks


def consistency_checks(
    splits: dict[str, list[Path]],
    vlm: CornerMaskedVlm,
    filters: Filters,
    device: torch.device,
) -> None:
    """Run the consistency checks for every seed and write them to CSV; raises if one fails."""
    target = OUT / "consistency_checks.csv"
    if target.exists() and pd.read_csv(target)["passed"].all():
        print("Consistency checks already passed; skipped.", flush=True)
        return
    rows = []
    for seed in SEEDS:
        _ids, base = load_split(seed, "val", splits, device)
        # (1) torch moves vs numpy `directions` (4 slices; band moves have no numpy version, the stage-1 moves are checked)
        worst_torch = 0.0
        for k in range(4):
            table = directions(*(base[key][k].cpu().numpy() for key in ("noisy", "ne2ne", "n2s")))
            start = base["n2s"][k].cpu().numpy() * 0.9 + base["noisy"][k].cpu().numpy() * 0.1  # a second test point
            for point in (base["n2s"][k].cpu().numpy(), start):
                x = torch.from_numpy(point).to(device).unsqueeze(0)
                single = {key: value[k : k + 1] for key, value in base.items()}
                for move in STAGE1_MOVES:
                    ours = filters.apply(move, x, single)[0].cpu().numpy()
                    worst_torch = max(worst_torch, float(np.abs(ours - table[move](point)).max()))
            # Gaussian band filters (sigma 1 / 4) vs scipy
            residual = base["wavelet"][k] - base["n2s"][k]
            for sigma in (SIGMA_FINE, SIGMA_COARSE):
                ours = gauss(residual.unsqueeze(0), sigma)[0].cpu().numpy()
                worst_torch = max(
                    worst_torch,
                    float(np.abs(ours - gaussian_filter(residual.cpu().numpy(), sigma)).max()),
                )
        rows.append(
            {
                "seed": seed,
                "check": "torch_moves_vs_numpy",
                "value": worst_torch,
                "tolerance": TORCH_CHECK_TOL,
                "passed": worst_torch <= TORCH_CHECK_TOL,
            }
        )
        # (2) first-step VLM means vs the stage-1 steps
        scorer = scorer_for(seed, vlm, device)
        recorded = pd.read_csv(STAGE1_ROOT / f"recipe_seed{seed}" / "steps" / "all_slices.csv").iloc[0]
        means = {move: float(scorer(filters.apply(move, base["n2s"], base)).mean()) for move in STAGE1_MOVES}
        worst_vlm = max(abs(means[move] - float(recorded[f"mean_{move}"])) for move in STAGE1_MOVES)
        rows.append(
            {
                "seed": seed,
                "check": "vlm_step1_means_vs_stage1_only",
                "value": worst_vlm,
                "tolerance": VLM_CHECK_TOL,
                "passed": worst_vlm <= VLM_CHECK_TOL,
            }
        )
        print(pd.DataFrame(rows[-2:]).to_markdown(index=False), flush=True)
        del base
        torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    _atomic_csv(table, target)
    if not table["passed"].all():
        raise RuntimeError("Consistency check failed; search not started.\n" + table.to_markdown(index=False))
    print("Consistency checks passed.", flush=True)


def band_self_test(filters: Filters, device: torch.device) -> None:
    """Check that the bands sum to r (CPU/GPU, no VLM)."""
    r = torch.randn(
        2,
        512,
        512,
        dtype=torch.float64,
        device=device,
        generator=torch.Generator(device).manual_seed(0),
    )
    total = sum(filters.band(r, band) for band in BANDS)
    error = float((total - r).abs().max())
    if error > 1e-10:
        raise RuntimeError(f"bands do not sum to r (difference {error:.2e}).")
