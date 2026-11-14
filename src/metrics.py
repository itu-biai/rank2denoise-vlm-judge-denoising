# src/metrics.py
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import patient_id_from_path
from src.masking import j_invariant_inference

HIGH_BAND: tuple[float, float] = (0.25, 0.5)


class LpipsMetric:
    """LPIPS distance (AlexNet) between two [0, 1] images."""

    def __init__(self, device: torch.device) -> None:
        self.model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        def as_input(image: np.ndarray) -> torch.Tensor:
            """Map a [0, 1] image to the LPIPS input: [-1, 1], three channels, on the device."""
            return (torch.from_numpy(image).float() * 2.0 - 1.0).expand(1, 3, *image.shape).to(self.device)

        return float(self.model(as_input(prediction), as_input(target)).item())


def high_band_ratio(prediction: np.ndarray, target: np.ndarray, band: tuple[float, float] = HIGH_BAND) -> float:
    """Power of the prediction over power of the target in a radial frequency band."""
    height, width = target.shape
    radius = np.hypot(*np.meshgrid(np.fft.fftfreq(height), np.fft.fftfreq(width), indexing="ij"))
    in_band = (radius >= band[0]) & (radius <= band[1])

    def band_power(image: np.ndarray) -> float:
        """Spectral power of the mean-removed image inside the band."""
        return float((np.abs(np.fft.fft2(image - image.mean())) ** 2)[in_band].sum())

    return band_power(prediction) / band_power(target)


def slice_metrics(prediction: np.ndarray, target: np.ndarray, lpips_metric: LpipsMetric) -> dict[str, float]:
    """PSNR, SSIM, LPIPS and high-band ratio of one slice; the prediction is clipped to [0, 1]."""
    prediction = np.clip(prediction, 0.0, 1.0)
    return {
        "psnr": float(peak_signal_noise_ratio(target, prediction, data_range=1.0)),
        "ssim": float(structural_similarity(target, prediction, data_range=1.0)),
        "lpips": lpips_metric(prediction, target),
        "high_band_ratio": high_band_ratio(prediction, target),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lpips_metric: LpipsMetric,
    method: str,
    seed: int,
    j_invariant: bool = True,
) -> pd.DataFrame:
    """Per-slice test metrics of a model on a loader, as a table."""
    model.eval()
    rows = []
    for noisy, clean, stems in tqdm(loader, desc=f"test {method}", leave=False):
        noisy = noisy.to(device)
        output = j_invariant_inference(model, noisy) if j_invariant else model(noisy)
        for prediction, target, stem in zip(output.cpu().numpy(), clean.numpy(), stems):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "slice_id": stem,
                    "patient_id": patient_id_from_path(Path(stem)),
                    **slice_metrics(prediction[0], target[0], lpips_metric),
                }
            )
    return pd.DataFrame(rows)


def save_per_slice_metrics(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Write the per-slice table to CSV and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
