# src/moves.py
"""The five stage-1 moves (numpy reference) and PSNR. No GT judge, no search here.

Moves (each candidate is clipped to [0, 1]):
- x + 0.1 (noisy - x)      blend towards the noisy input
- x + 0.1 (N2N - x)        blend towards the Neighbor2Neighbor output
- x + 0.1 (N2S - x)        blend towards the N2S output
- Gauss(x, sigma = 0.5 px) smoothing (edge mode `reflect`)
- x + 0.1 * High(x)        high-frequency boost (ideal radial mask, |f| > 0.25 cycles / pixel)
The torch version of these moves (`scripts/stage2_moves.py`) is checked against this one.
"""

from collections.abc import Callable

import numpy as np
from scipy.ndimage import gaussian_filter

SIZE = 512
MAX_STEPS = 30
BLEND = 0.1
GAUSS_SIGMA = 0.5
HIGH_BAND_CUTOFF = 0.25  # cycles / pixel

_frequency = np.fft.fftfreq(SIZE)
HIGH_BAND_MASK = np.hypot(*np.meshgrid(_frequency, _frequency, indexing="ij")) > HIGH_BAND_CUTOFF


def psnr(image: np.ndarray, clean: np.ndarray) -> float:
    """PSNR with data_range = 1.0."""
    return float(10.0 * np.log10(1.0 / np.mean((image - clean) ** 2)))


def high_band(image: np.ndarray) -> np.ndarray:
    """High-frequency band of an image (FFT with the fixed high-band mask)."""
    return np.real(np.fft.ifft2(np.fft.fft2(image) * HIGH_BAND_MASK))


def directions(noisy: np.ndarray, ne2ne: np.ndarray, n2s: np.ndarray) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """The five moves; each one builds a candidate from the current image, clipped to [0, 1]."""
    steps = {
        "to_noisy": lambda x: x + BLEND * (noisy - x),
        "to_ne2ne": lambda x: x + BLEND * (ne2ne - x),
        "to_n2s": lambda x: x + BLEND * (n2s - x),
        "gauss": lambda x: gaussian_filter(x, GAUSS_SIGMA, mode="reflect"),
        "high_band": lambda x: x + BLEND * high_band(x),
    }
    return {name: (lambda x, step=step: np.clip(step(x), 0.0, 1.0)) for name, step in steps.items()}
