# scripts/wavelet_expert.py
"""Wavelet expert: BayesShrink soft thresholding, applied to every validation / test slice (CPU, no training, no GT).

`skimage.restoration.denoise_wavelet(x, method="BayesShrink", mode="soft", rescale_sigma=True)` with library defaults
(the noise level is estimated from each noisy slice); the output is clipped to [0, 1] and stored as float16 (`scripts/expert_outputs.py`).
Slices are independent, so they are processed in parallel.

Output: `<R2D_RESULTS>/experts/outputs/wavelet/{val,test}.npy` (`scripts/expert_outputs.py`).
Usage (cwd = repository root):  python -m scripts.wavelet_expert
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from skimage.restoration import denoise_wavelet

from scripts.expert_outputs import noisy_inputs, output_done, save_output, split_files

KEY = "wavelet"


def wavelet(x: np.ndarray) -> np.ndarray:
    """BayesShrink soft-threshold wavelet denoising, clipped to [0, 1]."""
    output = denoise_wavelet(x.astype(np.float64), method="BayesShrink", mode="soft", rescale_sigma=True)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def main() -> None:
    """Write the wavelet outputs of the validation and test slices."""
    if output_done(KEY):
        print(f"{KEY}: done, skipped", flush=True)
        return
    splits = split_files()
    began = time.time()
    for split in ("val", "test"):
        ids, noisy = noisy_inputs(split, splits)
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
            result = list(pool.map(wavelet, list(noisy), chunksize=4))
        save_output(KEY, split, np.stack(result), ids)
    print(f"{KEY}: {(time.time() - began) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
