# paper/infer_supervised.py
"""Test outputs of the supervised U-Net for Fig. 2 (CPU, no training).

The seed-42 supervised checkpoint (`scripts/train_seed42.py`) is applied to the 220 noisy test slices (sorted file
order, batch 4); outputs are clipped to [0, 1]. Writes `slice_ids.txt` and `supervised.npy` to the output folder.
Usage: python -m paper.infer_supervised <checkpoint best_model.pth> <data>/test <output_dir>
"""

import sys
from pathlib import Path

import numpy as np
import torch

from src.unet import UNet

EXPECTED_TEST = 220


def main(checkpoint: Path, test_dir: Path, out: Path) -> None:
    """Apply the supervised checkpoint to the noisy test slices and save the clipped outputs."""
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(test_dir.glob("*.npz"))
    if len(files) != EXPECTED_TEST:
        raise RuntimeError(f"test slices: {len(files)}, expected {EXPECTED_TEST}")
    (out / "slice_ids.txt").write_text("\n".join(p.name[:-4] for p in files))
    noisy = np.stack([np.load(p)["noisy_clipped_100k"].astype(np.float32) for p in files])
    model = UNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])
    model.eval()
    result = np.zeros((len(files), 512, 512), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(files), 4):
            batch = torch.from_numpy(noisy[start : start + 4]).unsqueeze(1)
            result[start : start + 4] = model(batch).clamp(0.0, 1.0).numpy()[:, 0]
    np.save(out / "supervised.npy", result)


if __name__ == "__main__":
    main(*(Path(p) for p in sys.argv[1:4]))
