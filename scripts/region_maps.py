# scripts/region_maps.py
"""Region masks for a stack of slices (used by the test scripts for the background shift)."""

from __future__ import annotations

import numpy as np

from scripts.body_regions import REGIONS, region_masks


def region_stack(n2s: np.ndarray) -> dict[str, np.ndarray]:
    """Region masks per slice, (N, H, W) bool."""
    per_slice = [region_masks(image) for image in n2s]
    return {region: np.stack([masks[region] for masks in per_slice]) for region in REGIONS}
