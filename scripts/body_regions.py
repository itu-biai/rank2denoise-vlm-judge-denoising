# scripts/body_regions.py
"""Image regions from the N2S output (no GT): corner, border strip, air and body (edge / flat / other).

Used by the region rule (`stage2_boundary_mask.py`, `stage2_region_rule.py`) and by the background-shift metric (`air`).
- corner: the four 8x8 corner squares; border: strip < 16 px from the image edge (corners excluded);
- body: N2S value >= 0.02 inside the border strip; air: the rest of the inner image;
- body_edge: the 10% body pixels with the largest gradient (Gaussian sigma 1.5); body_flat: gradient <= body median.
The regions partition the image (checked).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from scripts.stage1_search import CORNER_PX

BORDER_PX = 16  # border strip width in pixels
BODY_THRESHOLD = 0.02  # body threshold on the N2S output
EDGE_Q, FLAT_Q = 0.9, 0.5  # edge / flat quantiles
REGIONS = (
    "corner",
    "border",
    "air",
    "body_edge",
    "body_flat",
    "body_other",
)  # the regions partition the image


def region_masks(n2s: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean region masks of one slice from its N2S output."""
    n = n2s.shape[0]
    r, c = np.indices(n2s.shape)
    dist = np.minimum.reduce([r, c, n - 1 - r, n - 1 - c])
    corner = ((r < CORNER_PX) | (r >= n - CORNER_PX)) & ((c < CORNER_PX) | (c >= n - CORNER_PX))
    border = (dist < BORDER_PX) & ~corner
    inner = dist >= BORDER_PX
    body = inner & (n2s >= BODY_THRESHOLD)
    masks = {"corner": corner, "border": border, "air": inner & ~body}
    if body.any():
        gradient = np.hypot(*np.gradient(ndimage.gaussian_filter(n2s, 1.5)))
        edge_cut, flat_cut = (
            np.quantile(gradient[body], EDGE_Q),
            np.quantile(gradient[body], FLAT_Q),
        )
        masks["body_edge"] = body & (gradient >= edge_cut)
        masks["body_flat"] = body & (gradient <= flat_cut) & ~masks["body_edge"]
    else:
        masks["body_edge"] = masks["body_flat"] = np.zeros_like(body)
    masks["body_other"] = body & ~masks["body_edge"] & ~masks["body_flat"]
    if int(sum(mask.sum() for mask in masks.values())) != n2s.size:
        raise RuntimeError("regions do not partition the image.")
    return masks
