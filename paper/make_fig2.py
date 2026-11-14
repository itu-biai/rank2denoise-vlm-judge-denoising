"""Fig. 2 of the paper: two regions, 96 px ROI, seed 42.

Images come from `figure_slices.npz` written by `scripts/compare_judges.py`; the supervised row comes from
`paper/infer_supervised.py` (`supervised.npy`). The two slices and their 96 x 96 windows are fixed in
`fig2_selection.csv` (next to this file).
Check: full-slice PSNR of 'ours' and 'psnr_sel' in the npz vs `compare_judges/seed42/test_per_slice.csv` (<= 1e-6 dB).
Usage: python paper/make_fig2.py <results_root> <output_dir> <supervised_inference_dir>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

results, art, infer = (Path(p) for p in sys.argv[1:4])
for sub in ("data", "figures"):
    (art / sub).mkdir(parents=True, exist_ok=True)
ROI = 96
refs = results / "compare_judges" / "seed42"
z = np.load(refs / "figure_slices.npz", allow_pickle=False)
per_slice = pd.read_csv(refs / "test_per_slice.csv").set_index("slice_id")
sel = pd.read_csv(Path(__file__).resolve().parent / "fig2_selection.csv").set_index("region")
ids = (infer / "slice_ids.txt").read_text().split("\n")
sup = np.load(infer / "supervised.npy", mmap_mode="r")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "pdf.fonttype": 42,
    }
)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR with data range 1."""
    return float(10.0 * np.log10(1.0 / np.mean((a - b) ** 2)))


regions = [
    ("chest, soft-tissue window", "Chest\n(soft tissue)"),
    ("upper abdomen (liver, spleen, stomach)", "Upper\nabdomen"),
]
columns = [
    ("Noisy", "noisy"),
    ("N2S", "n2s"),
    ("Equal blend", "equal_n2s_n2n"),
    ("Val.-PSNR judge", "psnr_sel"),
    ("Rank2Denoise", "ours"),
    ("Supervised", None),
    ("GT", "clean"),
]
npz_ids = [str(s) for s in z["slice_ids"]]
fig, axes = plt.subplots(
    len(regions),
    len(columns) + 1,
    figsize=(7.0, 2.25),
    gridspec_kw={"wspace": 0.03, "hspace": 0.14},
)
rows, worst = [], 0.0
for r, (key, label) in enumerate(regions):
    sid, top, left = (
        sel.loc[key, "slice_id"],
        int(sel.loc[key, "roi_top"]),
        int(sel.loc[key, "roi_left"]),
    )
    k = npz_ids.index(sid)
    images = {name: np.asarray(z[src][k], np.float64) for name, src in columns if src}
    images["Supervised"] = np.asarray(sup[ids.index(sid)], np.float64)
    for name, src in (("Rank2Denoise", "ours"), ("Val.-PSNR judge", "psnr_sel")):
        worst = max(
            worst,
            abs(psnr(images[name], images["GT"]) - per_slice.loc[sid, f"psnr_{src}"]),
        )
    window = (slice(top, top + ROI), slice(left, left + ROI))
    crop = images["GT"][window]
    vmin, vmax = float(np.percentile(crop, 1)), float(np.percentile(crop, 99))
    axes[r, 0].imshow(images["GT"], cmap="gray", vmin=0, vmax=1)
    axes[r, 0].add_patch(Rectangle((left, top), ROI, ROI, fill=False, edgecolor="yellow", lw=0.8))
    axes[r, 0].set_ylabel(label, fontsize=9, labelpad=2, linespacing=0.95)
    for c, (name, _) in enumerate(columns, 1):
        ax = axes[r, c]
        ax.imshow(
            images[name][window],
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        if r == 0:
            ax.set_title(
                name.replace("Val.-PSNR judge", "Val.-PSNR\njudge")
                .replace("Equal blend", "Equal\nblend")
                .replace("Rank2Denoise", "Rank2Denoise\n(Ours)"),
                fontsize=8,
                pad=2,
                linespacing=0.9,
            )
        if name != "GT":
            value = psnr(images[name], images["GT"])
            ax.set_xlabel(f"{value:.2f}", fontsize=9, labelpad=1)
            rows.append(
                {
                    "region": key,
                    "slice_id": sid,
                    "method": name,
                    "psnr_full_slice": value,
                    "roi_top": top,
                    "roi_left": left,
                }
            )
    if r == 0:
        axes[r, 0].set_title("GT (full)", fontsize=9, pad=2)
if worst > 1e-6:
    raise SystemExit(f"figure PSNR does not match the CSV: {worst}")
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
pd.DataFrame(rows).to_csv(art / "data" / "fig_zoom_data.csv", index=False)
fig.savefig(art / "figures" / "fig_zoom.pdf", bbox_inches="tight", pad_inches=0.01, dpi=300)
fig.savefig(art / "figures" / "fig_zoom.png", dpi=200, bbox_inches="tight", pad_inches=0.01)
print("check, largest difference (dB):", worst)
