# Choosing Without Ground Truth: A Self-Ranked Vision–Language Judge for Post-hoc Recipes in Self-Supervised Low-Dose CT Denoising

> **Status:** Research code for a manuscript submitted to **IEEE ISBI 2027** (under review).
> This repository is **private**. Please do not share the code, data or results outside the author group.

## Authors

1. **Batuhan Gedik**
2. **Mehmet Ozan Ünal**
3. **Dr. Metin Ertaş**
4. **Assoc. Prof. Dr. İsa Yıldırım**

Department of Electronics and Communication Engineering and
Biomedical Imaging and AI Laboratory (BIAI Lab),
Istanbul Technical University, Istanbul, Turkey

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Method](#method)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Data](#data)
- [Reproducing the Results](#reproducing-the-results)
- [Where Ground Truth Is Used](#where-ground-truth-is-used)
- [Limitations](#limitations)
- [Ethics and Data Use](#ethics-and-data-use)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

Self-supervised denoisers for low-dose CT (LDCT) learn from noisy images only. However, their
outputs are usually **tuned by measuring PSNR against clean reference images**. Once this
happens, the pipeline is no longer free of ground truth (GT).

This project replaces the GT-based judge with a **frozen medical vision–language model (VLM)**.
The judge is a MedSigLIP-448 image embedding, projected onto a *"less noisy"* direction. This
direction is learned **only from noisy slices and their re-noised copies**. It uses no clean
image and no text.

The judge selects a short post-hoc **recipe**: a sequence of small blending moves between two
self-supervised experts:

- **C0**: a J-invariant U-Net trained with the Noise2Self loss, and
- **N2N**: the same U-Net trained with the Neighbor2Neighbor loss.

The recipe is selected on validation slices **without reading their clean images**. At test time,
the fixed recipe needs neither the VLM nor GT; it only adds pixel operations.

## Key Results

Results on 220 test slices (31 patients), mean ± std over three C0 seeds (42, 123, 456).
All numbers are taken from `article/figures/table_main.tex` and `article/figures/table_judges.tex`.

| Method | GT used | PSNR (dB) ↑ | Δ vs. C0 (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- | --- |
| Noisy input | – | 23.04 | – | 0.742 | 0.284 |
| N2N (seed 42) | no | 27.84 | – | 0.815 | 0.136 |
| C0 (J-invariant) | no | 28.54 ± 0.05 | 0 | 0.826 | 0.113 |
| C0 + VLM loss term (seed 42) | no | 28.55 | +0.06 | 0.818 | 0.114 |
| **Ours, global recipe** | **no** | **29.04 ± 0.06** | **+0.51 ± 0.02** | **0.841** | **0.101** |
| *Q*-selected, global | no | 29.05 ± 0.03 | +0.51 ± 0.03 | 0.840 | 0.101 |
| PSNR-selected, global | val | 29.07 ± 0.02 | +0.54 ± 0.02 | 0.841 | 0.100 |
| GT oracle (per slice, not a method) | test | 29.33 ± 0.03 | +0.79 ± 0.02 | 0.854 | 0.079 |

**Main findings**

- Without any clean image, the VLM-selected recipe improves **every test slice and every
  patient in all three seeds** (smallest slice gain: +0.29 to +0.33 dB).
- It keeps **87–101%** of the gain of the same search judged by validation PSNR.
- The same VLM used as a **training loss** adds at most **+0.06 dB**. In other words, the VLM is
  useful as a *judge*, not as a *loss*.
- On the C0–N2N blend line, our judge is the best GT-free judge we tested. It beats random
  directions, a text-prompt score, an equal blend and the classical no-reference index *Q*
  (+0.02 dB, 95% CI [+0.01, +0.04]).
- It **ties** with *Q* in the full greedy search and **fails** in a small wide-window group
  (see [Limitations](#limitations)).

## Method

The method has three stages (Fig. 1 of the paper).

**(a) Judge direction from train-noisy slices.**
For every train-noisy slice $y_i$, we take the noise field of C0, $n_i = y_i - \hat{x}_J(y_i)$.
A re-noised copy adds the field of another random slice $k$, flipped left–right with probability
0.5: $y_i^{+} = \mathrm{clip}(y_i + \pi(n_k), 0, 1)$. Because $y_i^{+}$ is noisier than $y_i$,
the mean embedding difference points towards lower noise:

$$
w = \frac{\sum_i \big(\phi(y_i) - \phi(y_i^{+})\big)}{\big\lVert \sum_i \big(\phi(y_i) - \phi(y_i^{+})\big) \big\rVert_2},
\qquad s(x) = \langle \phi(x), w \rangle
$$

Here $\phi(\cdot) \in \mathbb{R}^{1152}$ is the $\ell_2$-normalised MedSigLIP-448 image embedding.
Before the judge sees an image, the four $8 \times 8$ corners are set to zero. Each C0 seed has
its own $w$.

**(b) Greedy recipe search on validation slices.**
Starting from the C0 output, five candidate moves are tested at each step:

| Move | Definition |
| --- | --- |
| Blend towards noisy input | $x + \alpha(y - x)$ |
| Blend towards N2N | $x + \alpha(\hat{x}_N - x)$ |
| Blend towards C0 | $x + \alpha(\hat{x}_J - x)$ |
| Gaussian smoothing | $g_\sigma * x$, $\sigma = 0.5$ px |
| High-frequency boost | $x + \alpha\,\mathcal{F}^{-1}[\mathbb{1}_{\lVert\kappa\rVert > 0.25}\,\mathcal{F}x]$ |

with $\alpha = 0.1$. The move with the highest mean judge score $\bar{s}$ is kept. The search
stops when $\bar{s}$ does not increase, or after 30 moves.

**(c) Test.** The fixed recipe is applied to the C0 and N2N outputs. No VLM and no GT are needed.

## Repository Structure

Paths are relative to the project root (`W17 ISBI_2027_plan/`). Only the parts used by the
final paper are listed.

```
.
├── src/                          # Shared library: U-Net, masking, Neighbor2Neighbor,
│                                 #   re-noising, metrics, VLM wrappers, training loop
├── scripts/                      # Step 1.x / 2.x notebook generators and helpers
│   ├── make_step_1_4_notebook.py #   trains C0 (seed 42), N2N and the supervised U-Net
│   ├── make_step_2_2_notebook.py #   builds the self-ranked direction w
│   └── make_rank.py              #   computation of w and validation scores
├── Diagnostics_2/
│   ├── diag2_direction_search.py # The five recipe moves and the greedy search
│   ├── composite_vlm/
│   │   ├── src/                  # Copy of the shared library used by Diagnostics_2
│   │   ├── scripts/              # rank_recipe.py, masked_rerun.py, masked_judges.py, ...
│   │   └── make_*_notebook.py    # Colab notebook generators (one per experiment)
│   ├── d2_19_judges/             # GT-free judge comparison on the C0–N2N blend line
│   ├── d2_26_results/            # Corner-masked judge: w + recipe + test (3 seeds)
│   └── d2_27_results/            # Corner-masked judges + dose check
├── article/
│   ├── main.tex, references.bib  # ISBI 2027 manuscript (official spconf template)
│   ├── scripts/                  # Local inference, metrics, tables, figures, statistics
│   ├── data/                     # CSV files behind every number in the paper
│   └── figures/                  # Tables (.tex), figures and README_numbers.md
└── tests/                        # Unit tests
```

> **Notebooks are generated, not edited by hand.** Each `.ipynb` file is produced by a
> `make_*_notebook.py` generator, which embeds the code from `src/` and `scripts/`. If these
> files change, the notebook must be generated again.

## Requirements

The GPU experiments were run on **Google Colab with an NVIDIA A100-SXM4-40GB**. The recorded
environment of the judge experiments is:

| Package | Version |
| --- | --- |
| Python | 3.13.15 |
| PyTorch | 2.11.0 (CUDA 13.0) |
| torchvision | 0.26.0 |
| transformers | 4.57.6 (the notebooks install `transformers<5`) |
| open_clip_torch | 3.3.0 |
| huggingface_hub | 0.36.2 |
| NumPy / SciPy / pandas | 2.1.3 / 1.16.3 / 2.2.3 |
| scikit-image | 0.25.2 |

The full `pip freeze` output is in `Diagnostics_2/judge_search/requirements_judge_search.txt`.

**MedSigLIP access.** The judge uses [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448),
which is a gated model on Hugging Face. You must accept its terms of use and log in with a
Hugging Face token before running any VLM notebook.

The local scripts in `article/scripts/` run on CPU and need only NumPy, pandas, SciPy,
PyTorch, scikit-image and matplotlib.

## Data

We use **Mini-DeepLesion** from the **BIAILAB I2I 2025v1** release, which is derived from
DeepLesion (Yan et al., 2018).

- $512 \times 512$ slices, single channel, values in $[0, 1]$, stored as `.npz` files.
- Simulated Poisson noise at $10^5$ photons (key `noisy_clipped_100k`); clean reference
  under `gt_clipped`.
- The noise is **spatially correlated** (lag-1 autocorrelation: 0.19 horizontal, 0.10 vertical).
- **Patient-level split:** 647 train-noisy slices (80 patients), 182 validation (34),
  220 test (31).
- **Window groups** (rule: minimum of the noisy input $> 0.1$ → wide): test 192 narrow / 28 wide;
  validation 137 / 45.

The dataset is **not included** in this repository. Expected location on Colab:

```
/content/drive/MyDrive/BI_AI/ayiklanan_dosyalar/minideeplesion/{train,val,test}/*.npz
```

## Reproducing the Results

### 1. Train the experts (Colab, GPU)

| Notebook generator | What it does |
| --- | --- |
| `scripts/make_step_1_4_notebook.py` | Trains C0 (seed 42), N2N (seed 42) and the supervised U-Net reference |
| `Diagnostics_2/composite_vlm/make_c0_seeds_notebook.py` | Trains C0 with seeds 123 and 456 (no VLM; a T4 GPU is enough) |

Training setup: U-Net with 7.70 M parameters, AdamW, learning rate $10^{-4}$, batch size 8,
30 epochs. The checkpoint is selected on the masked loss computed on **noisy** validation data.
All training runs can be resumed at epoch level.

### 2. Build the judge and select the recipe (Colab, A100)

```bash
python3 Diagnostics_2/composite_vlm/make_masked_rerun_notebook.py   # w + recipe + test, corner-masked judge, 3 seeds
python3 Diagnostics_2/composite_vlm/make_masked_judges_notebook.py  # Table 2 embeddings + dose check
```

Run the generated notebooks on Colab. Recipe selection takes about **10–15 minutes per seed**
on one A100.

### 3. Rebuild tables, figures and paper numbers (local, CPU)

All scripts are run in isolated mode (`python -I`). Every number in `main.tex` is read from a CSV
file; the mapping is documented in `article/figures/README_numbers.md`.

```bash
# Local test inference: C0 (3 seeds), N2N and supervised outputs
python -I article/scripts/infer_test.py      <project_root> <minideeplesion/test> <infer_dir>

# Per-slice PSNR / SSIM / LPIPS / R_H  →  data/table_per_slice.csv, data/table_per_seed.csv
python -I article/scripts/table_metrics.py   <project_root> <infer_dir> <out_dir>

# Table 1 (LaTeX)
python -I article/scripts/make_table.py      <table_per_seed.csv> <direction_search_summary.csv> \
                                             <q_global_metrics_per_slice.csv> article/figures/table_main.tex

# Table 2 (LaTeX)
python -I article/scripts/make_judge_table.py <judge_summary.csv> article/figures/table_judges.tex

# Numbers in the text: gains, patient-cluster bootstrap CIs, Wilcoxon tests
python -I article/scripts/paper_numbers.py   <table_per_slice.csv> <out.csv>
python -I article/scripts/compare_judges.py  <project_root> <infer_dir> <out_dir>

# Figure 2 (zoomed test examples) and parity checks
python -I article/scripts/make_figures.py    <Diagnostics_2> <infer_dir> <out_dir>

# Check that the window threshold could be chosen from train/val inputs alone
python -I article/scripts/window_threshold_check.py <minideeplesion_root> <out.csv>
```

### 4. Compile the paper

The manuscript uses the official ISBI LaTeX kit (`spconf.sty`, `IEEEbib.bst`):

```bash
cd article
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### Evaluation protocol

- PSNR and SSIM use `data_range = 1.0`; LPIPS uses AlexNet.
- $R_H$ is the output/GT power ratio in the 0.25–0.5 cycles/pixel band
  ($< 1$: over-smoothed, $> 1$: residual noise).
- Confidence intervals are **patient-cluster bootstrap** 95% CIs (5000 draws) of paired per-slice
  differences.
- Seeds: 42, 123, 456.

## Where Ground Truth Is Used

GT-free operation is the central claim of this work, so we list every use of clean images:

| Stage | GT used? |
| --- | --- |
| Training C0 and N2N | **No** |
| Building the judge direction $w$ | **No** |
| Selecting the recipe on validation data | **No** (only `noisy_clipped_100k` is loaded) |
| Applying the recipe at test time | **No** |
| Supervised U-Net reference | Yes (training) |
| PSNR-selected and GT-oracle references | Yes (validation / test GT) |
| Dose check of the judge (simulated doses) | Yes; it does not change $w$ |
| All reported test metrics | Yes (evaluation only) |

Three design choices followed test results: one $w$ per seed, the corner mask, and the global
recipe as the main form. For this reason, the test numbers are not a fully independent estimate.

## Limitations

- The noise is **simulated**; we use **one dataset** and **one N2N run**.
- The judge is the weak point. In the wide-window group (28 test slices), a group recipe selected
  by the VLM judge starts with Gaussian smoothing and **loses 0.15 ± 0.02 dB**, while the global
  recipe gains +0.58 ± 0.04 dB. Here the VLM score rewards blur.
- With one shared $w$ for all seeds, one seed lost 0.24 dB; without the corner mask, seed 123
  gained only +0.14 dB.
- A second dataset is needed to confirm the fixed method.

## Ethics and Data Use

This retrospective study uses only public DeepLesion images (BIAILAB I2I 2025v1 release).
No new data were collected and no separate ethical approval was obtained. Users of this code must
follow the DeepLesion terms of use.

## Acknowledgments

No funding was received, and the authors declare no conflicts of interest. Generative AI
(Claude, Anthropic) helped write analysis code and draft and edit the text; the authors checked
all code, numbers and text.

## Contact

For questions about this repository, please contact the first author, **Batuhan Gedik**
(Biomedical Imaging and AI Laboratory, Istanbul Technical University).
