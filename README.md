# Rank2Denoise: Choosing Band-wise Post-hoc Recipes without Ground Truth for Self-supervised Low-dose CT Denoising

> Manuscript submitted to **IEEE ISBI 2027** (4 pages, references included).
> Code: <https://github.com/itu-biai/Rank2Denoise-VLM-judge-denoising>

## The problem

Paired low-dose / normal-dose CT data are rare. For this reason, self-supervised denoisers
(Noise2Self, Noise2Void, Neighbor2Neighbor, Recorrupted-to-Recorrupted, Blind2Unblind, Self2Self)
learn from noisy images only. Each method has its own noise assumption. Reconstructed CT noise is
spatially correlated, so different denoisers remove different parts of the noise.

To choose and combine these denoisers, we need a **judge**. In practice, this judge is PSNR
against clean images. Thus the ground truth (GT), which self-supervision avoids, comes back at the
selection step.

Existing GT-free judges do not fit this setting:

- Risk estimates such as PURECOMB assume independent noise.
- The J-invariant loss scores one network, not a mix of networks.
- Learned quality scores need other supervision: CLIP-IQA uses text, RAVE uses sets of good and
  poor images, and SSIQA trains a regressor of CT noise level.

**Goal:** choose and combine self-supervised denoisers **without any clean image**.

## The idea

When we add noise to a noisy slice, the copy is surely worse than the slice. In the embedding space
of a frozen medical vision–language model (**MedSigLIP-448**), the mean difference between noisy
slices and their noisier copies is therefore a **"less noisy" direction**. We learn this direction
with no text, no clean image and no explicit noise model.

The projection on this direction gives a score. The score **ranks** candidate images (hence the
name *Rank2Denoise*) and steers a greedy search over small post-hoc moves towards several denoisers.
The judge only selects the recipe. At test time, the fixed recipe needs no VLM and no GT.

## Method

The method has three stages (Fig. 1 of the paper).

### (a) Judge direction — noisy training slices, no GT

For each noisy training slice $y_i$, the N2S noise field is $n_i = y_i - \hat{x}_S(y_i)$.
A re-noised copy adds the noise field of another random slice $k$, randomly flipped ($\pi$):
$y_i^{+} = \mathrm{clip}(y_i + \pi(n_k), 0, 1)$. Because $y_i^{+}$ is noisier, the mean embedding
difference points towards cleaner images:

$$
w = \frac{\sum_i \big(\phi(y_i) - \phi(y_i^{+})\big)}{\big\lVert \sum_i \big(\phi(y_i) - \phi(y_i^{+})\big) \big\rVert_2},
\qquad s(x) = \langle \phi(x), w \rangle
$$

$\phi$ is the L2-normalised MedSigLIP-448 image embedding. Before scoring, the $8\times8$ corners
of the slice are set to zero; the slice is resized to $448\times448$, copied to three channels and
scaled to $[-1, 1]$. A larger $s$ means a less noisy image. Each N2S seed has its own $w$.

### (b) Two-stage greedy search — validation slices, no GT

A recipe is a list of **moves**. Each move changes the current image a little (step $\alpha = 0.1$)
and is clipped to $[0, 1]$.

- **Stage 1** starts from the N2S output and uses five global moves: blend towards the noisy input,
  the N2N output or the N2S output; Gaussian smoothing ($\sigma = 0.5$ px); a high-frequency boost
  (ideal radial high-pass above 0.25 cycles/pixel).
- **Stage 2** adds 15 band moves: one band (low, mid or high) of the difference to one of five
  experts (wavelet, N2V, R2R, S2S, B2U). The low band is made zero-mean, so no move shifts the mean
  grey level.
- **Region rule.** Band moves are not applied on the background and on organ boundaries. The
  region mask $m$ is computed once from the N2S output, without GT. Without this mask, band moves
  shift the background grey level and blur boundaries.

At each step, the move with the best mean score $\bar{s}$ over the validation slices is kept. A
stage stops when $\bar{s}$ no longer rises, or after 30 moves.

### (c) Test — fixed recipe, no VLM, no GT

The selected recipe is applied to the outputs of the seven experts.

### Experts

| Expert | Description |
| --- | --- |
| N2S | U-Net, Noise2Self loss (two complementary checkerboard masks) — main expert |
| N2N | Same U-Net, Neighbor2Neighbor loss |
| N2V | Noise2Void |
| R2R | Recorrupted-to-Recorrupted (re-corrupted with N2S noise fields) |
| B2U | Blind2Unblind |
| S2S | Dataset-level Self2Self variant |
| Wavelet | BayesShrink wavelet thresholding |

## Data and setup

- 1301 DeepLesion slices ($512\times512$) with **10% dose** simulated following LoDoPaB-CT.
- Each slice is clipped to its display window (mostly $[-175, 275]$ HU; $[-1500, 500]$ HU for 196
  lung slices) and scaled to $[0, 1]$.
- The noise is correlated: lag-1 autocorrelation 0.19 (horizontal) / 0.10 (vertical).
- **Patient-level split:** 647 noisy training slices (80 patients), 182 validation (34),
  220 test (31).
- N2S, N2N, N2V, R2R and B2U share one U-Net (7.70 M parameters; AdamW, learning rate $10^{-4}$,
  batch 8, 30 epochs). Checkpoints are selected by their own loss on **noisy** validation data.
- Seeds 42, 123 and 456. The search takes under one hour per seed on an A100.
- PSNR and SSIM use a data range of 1; LPIPS uses AlexNet. Statistics: exact Wilcoxon signed-rank
  test on per-patient paired differences and a patient-cluster bootstrap 95% CI (5000 draws).

## Main results

Test set (220 slices, 31 patients), mean ± sample std over three seeds. *Gain*: PSNR gain (dB) over
the equal blend (0.5 N2S + 0.5 N2N). Source: `figures/table_main.tex`.

| Method | GT used | PSNR (dB) ↑ | Gain | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- | --- |
| Noisy input | – | 23.04 | – | 0.742 | 0.284 |
| Supervised (same U-Net, MSE) | train | 31.47 | – | 0.874 | 0.113 |
| N2S | no | 28.54 ± 0.05 | −0.48 | 0.826 | 0.113 |
| S2S (best single denoiser) | no | 28.57 | −0.45 | 0.847 | 0.185 |
| Equal blend | no | 29.02 ± 0.02 | 0 | 0.841 | 0.102 |
| Equal blend, 7 experts | no | 28.48 ± 0.02 | −0.54 | 0.825 | 0.124 |
| Band moves, no judge | no | 28.99 ± 0.02 | −0.03 | 0.838 | 0.106 |
| **Rank2Denoise (ours)** | **no** | **29.49 ± 0.01** | **+0.47** | **0.853** | **0.098** |
| Residual-variance judge | no | 28.84 ± 0.05 | −0.18 | 0.836 | 0.105 |
| Random direction | no | 27.70 ± 0.86 | −1.32 | 0.827 | 0.141 |
| ResNet-50 direction | no | 27.66 ± 0.05 | −1.36 | 0.830 | 0.180 |
| Val.-PSNR judge (GT reference) | val | 29.66 ± 0.01 | +0.64 | 0.862 | 0.113 |

**Key findings**

- The recipe improves PSNR over the equal blend by **+0.47 dB** ($p < 10^{-7}$; 29 of 31 patients)
  and SSIM from 0.841 to 0.853 ($p < 10^{-5}$).
- It raises PSNR over N2S on **every one** of the 220 test slices and 31 patients.
- It beats the best single denoiser (S2S) by **0.92 dB**, and a residual-variance judge in the same
  search by **0.65 dB**.
- It recovers **73%** of the gain of the GT reference judge, with a lower LPIPS (0.098 vs. 0.113).
- Plain ensembling does not explain the gain: the average of all seven experts is below N2S, and
  band moves without a judge stay at the equal blend.
- The learned direction drives the gain: with a random or ResNet-50 direction, the search ends
  below N2S.

## Where ground truth is used

| Step | GT used? |
| --- | --- |
| Training the self-supervised experts | No |
| Building the judge direction $w$ | No |
| Region mask $m$ | No |
| Selecting the recipe on validation slices | No |
| Applying the recipe at test time | No |
| Supervised reference | Yes (training) |
| Val.-PSNR reference judge | Yes (validation GT) |
| Test metrics | Yes (evaluation only) |

## Scope

- Simulated noise at one dose, on one dataset, with one expert pool.
- The clinical-dose GT is itself noisy (most visible in lung windows).
- The region rule was designed after inspecting GT-based region errors of the unmasked test search.
  Its validation signs support it, but a fresh test set should confirm its gain.
- Next step: real scans and other doses.

## Ethics

This retrospective study uses only public DeepLesion images with simulated noise. No new data were
collected; no separate ethical approval was obtained.

## Acknowledgments

No funding; no conflicts of interest. Generative AI (Claude, Anthropic) helped write code and text;
the authors checked all of it.

## Authors and contributions

| Author | Affiliation | Contribution |
| --- | --- | --- |
| **Batuhan Gedik** | ¹ ² | Planning, method development, implementation, experiments, analysis and writing |
| Mehmet Ozan Ünal | ¹ ² | Review |
| Metin Ertaş | ² | Review |
| İsa Yıldırım | ¹ ² | Review |

¹ Department of Electronics and Communication Engineering, Istanbul Technical University
² Biomedical Imaging and AI Laboratory (BIAI Lab), Istanbul, Turkey
