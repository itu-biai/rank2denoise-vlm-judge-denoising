# scripts/expert_methods.py
"""Networks, losses and inference protocols of the self-supervised experts (torch only; no file I/O).

Experts of the paper trained here: N2V, B2U and R2R (same U-Net and training setup as N2S), plus the dataset-level
Self2Self network. Method-specific settings follow the original papers; the source of each constant is given next to it.
R2R uses N2S noise fields as its noise model. The B2U lambda schedule of the official code is scaled to the training length (30 epochs).

No GT is used in this file: every loss is computed from noisy images only (and from the GT-free N2S noise fields).
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TRAINED_METHODS: tuple[str, ...] = ("n2v", "b2u", "r2r")

# ----------------------------------------------------------------------------- Noise2Void (Krull 2019)
N2V_CELL = 8  # Sec. 4: N = 64 pixels in a 64x64 patch -> one pixel per 8x8 cell (stratified)
N2V_UPS_RADIUS = 2  # UPS 5x5 (N2V definition in Xie 2020, Table 1; centre included, Sec. 3.1)

# ----------------------------------------------------------------------------- Blind2Unblind (Wang 2022)
B2U_CELL = 2  # Sec. 4.1 "we set s = 2"
B2U_ETA = 1.0  # Sec. 5.1 eta = 1
B2U_LAMBDA_START, B2U_LAMBDA_END = 2.0, 20.0  # Sec. 5.1 lambda_s = 2, lambda_f = 20
B2U_RAMP_START = 0.8  # official `train_b2u.py`: lambda goes 2 -> 20 in the last 20% of training (Thread1 = 0.8)
B2U_KERNEL = (
    torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], [0.5, 1.0, 0.5]]) / 6.0
)  # App. B + official `interpolate_mask`

# ----------------------------------------------------------------------------- R2R (Pang 2021)
R2R_ALPHA = 0.5  # Sec. 5.1 "D = alpha I ... alpha = 0.5"
R2R_SAMPLES = 50  # Sec. 5.1 "we adopt (16) with T = 50"

# ----------------------------------------------------------------------------- Self2Self (dataset-level variant)
S2S_KEEP_PROB, S2S_DROPOUT, S2S_SAMPLES = (
    0.7,
    0.3,
    50,
)  # keep probability, dropout, number of passes
S2S_EVAL_BATCH, S2S_SELECTION_SEED = 8, 4242  # evaluation batch, selection seed


# ============================================================================= shared masking


def stratified_mask(shape: tuple[int, ...], cell: int, generator: torch.Generator) -> torch.Tensor:
    """(B, 1, H, W) 0/1 mask: exactly one random pixel in every cell x cell block (edge cells are cropped)."""
    batch, _, height, width = shape
    rows, cols = math.ceil(height / cell), math.ceil(width / cell)
    offset_y = torch.randint(0, cell, (batch, rows, cols), generator=generator)
    offset_x = torch.randint(0, cell, (batch, rows, cols), generator=generator)
    ys = (torch.arange(rows).view(1, rows, 1) * cell + offset_y).clamp(max=height - 1)
    xs = (torch.arange(cols).view(1, 1, cols) * cell + offset_x).clamp(max=width - 1)
    mask = torch.zeros(batch, height, width)
    mask[torch.arange(batch).view(batch, 1, 1).expand(batch, rows, cols), ys, xs] = 1.0
    return mask.unsqueeze(1)


def ups_fill(noisy: torch.Tensor, mask: torch.Tensor, radius: int, generator: torch.Generator) -> torch.Tensor:
    """N2V UPS: each masked pixel takes the value of a random pixel in its (2r+1)^2 neighbourhood (centre included)."""
    batch, _, height, width = noisy.shape
    dy = torch.randint(-radius, radius + 1, (batch, 1, height, width), generator=generator)
    dx = torch.randint(-radius, radius + 1, (batch, 1, height, width), generator=generator)
    yy = (torch.arange(height).view(1, 1, height, 1) + dy).clamp(0, height - 1)
    xx = (torch.arange(width).view(1, 1, 1, width) + dx).clamp(0, width - 1)
    index = (yy * width + xx).view(batch, 1, -1).to(noisy.device)
    source = torch.gather(noisy.reshape(batch, 1, -1), 2, index).view_as(noisy)
    mask = mask.to(noisy.device)
    return noisy * (1.0 - mask) + source * mask


# ============================================================================= losses (U-Net methods)


def n2v_loss(model: nn.Module, noisy: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Noise2Void loss: masked pixels replaced from their neighbourhood, MSE on the masked pixels."""
    mask = stratified_mask(noisy.shape, N2V_CELL, generator).to(noisy.device)
    prediction = model(ups_fill(noisy, mask, N2V_UPS_RADIUS, generator))
    return (((prediction - noisy) ** 2) * mask).sum() / mask.sum()


def b2u_lambda(progress: float) -> float:
    """progress = (epoch + 1) / epochs. Schedule of the official code, scaled to the training length."""
    if progress <= B2U_RAMP_START:
        return B2U_LAMBDA_START
    return B2U_LAMBDA_START + (progress - B2U_RAMP_START) * (B2U_LAMBDA_END - B2U_LAMBDA_START) / (1.0 - B2U_RAMP_START)


def b2u_blind(model: nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    """h(f(Omega_y)): one masked copy per position of the s x s cell; masked pixels are filled with the neighbour mean (App. B)."""
    kernel = B2U_KERNEL.to(noisy.device, noisy.dtype).view(1, 1, 3, 3)
    interpolated = F.conv2d(noisy, kernel, padding=1)  # official code: zero padding, padding = 1
    blind = torch.zeros_like(noisy)
    for i in range(B2U_CELL):
        for j in range(B2U_CELL):
            mask = torch.zeros_like(noisy)
            mask[..., i::B2U_CELL, j::B2U_CELL] = 1.0
            blind = blind + model(noisy * (1.0 - mask) + interpolated * mask) * mask
    return blind


def b2u_loss(model: nn.Module, noisy: torch.Tensor, lam: float) -> torch.Tensor:
    """Eq. 12: ||h + lambda f_hat(y) - (lambda + 1) y||^2 + eta ||h - y||^2; no gradient through f_hat (Sec. 4.2)."""
    diff = b2u_blind(model, noisy) - noisy
    with torch.no_grad():
        visible = model(noisy) - noisy
    return B2U_ETA * (diff**2).mean() + ((diff + lam * visible) ** 2).mean()


def r2r_loss(model: nn.Module, noisy: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Eq. 14, D = alpha I: input y + alpha n, target y - n / alpha. n = N2S noise field of another slice (keeps the correlation)."""
    return F.mse_loss(model(noisy + R2R_ALPHA * noise), noisy - noise / R2R_ALPHA)


# ============================================================================= inference (trained methods)


@torch.no_grad()
def infer_trained(
    method: str,
    model: nn.Module,
    noisy: torch.Tensor,
    generator: torch.Generator,
    noise_bank: NoiseBank | None = None,
) -> torch.Tensor:
    """Inference protocol of each method; the result is clipped to [0, 1]."""
    if method in ("n2v", "b2u"):  # single pass (N2V, B2U Fig. 1b)
        output = model(noisy)
    elif method == "r2r":  # Eq. 16, T = 50
        output = torch.zeros_like(noisy)
        for _ in range(R2R_SAMPLES):
            output += model(noisy + R2R_ALPHA * noise_bank.sample(len(noisy), None, generator).to(noisy.device))
        output /= R2R_SAMPLES
    else:
        raise ValueError(f"unknown method: {method}")
    return output.clamp(0.0, 1.0)


# ============================================================================= N2S noise-field bank


class NoiseBank:
    """N2S noise fields of the noisy training slices, n_k = y_k - x_J(y_k) (no GT). A sample is the field of another random
    slice, mirrored left-right with probability 1/2 (same pi as in the judge direction w).
    """

    def __init__(self, fields: torch.Tensor) -> None:
        if fields.dim() != 4 or fields.shape[1] != 1:
            raise ValueError(f"field shape must be (N, 1, H, W), found {tuple(fields.shape)}")
        self.fields = fields

    def sample(self, count: int, exclude: torch.Tensor | None, generator: torch.Generator) -> torch.Tensor:
        """Noise fields of random slices (other than `exclude`), each mirrored left-right with probability 1/2."""
        index = torch.randint(0, len(self.fields), (count,), generator=generator)
        if exclude is not None:  # another slice, not its own field
            clash = index == exclude
            index[clash] = (
                index[clash] + 1 + torch.randint(0, len(self.fields) - 1, (int(clash.sum()),), generator=generator)
            ) % len(self.fields)
        noise = self.fields[index]
        flip = torch.rand(count, generator=generator) < 0.5
        noise[flip] = torch.flip(noise[flip], dims=[-1])
        return noise


# ============================================================================= Self2Self network


class PartialConv2d(nn.Module):
    """Partial convolution over the visible pixels of a mask, renormalised by the visible count per window."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps
        self.window_size = float(kernel_size * kernel_size * in_channels)
        std = float(np.sqrt(2.0) / np.sqrt(kernel_size * kernel_size * in_channels))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.normal_(self.weight, mean=0.0, std=std)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer(
            "ones_kernel",
            torch.ones(out_channels, in_channels, kernel_size, kernel_size),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Partial convolution renormalised by the visible pixels; returns the output and the updated mask."""
        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        m_pad = F.pad(mask, (pad, pad, pad, pad), mode="constant", value=1.0)
        with torch.no_grad():
            valid = F.conv2d(m_pad, self.ones_kernel)
            new_mask = torch.clamp(valid, 0.0, 1.0)
            mask_ratio = (self.window_size / (valid + self.eps)) * new_mask
        out = F.conv2d(x_pad * m_pad, self.weight)
        out = out * mask_ratio + self.bias.view(1, -1, 1, 1)
        return out, new_mask


class DropoutConv2d(nn.Module):
    """Decoder convolution of the Self2Self network: dropout, convolution and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float,
        activation: str = "lrelu",
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        self.drop = nn.Dropout(p=dropout_p)
        gain = float(np.sqrt(2.0)) if activation == "lrelu" else 1.0
        std = gain / float(np.sqrt(kernel_size * kernel_size * in_channels))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        nn.init.normal_(self.conv.weight, mean=0.0, std=std)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dropout, replicate padding, convolution, then leaky ReLU or sigmoid."""
        x = self.drop(x)
        pad = self.kernel_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        x = self.conv(x)
        if self.activation == "lrelu":
            return F.leaky_relu(x, negative_slope=0.1)
        return torch.sigmoid(x)


class S2SPartialConvUNet(nn.Module):
    """Self2Self U-Net with a partial-convolution encoder and a dropout decoder."""

    def __init__(self, channels: int = 1, dropout_p: float = 0.3, width: int = 48) -> None:
        super().__init__()
        self.enc0 = PartialConv2d(channels, width)
        self.enc1 = PartialConv2d(width, width)
        self.enc2 = PartialConv2d(width, width)
        self.enc3 = PartialConv2d(width, width)
        self.enc4 = PartialConv2d(width, width)
        self.enc5 = PartialConv2d(width, width)
        self.enc6 = PartialConv2d(width, width)
        self.dec5 = DropoutConv2d(width * 2, 96, dropout_p)
        self.dec5b = DropoutConv2d(96, 96, dropout_p)
        self.dec4 = DropoutConv2d(96 + width, 96, dropout_p)
        self.dec4b = DropoutConv2d(96, 96, dropout_p)
        self.dec3 = DropoutConv2d(96 + width, 96, dropout_p)
        self.dec3b = DropoutConv2d(96, 96, dropout_p)
        self.dec2 = DropoutConv2d(96 + width, 96, dropout_p)
        self.dec2b = DropoutConv2d(96, 96, dropout_p)
        self.dec1a = DropoutConv2d(96 + channels, 64, dropout_p)
        self.dec1b = DropoutConv2d(64, 32, dropout_p)
        self.dec_out = DropoutConv2d(32, channels, dropout_p, activation="sigmoid")

    @staticmethod
    def _pool(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return F.max_pool2d(x, 2), F.max_pool2d(mask, 2)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, x_masked: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Partial-convolution U-Net on the masked input and its mask."""
        skips = [x_masked]
        n, m = self.enc0(x_masked, mask)
        n = F.leaky_relu(n, 0.1)
        n, m = self.enc1(n, m)
        n = F.leaky_relu(n, 0.1)
        n, m = self._pool(n, m)
        skips.append(n)
        n, m = self.enc2(n, m)
        n = F.leaky_relu(n, 0.1)
        n, m = self._pool(n, m)
        skips.append(n)
        n, m = self.enc3(n, m)
        n = F.leaky_relu(n, 0.1)
        n, m = self._pool(n, m)
        skips.append(n)
        n, m = self.enc4(n, m)
        n = F.leaky_relu(n, 0.1)
        n, m = self._pool(n, m)
        skips.append(n)
        n, m = self.enc5(n, m)
        n = F.leaky_relu(n, 0.1)
        n, m = self._pool(n, m)
        n, m = self.enc6(n, m)
        n = F.leaky_relu(n, 0.1)
        n = torch.cat([self._up(n), skips.pop()], dim=1)
        n = self.dec5b(self.dec5(n))
        n = torch.cat([self._up(n), skips.pop()], dim=1)
        n = self.dec4b(self.dec4(n))
        n = torch.cat([self._up(n), skips.pop()], dim=1)
        n = self.dec3b(self.dec3(n))
        n = torch.cat([self._up(n), skips.pop()], dim=1)
        n = self.dec2b(self.dec2(n))
        n = torch.cat([self._up(n), skips.pop()], dim=1)
        n = self.dec1b(self.dec1a(n))
        return self.dec_out(n)


def _random_flip(x: torch.Tensor, flip_lr: bool, flip_ud: bool) -> torch.Tensor:
    if flip_lr:
        x = torch.flip(x, dims=[-1])
    if flip_ud:
        x = torch.flip(x, dims=[-2])
    return x


@torch.no_grad()
def s2s_ensemble(model: nn.Module, images: torch.Tensor, seed: int) -> torch.Tensor:
    """Self2Self ensemble: 50 passes; each pass applies a random flip -> Bernoulli mask (keep 0.7) -> dropout on ->
    flip back. Fixed seed per call; the global RNG state is restored afterwards.
    """
    was_training = model.training
    model.train()  # dropout stays on at inference (Self2Self)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py_state = random.getstate()
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        total = torch.zeros_like(images)
        for _ in range(S2S_SAMPLES):
            flip_lr, flip_ud = bool(random.getrandbits(1)), bool(random.getrandbits(1))
            x = _random_flip(images, flip_lr, flip_ud)
            mask = (torch.rand(x.shape, device=images.device) < S2S_KEEP_PROB).to(torch.float32)
            total += _random_flip(model(x * mask, mask), flip_lr, flip_ud)
        return total / float(S2S_SAMPLES)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        random.setstate(py_state)
        model.train(was_training)
