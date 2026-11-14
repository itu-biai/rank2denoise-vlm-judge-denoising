# src/masking.py
import torch
from torch import nn


def checkerboard_mask(height: int, width: int, phase: int, device: torch.device) -> torch.Tensor:
    """Checkerboard mask of shape (1, 1, H, W); 1 where (row + col) % 2 == phase."""
    rows = torch.arange(height, device=device).view(-1, 1)
    cols = torch.arange(width, device=device).view(1, -1)
    return (((rows + cols) % 2) == phase).float().view(1, 1, height, width)


def masked_input(noisy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The noisy image with the masked pixels set to zero."""
    return noisy * (1.0 - mask)


def masked_recon_loss(prediction: torch.Tensor, noisy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error to the noisy image on the masked pixels only."""
    return (((prediction - noisy) ** 2) * mask).sum() / (mask.sum() * noisy.size(0))


@torch.no_grad()
def j_invariant_inference(model: nn.Module, noisy: torch.Tensor) -> torch.Tensor:
    """Denoise with the two checkerboard phases; each pixel comes from the pass that masks it."""
    _, _, height, width = noisy.shape
    output = torch.zeros_like(noisy)
    for phase in (0, 1):
        mask = checkerboard_mask(height, width, phase, noisy.device)
        output += model(masked_input(noisy, mask)) * mask
    return output
