# src/ne2ne.py
import torch
import torch.nn.functional as F

NEIGHBOURS = torch.tensor([[1, 2], [0, 3], [0, 3], [1, 2]])


def neighbour_indices(noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Random pair of neighbouring positions in every 2x2 cell."""
    batch, _, height, width = noisy.shape
    cells = (height // 2) * (width // 2)
    first = torch.randint(0, 4, (batch, 1, 1, cells), device=noisy.device)
    choice = torch.randint(0, 2, first.shape, device=noisy.device)
    second = NEIGHBOURS.to(noisy.device)[first, choice]
    return first, second


def subsample(image: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Half-resolution image built from the chosen position of every 2x2 cell."""
    batch, channels, height, width = image.shape
    cells = F.unfold(image, kernel_size=2, stride=2).view(batch, channels, 4, -1)
    picked = torch.gather(cells, 2, index.expand(-1, channels, 1, -1))
    return picked.view(batch, channels, height // 2, width // 2)


def ne2ne_loss(model: torch.nn.Module, noisy: torch.Tensor, gamma: float) -> torch.Tensor:
    """Neighbor2Neighbor loss: reconstruction term plus gamma times the regularisation term."""
    first, second = neighbour_indices(noisy)
    with torch.no_grad():
        full = model(noisy)
    prediction = model(subsample(noisy, first))
    target = subsample(noisy, second)
    regular = prediction - target - (subsample(full, first) - subsample(full, second))
    return F.mse_loss(prediction, target) + gamma * regular.pow(2).mean()
