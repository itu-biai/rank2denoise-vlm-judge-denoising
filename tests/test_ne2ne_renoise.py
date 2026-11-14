import torch

from src.ne2ne import neighbour_indices, subsample
from src.renoise import renoised


def test_subsample_halves_the_resolution() -> None:
    """Neighbour sub-sampling halves the height and width."""
    noisy = torch.rand(2, 1, 16, 16)
    first, second = neighbour_indices(noisy)
    assert subsample(noisy, first).shape == (2, 1, 8, 8)
    assert not torch.equal(first, second)


def test_neighbour_pairs_are_different_positions() -> None:
    """The two sub-samples take different pixels in every 2x2 cell."""
    torch.manual_seed(0)
    first, second = neighbour_indices(torch.rand(4, 1, 32, 32))
    assert torch.all(first != second)


def test_renoised_stays_in_range_and_is_reproducible() -> None:
    """Re-noised images keep the input shape, stay in [0, 1] and repeat with the same generator seed."""
    noisy = torch.rand(3, 1, 16, 16)
    fields = torch.randn(5, 1, 16, 16)
    first = renoised(noisy, fields, torch.Generator().manual_seed(1))
    second = renoised(noisy, fields, torch.Generator().manual_seed(1))
    assert first.shape == noisy.shape
    assert first.min() >= 0.0 and first.max() <= 1.0
    assert torch.equal(first, second)
