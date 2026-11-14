import torch

from src.masking import (
    checkerboard_mask,
    j_invariant_inference,
    masked_input,
    masked_recon_loss,
)
from src.unet import UNet


def test_checkerboard_phases_are_disjoint_and_cover_the_image() -> None:
    """The two checkerboard phases are disjoint and together cover every pixel."""
    first = checkerboard_mask(8, 8, 0, torch.device("cpu"))
    second = checkerboard_mask(8, 8, 1, torch.device("cpu"))
    assert torch.all(first + second == 1.0)
    assert first.sum() == second.sum() == 32


def test_masked_input_zeroes_masked_pixels() -> None:
    """Masked pixels are set to zero and visible pixels are unchanged."""
    noisy = torch.rand(1, 1, 8, 8)
    mask = checkerboard_mask(8, 8, 0, torch.device("cpu"))
    output = masked_input(noisy, mask)
    assert torch.all(output[mask == 1] == 0.0)
    assert torch.equal(output[mask == 0], noisy[mask == 0])


def test_masked_loss_ignores_visible_pixels() -> None:
    """Errors on visible pixels do not change the masked loss."""
    noisy = torch.rand(2, 1, 8, 8)
    mask = checkerboard_mask(8, 8, 1, torch.device("cpu"))
    prediction = noisy + (1.0 - mask)  # error only on visible pixels
    assert masked_recon_loss(prediction, noisy, mask).item() == 0.0


def test_output_pixel_does_not_depend_on_its_own_input() -> None:
    """J-invariant inference: an output pixel does not change when only its own input pixel changes."""
    torch.manual_seed(0)
    model = UNet().eval()
    noisy = torch.rand(1, 1, 64, 64)
    changed = noisy.clone()
    changed[0, 0, 20, 31] += 0.5
    with torch.no_grad():
        before = j_invariant_inference(model, noisy)
        after = j_invariant_inference(model, changed)
    assert torch.allclose(before[0, 0, 20, 31], after[0, 0, 20, 31], atol=1e-6)
