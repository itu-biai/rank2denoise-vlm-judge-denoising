import torch

from src.unet import UNet


def test_output_shape_matches_input() -> None:
    """The output has the input shape (1, 1, 512, 512)."""
    model = UNet().eval()
    with torch.no_grad():
        output = model(torch.rand(1, 1, 512, 512))
    assert output.shape == (1, 1, 512, 512)


def test_parameter_count() -> None:
    """The U-Net has 7,696,193 parameters."""
    assert sum(p.numel() for p in UNet().parameters()) == 7_696_193
