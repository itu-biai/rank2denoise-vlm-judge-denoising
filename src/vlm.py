# src/vlm.py
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VlmSpec:
    """Input size and normalisation of a vision-language image encoder."""

    name: str
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


VLM_SPECS: dict[str, VlmSpec] = {
    "medsiglip": VlmSpec("medsiglip", 448, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


class FrozenVlm(nn.Module):
    """Frozen image encoder that returns L2-normalised embeddings of [0, 1] images."""

    def __init__(self, spec: VlmSpec, backbone: nn.Module) -> None:
        super().__init__()
        self.spec = spec
        self.backbone = backbone.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(spec.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec.std).view(1, 3, 1, 1))

    def train(self, mode: bool = True) -> "FrozenVlm":
        """Keep the model in evaluation mode for any requested mode."""
        return super().train(False)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Resize to the model input size, repeat to three channels and normalise."""
        resized = F.interpolate(
            images,
            size=self.spec.image_size,
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
        return (resized.expand(-1, 3, -1, -1) - self.mean) / self.std

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """L2-normalised image embedding."""
        return F.normalize(self.image_features(self.preprocess(images)), dim=-1)

    def image_features(self, pixels: torch.Tensor) -> torch.Tensor:
        """Image embedding of preprocessed pixels; set by each subclass."""
        raise NotImplementedError


class SiglipVlm(FrozenVlm):
    """Frozen SigLIP image encoder (get_image_features)."""

    def image_features(self, pixels: torch.Tensor) -> torch.Tensor:
        """Image embedding from the SigLIP backbone."""
        return self.backbone.get_image_features(pixel_values=pixels)


def load_vlm(name: str, device: torch.device) -> FrozenVlm:
    """Load the frozen MedSigLIP-448 image encoder on the device."""
    if name != "medsiglip":
        raise ValueError(f"unknown VLM {name}")
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained("google/medsiglip-448")
    return SiglipVlm(VLM_SPECS[name], backbone).to(device)
