# src/renoise.py
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from src.data import load_image
from src.masking import j_invariant_inference
from src.vlm import FrozenVlm


@torch.no_grad()
def noise_fields(model: nn.Module, files: list[Path], key: str, device: torch.device) -> torch.Tensor:
    """Noise fields of the given slices: noisy input minus the J-invariant output of the model."""
    model.eval()
    fields = []
    for path in tqdm(files, desc="noise fields", leave=False):
        noisy = load_image(path, key).unsqueeze(0).to(device)
        fields.append((noisy - j_invariant_inference(model, noisy)).cpu())
    return torch.cat(fields)


def renoised(noisy: torch.Tensor, fields: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Add the field of a random donor slice, mirrored left-right with probability 1/2; clip to [0, 1]."""
    donors = torch.randint(0, len(fields), (len(noisy),), generator=generator)
    donor_fields = fields[donors].to(noisy.device)
    flip = torch.rand(len(noisy), generator=generator) < 0.5
    donor_fields[flip] = donor_fields[flip].flip(-1)
    return (noisy + donor_fields).clamp(0.0, 1.0)


@torch.no_grad()
def rank_direction(
    vlm: FrozenVlm,
    files: list[Path],
    key: str,
    fields: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[torch.Tensor, float]:
    """Unit judge direction w (mean of the embedding differences) and the mean projection on it."""
    generator = torch.Generator().manual_seed(seed)
    differences = []
    for start in tqdm(range(0, len(files), batch_size), desc="rank direction", leave=False):
        noisy = torch.stack([load_image(p, key) for p in files[start : start + batch_size]])
        differences.append(
            (vlm.encode_image(noisy.to(device)) - vlm.encode_image(renoised(noisy, fields, generator).to(device))).cpu()
        )
    differences = torch.cat(differences)
    direction = differences.sum(0) / differences.sum(0).norm()
    return direction, float((differences @ direction).mean())
