# scripts/infer_n2s_n2n.py
"""N2S and N2N outputs of the validation and test slices (cache), and the stage-1 search tools. No GT is used here.

Inference (TF32 off): N2S (J-invariant, two passes) and N2N (direct), clipped to [0, 1], for one N2S seed; written per
chunk to `<R2D_RESULTS>/infer_n2s_n2n/seed<seed>/cache/<split>_<start>.npz` (resume). The validation cache holds no GT;
the test cache also holds the GT, read only for evaluation by later scripts.

Stage-1 tools used by `scripts/stage1_search.py`: `RankScorer` (s(x) = <phi(x), w>, VLM batch halves on CUDA OOM) and
`fit_recipe_vlm` (greedy search over the five stage-1 moves of `src.moves.directions`; start from the N2S output; at
most 30 steps; at each step the move with the largest mean score is kept; stop when the mean score does not rise;
ties go to the first move in MOVES).

Usage (cwd = repository root):  python -m scripts.infer_n2s_n2n --seed 42   (then 123 and 456)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src import paper_run
from src.data import _load_npz_arrays, load_splits
from src.masking import j_invariant_inference
from src.moves import MAX_STEPS, directions
from src.paths import DATA_ROOT, N2N_CHECKPOINT, RESULTS_ROOT, n2s_checkpoint
from src.unet import UNet
from src.vlm import FrozenVlm

MOVES = (
    "to_noisy",
    "to_ne2ne",
    "to_n2s",
    "gauss",
    "high_band",
)  # fixed move order (ties go to the first)
CKPT_SHA256 = {
    42: "95b3ac4f882fcdfe5e05235e5700a92d527a21817ba01608ef3b5d1b7502d44e",
    123: "c557202e8a4071378b5cf0c1f6e604521af0079f6a31045c05f3b90149e40e75",
    456: "2cc04b53375ff37dc65d6f2862132a5e72cf79c59307f81f05d33f1a570cf455",
    "ne2ne": "90876b59425bd8c5b67f124d275019f53d52175f8e1705f0b01718b6eaa1362a",
}  # sha256 of the released checkpoints


@dataclass(frozen=True)
class RankRecipeConfig:
    """Settings of the N2S / N2N inference and the stage-1 search for one seed."""

    seed: int = 42
    data_root: Path = DATA_ROOT
    input_key: str = "noisy_clipped_100k"
    clean_key: str = "gt_clipped"
    held_out_fraction: float = 0.3  # only for the load_splits signature
    split_seed: int = 42
    vlm_name: str = "medsiglip"
    chunk_files: int = 16  # inference chunk (resume unit)
    read_threads: int = 8  # files are read in parallel
    move_threads: int = 8  # moves (Gauss, FFT) run in parallel over slices
    unet_batch: int = 8
    start_batch: int = 512  # VLM batch starts here and halves on OOM

    @property
    def out_dir(self) -> Path:
        """Output folder of this seed."""
        return RESULTS_ROOT / "infer_n2s_n2n" / f"seed{self.seed}"

    @property
    def n2s_checkpoint(self) -> Path:
        """N2S checkpoint of this seed."""
        return n2s_checkpoint(self.seed)


# ----------------------------------------------------------------------------- helpers


def _atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)


def _atomic_json(payload: dict, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(temporary, target)


def _set_tf32(enabled: bool) -> None:
    """U-Net inference with TF32 off; VLM scoring with TF32 on."""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def load_unet(path: Path, expected_sha: str, device: torch.device) -> torch.nn.Module:
    """Load a U-Net checkpoint; its sha256 is compared with the released checkpoint (warning only, `src/paper_run.py`)."""
    paper_run.checkpoint(path, expected_sha)
    model = UNet()
    model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    return model.to(device).eval()


# ----------------------------------------------------------------------------- inference (resume)


@torch.no_grad()
def build_cache(split: str, files: list[Path], config: RankRecipeConfig, device: torch.device) -> None:
    """N2S + N2N outputs per chunk in `cache/<split>_<start>.npz`. Validation GT is not read."""
    keys = (config.input_key,) if split == "val" else (config.input_key, config.clean_key)
    folder = config.out_dir / "cache"
    folder.mkdir(parents=True, exist_ok=True)
    starts = [
        start
        for start in range(0, len(files), config.chunk_files)
        if not (folder / f"{split}_{start:04d}.npz").exists()
    ]
    if not starts:
        return
    _set_tf32(False)
    n2s_model = load_unet(config.n2s_checkpoint, CKPT_SHA256[config.seed], device)
    n2n_model = load_unet(N2N_CHECKPOINT, CKPT_SHA256["ne2ne"], device)
    for start in starts:
        began = time.time()
        chunk = files[start : start + config.chunk_files]
        with ThreadPoolExecutor(config.read_threads) as pool:
            samples = list(pool.map(lambda path: _load_npz_arrays(path, keys), chunk))
        noisy = torch.from_numpy(
            np.stack([sample[config.input_key].astype(np.float32) for sample in samples])
        ).unsqueeze(1)
        n2s, n2n = [], []
        for index in range(0, len(chunk), config.unet_batch):
            batch = noisy[index : index + config.unet_batch].to(device)
            n2s.append(j_invariant_inference(n2s_model, batch).clamp(0.0, 1.0).float().cpu().numpy()[:, 0])
            n2n.append(n2n_model(batch).clamp(0.0, 1.0).float().cpu().numpy()[:, 0])
        arrays = {
            "slice_ids": np.array([path.stem for path in chunk]),
            "noisy": noisy.numpy()[:, 0],
            "n2s": np.concatenate(n2s),
            "ne2ne": np.concatenate(n2n),
        }
        if split != "val":
            arrays["clean"] = np.stack([sample[config.clean_key].astype(np.float32) for sample in samples])
        target = folder / f"{split}_{start:04d}.npz"
        temporary = target.with_name(target.stem + ".tmp.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, target)
        print(
            f"inference {split}: {min(start + config.chunk_files, len(files))}/{len(files)} · {time.time() - began:.0f} s",
            flush=True,
        )


def load_cache(split: str, files: list[Path], config: RankRecipeConfig) -> dict[str, dict[str, np.ndarray]]:
    """Slice -> {noisy, n2s, ne2ne[, clean]} (float64). Slice order follows the split order."""
    slices: dict[str, dict[str, np.ndarray]] = {}
    for start in range(0, len(files), config.chunk_files):
        with np.load(config.out_dir / "cache" / f"{split}_{start:04d}.npz") as archive:
            expected = [path.stem for path in files[start : start + config.chunk_files]]
            if archive["slice_ids"].tolist() != expected:
                raise RuntimeError(
                    f"cache/{split}_{start:04d}.npz comes from another run (slice order differs). Delete the folder."
                )
            for index, slice_id in enumerate(expected):
                slices[slice_id] = {
                    key: archive[key][index].astype(np.float64)
                    for key in ("noisy", "n2s", "ne2ne", "clean")
                    if key in archive.files
                }
    if split == "val" and any("clean" in data for data in slices.values()):
        raise RuntimeError("validation cache contains GT; it must not.")
    return slices


# ----------------------------------------------------------------------------- applying a recipe


def apply_moves(data: dict[str, np.ndarray], recipe: list[str]) -> np.ndarray:
    """Apply a recipe to one slice and return the image."""
    moves = directions(data["noisy"], data["ne2ne"], data["n2s"])
    current = data["n2s"]
    for move in recipe:
        current = moves[move](current)
    return current


# ----------------------------------------------------------------------------- VLM score


class RankScorer:
    """s(x) = <phi(x), w>. Starts with a large batch; on CUDA OOM the batch halves and the chunk is retried."""

    def __init__(
        self,
        vlm: FrozenVlm,
        direction: torch.Tensor,
        config: RankRecipeConfig,
        device: torch.device,
    ) -> None:
        self.vlm, self.device, self.batch = vlm, device, config.start_batch
        self.direction = direction.to(device=device, dtype=torch.float32)
        self.embedded = 0
        self.seconds = 0.0

    @torch.no_grad()
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """images: (N, 512, 512) -> (N,) scores."""
        began = time.time()
        scores, index = [], 0
        while index < len(images):
            try:
                batch = torch.from_numpy(np.ascontiguousarray(images[index : index + self.batch], dtype=np.float32))
                embedding = self.vlm.encode_image(batch.unsqueeze(1).to(self.device, non_blocking=True)).float()
                scores.append((embedding @ self.direction).cpu().numpy())
                index += len(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch == 1:
                    raise
                self.batch //= 2
                print(f"  CUDA OOM → VLM batch {self.batch}", flush=True)
        self.embedded += len(images)
        self.seconds += time.time() - began
        return np.concatenate(scores)


# ----------------------------------------------------------------------------- stage-1 recipe selection (VLM, resume)


def _parallel(function: Callable, items: list, threads: int) -> list:
    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(function, items))


def fit_recipe_vlm(
    name: str,
    ids: list[str],
    val: dict[str, dict[str, np.ndarray]],
    scorer: RankScorer,
    config: RankRecipeConfig,
) -> list[str]:
    """Greedy recipe selection with the mean VLM score as the criterion.

    Every step is written to `steps/<name>.csv` (mean of the five moves, accepted move, best minus second).
    An interrupted run replays the accepted steps and continues. The final recipe goes to `recipes/<name>.json`.

    """
    recipe_path = config.out_dir / "recipes" / f"{name}.json"
    if recipe_path.exists():
        return json.loads(recipe_path.read_text(encoding="utf-8"))["recipe"]
    steps_path = config.out_dir / "steps" / f"{name}.csv"
    steps_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    moves = {
        slice_id: directions(val[slice_id]["noisy"], val[slice_id]["ne2ne"], val[slice_id]["n2s"]) for slice_id in ids
    }
    current = np.stack([val[slice_id]["n2s"] for slice_id in ids])
    rows = pd.read_csv(steps_path, float_precision="round_trip").to_dict("records") if steps_path.exists() else []
    recipe = [row["best_move"] for row in rows if row["accepted"]]
    if rows:  # resume: replay the accepted steps (numpy, deterministic)
        for move in recipe:
            current = np.stack(
                _parallel(
                    lambda pair, move=move: moves[pair[0]][move](pair[1]),
                    list(zip(ids, current)),
                    config.move_threads,
                )
            )
        score = float(rows[-1]["score_after"])
        print(
            f"  {name}: resume, {len(recipe)} steps from disk · score {score:.6f}",
            flush=True,
        )
        if not rows[-1]["accepted"]:  # the stop step is already written
            _atomic_json(
                {
                    "name": name,
                    "n_slices": len(ids),
                    "recipe": recipe,
                    "final_mean_score": score,
                },
                recipe_path,
            )
            return recipe
    else:
        score = float(scorer(current).mean())
        print(
            f"  {name}: start (N2S) mean score {score:.6f} · {len(ids)} slices",
            flush=True,
        )
    while len(recipe) < MAX_STEPS:
        means, best_images, best_mean = {}, None, -np.inf
        for move in MOVES:  # move by move: at most two candidate stacks in memory
            candidates = np.stack(
                _parallel(
                    lambda pair, move=move: moves[pair[0]][move](pair[1]),
                    list(zip(ids, current)),
                    config.move_threads,
                )
            )
            means[move] = float(scorer(candidates).mean())
            if means[move] > best_mean:  # ties keep the first move
                best_mean, best_images = means[move], candidates
        best = max(means, key=means.get)
        second = sorted(means.values())[-2]
        accepted = means[best] > score
        row = {
            "step": len(recipe) + 1,
            **{f"mean_{move}": means[move] for move in MOVES},
            "score_before": score,
            "best_move": best,
            "accepted": accepted,
            "score_after": means[best] if accepted else score,
            "gap_best_second": means[best] - second,
            "gain_over_current": means[best] - score,
        }
        rows.append(row)
        _atomic_csv(pd.DataFrame(rows), steps_path)
        if not accepted:
            break
        current, score = best_images, means[best]
        recipe.append(best)
        print(
            f"  {name} step {len(recipe)}: {best} → {score:.6f} (best - second {row['gap_best_second']:.2e})",
            flush=True,
        )
    _atomic_json(
        {
            "name": name,
            "n_slices": len(ids),
            "recipe": recipe,
            "final_mean_score": score,
        },
        recipe_path,
    )
    return recipe


# ----------------------------------------------------------------------------- main


def main(config: RankRecipeConfig) -> None:
    """Write the validation and test cache of one N2S seed (existing chunks are skipped)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(config.data_root, config.held_out_fraction, config.split_seed)
    for required in (config.n2s_checkpoint, N2N_CHECKPOINT):
        if not Path(required).is_file():
            raise FileNotFoundError(f"Missing input; nothing was run: {required}")
    for split in ("val", "test"):
        build_cache(split, splits[split], config, device)
    print(f"cache written: {config.out_dir / 'cache'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, choices=(42, 123, 456))
    main(RankRecipeConfig(seed=parser.parse_args().seed))
