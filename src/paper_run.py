# src/paper_run.py
"""Checks against the values of the paper run.

A few checks compare a result with the value that the paper run gave: the sha256 of the released N2S / N2N
checkpoints, the stage-1 recipes of the Val.-PSNR and Q judges, and the mean test PSNR of the Q stage-1 recipe.
With the released checkpoints these values match. With retrained models or another GPU small differences are
expected, so a difference never stops the run: a note is printed and the row is written with
`matches_paper_run = False`. Matching values print nothing. Every row goes to `<R2D_RESULTS>/paper_run_checks.csv` (one row per check and seed; a rerun replaces it).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.paths import RESULTS_ROOT

LOG = RESULTS_ROOT / "paper_run_checks.csv"
EXPECTED = "Small differences are expected with retrained models or a different GPU. The run continues."
EXPECTED_CHECKPOINT = "This is expected if you trained the models yourself. The run continues."


def _where(seed: int | None) -> str:
    return f" (seed {seed})" if seed is not None else ""


def _seed_key(seed: float | None) -> str:
    """Same text for 42, 42.0 (read back from CSV) and for a missing seed."""
    return "" if seed is None or pd.isna(seed) else str(int(seed))


def _log(
    check: str,
    seed: int | None,
    value: float | str,
    paper_run: float | str,
    tolerance: float | None,
    matches: bool,
    note: str,
) -> dict:
    row = {
        "check": check,
        "seed": seed,
        "value": value,
        "paper_run": paper_run,
        "tolerance": tolerance,
        "matches_paper_run": bool(matches),
    }
    LOG.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([row])
    if LOG.exists():  # a rerun replaces the earlier row of the same check and seed
        earlier = pd.read_csv(LOG)
        same_seed = earlier["seed"].map(_seed_key) == _seed_key(seed)
        keep = ~((earlier["check"] == check) & same_seed)
        table = pd.concat([earlier[keep], table], ignore_index=True)
    table.to_csv(LOG, index=False)
    if not matches:
        print(f"Note: {note}", flush=True)
    return row


def number(
    check: str,
    value: float,
    paper_run: float,
    tolerance: float,
    unit: str = "",
    seed: int | None = None,
) -> dict:
    """A number vs the paper run, within `tolerance`."""
    difference = float(value) - float(paper_run)
    note = (
        f"{check}{_where(seed)} is {value:.3f}{unit}; the paper run gave {paper_run:.3f}{unit} "
        f"(difference {abs(difference):.3f}{unit}, tolerance {tolerance}{unit}).\n{EXPECTED}"
    )
    return _log(check, seed, value, paper_run, tolerance, abs(difference) <= tolerance, note)


def recipe(check: str, value: list[str], paper_run: list[str], seed: int | None = None) -> dict:
    """A recipe (list of moves) vs the paper run; must be equal."""
    found, paper = " ".join(value), " ".join(paper_run)
    note = f'{check}{_where(seed)} is "{found}"; the paper run found "{paper}".\n{EXPECTED}'
    return _log(check, seed, found, paper, None, found == paper, note)


def checkpoint(path: Path, paper_run_sha256: str, seed: int | None = None) -> dict:
    """sha256 of a checkpoint file vs the released checkpoint of the paper run."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = f"{path.parent.name}/{path.name}"
    note = (
        f"{name} is not the released checkpoint of the paper (sha256 {digest[:6]}… vs {paper_run_sha256[:6]}…).\n"
        f"{EXPECTED_CHECKPOINT}"
    )
    return _log(
        f"checkpoint_sha256 {name}",
        seed,
        digest,
        paper_run_sha256,
        None,
        digest == paper_run_sha256,
        note,
    )
