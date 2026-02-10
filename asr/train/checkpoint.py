from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointPaths:
    dir: Path
    best: Path
    last: Path


def make_ckpt_paths(base_dir: str | Path, run_name: str, exp_name: str) -> CheckpointPaths:
    d = Path(base_dir) / run_name / exp_name
    d.mkdir(parents=True, exist_ok=True)
    return CheckpointPaths(dir=d, best=d / "best.pt", last=d / "last.pt")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("saved_at_unix", int(time.time()))
    torch.save(payload, p)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)
