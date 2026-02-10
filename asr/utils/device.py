from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    device = (device or "auto").lower()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)
    raise ValueError(f"Unknown device: {device}")
