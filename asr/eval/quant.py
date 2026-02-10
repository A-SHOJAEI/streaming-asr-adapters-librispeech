from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class QuantResult:
    scheme: str
    wer: float
    rtf: float
    model_size_mb: float
    num_utts: int


def model_state_size_mb(model: torch.nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return float(buf.tell()) / (1024.0 * 1024.0)


def dynamic_int8_quantize_cpu(model: torch.nn.Module) -> torch.nn.Module:
    # Weight-only dynamic quantization is CPU-only.
    return torch.ao.quantization.quantize_dynamic(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)


def timed_forward(model: torch.nn.Module, fn, *, audio_sec: float) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return float(t1 - t0) / max(1e-9, float(audio_sec))
