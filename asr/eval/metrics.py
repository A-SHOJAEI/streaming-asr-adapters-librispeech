from __future__ import annotations

from dataclasses import dataclass

from jiwer import wer


@dataclass(frozen=True)
class WERResult:
    wer: float


def compute_wer(refs: list[str], hyps: list[str]) -> WERResult:
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must have same length")
    return WERResult(wer=float(wer(refs, hyps)))
