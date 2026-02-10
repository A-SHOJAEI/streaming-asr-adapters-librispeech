from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from asr.decoding.greedy import greedy_decode_logits


@dataclass(frozen=True)
class DecodeArtifacts:
    refs: list[str]
    hyps: list[str]


def decode_batch_greedy_ctc(
    *,
    logits: torch.Tensor,
    logit_lengths: torch.Tensor,
    texts: list[str],
    blank_id: int,
    id2token: dict[int, str],
    word_delim: str = "|",
) -> DecodeArtifacts:
    # logits: (T, B, V)
    logits = logits.detach().cpu()
    out_hyps: list[str] = []
    for b in range(logit_lengths.numel()):
        T = int(logit_lengths[b].item())
        out_hyps.append(
            greedy_decode_logits(
                logits[:T, b, :], blank_id=blank_id, id2token=id2token, word_delim=word_delim
            )
        )
    return DecodeArtifacts(refs=texts, hyps=out_hyps)


def decode_logits_kenlm(
    *,
    logits: torch.Tensor,
    logit_lengths: torch.Tensor,
    texts: list[str],
    decoder: Any,
) -> DecodeArtifacts:
    logits = logits.detach().cpu().float().numpy()
    out_hyps: list[str] = []
    for b in range(logit_lengths.shape[0]):
        T = int(logit_lengths[b])
        out_hyps.append(decoder.decode(np.asarray(logits[:T, b, :], dtype=np.float32)))
    return DecodeArtifacts(refs=texts, hyps=out_hyps)
