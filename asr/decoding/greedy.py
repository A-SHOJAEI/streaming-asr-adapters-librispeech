from __future__ import annotations

from typing import Iterable

import torch


def ctc_collapse(ids: Iterable[int], blank_id: int) -> list[int]:
    out: list[int] = []
    prev = None
    for i in ids:
        if i == blank_id:
            prev = i
            continue
        if prev is not None and i == prev:
            continue
        out.append(int(i))
        prev = i
    return out


def ids_to_text(ids: Iterable[int], id2token: dict[int, str], *, word_delim: str = "|") -> str:
    chars: list[str] = []
    for i in ids:
        tok = id2token.get(int(i), "")
        if not tok:
            continue
        if tok.startswith("<") and tok.endswith(">"):
            continue
        if tok == word_delim:
            chars.append(" ")
        else:
            chars.append(tok)
    return "".join(chars).strip()


def greedy_decode_logits(
    logits: torch.Tensor,
    *,
    blank_id: int,
    id2token: dict[int, str],
    word_delim: str = "|",
) -> str:
    """Greedy CTC decode for a single utterance.

    Args:
      logits: (T, V) or (1, T, V)
    """

    if logits.dim() == 3:
        logits = logits.squeeze(0)
    pred = logits.argmax(dim=-1).tolist()
    collapsed = ctc_collapse(pred, blank_id=blank_id)
    return ids_to_text(collapsed, id2token=id2token, word_delim=word_delim)
