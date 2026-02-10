from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KenLMDecoder:
    decoder: object

    def decode(self, logits: np.ndarray) -> str:
        # logits: (T, V) as float32
        return self.decoder.decode(logits)


def build_kenlm_ctc_decoder(
    *,
    labels: list[str],
    arpa_path: str | Path,
) -> KenLMDecoder:
    """Build a CTC beam-search decoder backed by KenLM.

    Requires optional deps: pyctcdecode + kenlm.
    """

    from pyctcdecode import build_ctcdecoder

    dec = build_ctcdecoder(labels, str(arpa_path))
    return KenLMDecoder(decoder=dec)
