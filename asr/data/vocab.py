"""Character-level vocabulary for CTC ASR."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CharVocab:
    """Simple character vocabulary.

    Index 0 is always the CTC blank token.
    """

    def __init__(self, tokens: list[str] | None = None):
        if tokens is None:
            tokens = ["<blank>"]
        self.tokens = list(tokens)
        self._tok2id = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def blank_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        """Encode text as list of token ids, skipping unknown characters."""
        ids = []
        for ch in text:
            if ch in self._tok2id:
                ids.append(self._tok2id[ch])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text."""
        chars = []
        for i in ids:
            if 0 <= i < len(self.tokens) and i != self.blank_id:
                chars.append(self.tokens[i])
        return "".join(chars)

    def id_to_token(self) -> dict[int, str]:
        """Return mapping from id to token string."""
        return {i: t for i, t in enumerate(self.tokens)}


def build_char_vocab_from_manifests(manifest_paths: list[str | Path]) -> CharVocab:
    """Build a character vocabulary from JSONL manifest files."""
    chars: set[str] = set()
    for path in manifest_paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text", "")
                chars.update(text)
    sorted_chars = sorted(chars)
    tokens = ["<blank>"] + sorted_chars
    return CharVocab(tokens=tokens)
