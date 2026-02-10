from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Simple text normalization suitable for LibriSpeech-style transcripts."""
    t = text.strip().lower()
    # Keep letters, apostrophes, and spaces; map everything else to space.
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t
