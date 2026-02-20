"""Dataset and data loading utilities for ASR manifests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


@dataclass
class Example:
    """A single utterance."""
    audio_path: str
    text: str
    duration: float = 0.0
    sample_rate: int = 16000

    @property
    def wav_path(self) -> str:
        """Alias for audio_path (used by wav2vec2 collate)."""
        return self.audio_path


def load_audio(path: str | Path, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and return a 1-D float tensor at target_sr."""
    wav, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(wav)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav


class ManifestDataset(Dataset):
    """Reads a JSONL manifest: each line has {audio_path, text, duration?, sample_rate?}."""

    def __init__(self, manifest_path: str | Path, max_items: int | None = None):
        self.examples: list[Example] = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.examples.append(Example(
                    audio_path=obj["audio_path"],
                    text=obj.get("text", ""),
                    duration=float(obj.get("duration", 0.0)),
                    sample_rate=int(obj.get("sample_rate", 16000)),
                ))
                if max_items is not None and len(self.examples) >= max_items:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def collate_waveforms(batch: list[Example], sample_rate: int = 16000) -> dict[str, Any]:
    """Load audio, pad waveforms to same length and stack."""
    waveforms = [load_audio(ex.audio_path, target_sr=sample_rate) for ex in batch]
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded[i, :w.shape[0]] = w
    return {
        "waveform": padded,
        "waveform_lengths": lengths,
        "text": [ex.text for ex in batch],
        "audio_path": [ex.audio_path for ex in batch],
    }


def collate_ctc_labels(texts: list[str], vocab: "CharVocab") -> tuple[torch.Tensor, torch.Tensor]:
    """Encode texts to integer label sequences for CTC loss."""
    encoded = [vocab.encode(t) for t in texts]
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(lengths) > 0 else 0
    labels = torch.zeros(len(encoded), max_len, dtype=torch.long)
    for i, e in enumerate(encoded):
        labels[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return labels, lengths
