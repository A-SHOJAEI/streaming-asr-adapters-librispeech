"""LibriSpeech data utilities and synthetic dataset generator."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def generate_synthetic_ctc_dataset(
    *,
    root: Path,
    sample_rate: int = 16000,
    num_train: int = 64,
    num_dev: int = 16,
    num_test: int = 16,
    min_sec: float = 0.6,
    max_sec: float = 1.2,
    vocab: list[str] | None = None,
    seed: int = 1337,
) -> None:
    """Generate a tiny synthetic audio dataset for smoke testing.

    Creates JSONL manifests and .wav files under root/.
    """
    if vocab is None:
        vocab = list("abcde ")

    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    root.mkdir(parents=True, exist_ok=True)

    def _gen_split(split: str, n: int) -> None:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = root / f"{split}.jsonl"

        with open(manifest_path, "w") as mf:
            for i in range(n):
                # Generate random text
                text_len = rng.randint(3, 8)
                text = "".join(rng.choices(vocab, k=text_len))

                # Generate random audio (white noise)
                duration = rng.uniform(min_sec, max_sec)
                num_samples = int(duration * sample_rate)
                audio = np_rng.randn(num_samples).astype(np.float32) * 0.01

                # Save wav
                wav_path = split_dir / f"{i:04d}.wav"
                sf.write(str(wav_path), audio, sample_rate)

                # Write manifest line
                entry = {
                    "audio_path": str(wav_path),
                    "text": text,
                    "duration": float(duration),
                    "sample_rate": sample_rate,
                }
                mf.write(json.dumps(entry) + "\n")

    _gen_split("train", num_train)
    _gen_split("dev", num_dev)
    _gen_split("test", num_test)


def download_librispeech(
    *,
    raw_dir: Path,
    splits: list[str],
    download_lm: bool = False,
    verify: bool = True,
) -> None:
    """Download LibriSpeech splits. Placeholder for full implementation."""
    raise NotImplementedError(
        "Full LibriSpeech download not implemented. "
        "Use dataset.kind=synthetic for testing."
    )


def prepare_librispeech_manifests(
    *,
    raw_dir: Path,
    manifests_dir: Path,
    splits: list[str],
    sample_rate: int = 16000,
    compute_duration: bool = True,
) -> None:
    """Prepare JSONL manifests from downloaded LibriSpeech. Placeholder."""
    raise NotImplementedError(
        "LibriSpeech manifest preparation not implemented. "
        "Use dataset.kind=synthetic for testing."
    )
