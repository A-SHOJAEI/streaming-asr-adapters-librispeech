from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio
from torch import nn


@dataclass(frozen=True)
class BaselineConfig:
    vocab_size: int
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    lstm_hidden: int = 512
    lstm_layers: int = 3
    dropout: float = 0.1


class LogMelBiLSTMCTC(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=0.0,
            f_max=cfg.sample_rate / 2,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

        self.lstm = nn.LSTM(
            input_size=cfg.n_mels,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=False,
        )
        self.proj = nn.Linear(2 * cfg.lstm_hidden, cfg.vocab_size)

    def forward(self, waveforms: torch.Tensor, waveform_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
          waveforms: (B, T)
          waveform_lengths: (B,)

        Returns:
          logits: (T', B, V)
          logit_lengths: (B,)
        """

        # (B, n_mels, frames)
        feats = self.mel(waveforms)
        feats = self.amplitude_to_db(feats)
        feats = feats.transpose(1, 2)  # (B, frames, n_mels)

        # Per-utterance mean/var normalization.
        mean = feats.mean(dim=1, keepdim=True)
        std = feats.std(dim=1, keepdim=True).clamp_min(1e-5)
        feats = (feats - mean) / std

        feats = feats.transpose(0, 1)  # (frames, B, n_mels)

        # Frames length is approximately floor((T - n_fft)/hop) + 1. Torchaudio exposes it via this formula.
        # We compute a conservative lower-bound to keep CTC loss well-defined.
        frame_lengths = (waveform_lengths - self.cfg.n_fft) // self.cfg.hop_length + 1
        frame_lengths = frame_lengths.clamp_min(1)

        out, _ = self.lstm(feats)
        logits = self.proj(out)
        return logits, frame_lengths
