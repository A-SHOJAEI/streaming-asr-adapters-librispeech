from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from asr.decoding.greedy import greedy_decode_logits


@dataclass(frozen=True)
class StreamingResult:
    wer: float
    rtf: float
    avg_chunk_ms: float
    num_utts: int


def _concat_logits(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(chunks, dim=0)


def streaming_greedy_decode_wav2vec2(
    *,
    model: torch.nn.Module,
    processor,
    waveform: torch.Tensor,
    text: str,
    sample_rate: int,
    chunk_sec: float,
    left_context_sec: float,
    device: torch.device,
    warmup_chunks: int = 0,
) -> tuple[str, dict[str, float]]:
    """Chunked inference simulation for wav2vec2-style models.

    Implementation:
    - For each chunk, run the model on [left-context + current-chunk].
    - Slice off logits corresponding to the left-context proportion and append the remainder.

    This is an approximation, but it produces a consistent tradeoff curve for chunk/context sweeps.
    """

    model.eval()

    wav = waveform.detach().cpu()
    n = wav.numel()
    chunk_n = max(1, int(chunk_sec * sample_rate))
    left_n = max(0, int(left_context_sec * sample_rate))

    all_logits: list[torch.Tensor] = []
    chunk_times: list[float] = []

    start_time = time.perf_counter()

    idx = 0
    chunk_idx = 0
    while idx < n:
        s = idx
        e = min(n, idx + chunk_n)
        s_left = max(0, s - left_n)

        seg = wav[s_left:e]
        seg_audio_n = int(seg.numel())
        # Wav2Vec2's convolutional feature extractor requires a minimum input length. When
        # `chunk_sec` doesn't divide the utterance duration, the final remainder chunk can be too
        # short (especially with 0 left context). Pad the segment to a safe length, then drop
        # logits that correspond to padded samples (approximate proportional mapping).
        if seg_audio_n < chunk_n:
            seg = F.pad(seg, (0, int(chunk_n - seg_audio_n)))
        seg_np = seg.numpy()

        inputs = processor([seg_np], sampling_rate=sample_rate, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_values=input_values, attention_mask=attention_mask)
            logits = out.logits.squeeze(0).detach().cpu()  # (T, V)
        t1 = time.perf_counter()

        if seg_audio_n != int(seg.numel()):
            # Keep only the portion of logits corresponding to real (unpadded) audio.
            T = int(logits.shape[0])
            seg_total_n = float(seg.numel())
            T_audio = int(np.floor(T * (float(seg_audio_n) / seg_total_n))) if seg_total_n > 0 else 0
            T_audio = max(0, min(T, T_audio))
            logits = logits[:T_audio, :]

        # Estimate how many frames correspond to left context.
        seg_total = float(e - s_left)
        seg_left = float(s - s_left)
        if seg_total <= 0:
            break
        T = logits.shape[0]
        T_left = int(np.floor(T * (seg_left / seg_total)))
        T_left = max(0, min(T, T_left))

        all_logits.append(logits[T_left:, :])

        if chunk_idx >= warmup_chunks:
            chunk_times.append((t1 - t0) * 1000.0)

        idx = e
        chunk_idx += 1

    total_time = time.perf_counter() - start_time

    full_logits = _concat_logits(all_logits)
    blank_id = int(getattr(model, "config", None).pad_token_id if getattr(model, "config", None) else 0)

    # id2token
    tok = processor.tokenizer
    id2token = {i: t for i, t in enumerate(tok.convert_ids_to_tokens(range(len(tok))))}

    hyp = greedy_decode_logits(full_logits, blank_id=blank_id, id2token=id2token, word_delim="|")

    metrics = {
        "rtf": float(total_time) / float(n / sample_rate),
        "avg_chunk_ms": float(np.mean(chunk_times)) if chunk_times else float("nan"),
    }
    return hyp, metrics
