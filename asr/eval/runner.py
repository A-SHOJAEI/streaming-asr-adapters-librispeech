from __future__ import annotations

import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from asr.data.dataset import ManifestDataset, collate_waveforms, load_audio
from asr.decoding.greedy import greedy_decode_logits
from asr.decoding.lm_decoder import build_kenlm_ctc_decoder
from asr.eval.metrics import compute_wer
from asr.eval.quant import dynamic_int8_quantize_cpu, model_state_size_mb
from asr.eval.streaming import streaming_greedy_decode_wav2vec2
from asr.models.baseline_bilstm_ctc import BaselineConfig, LogMelBiLSTMCTC
from asr.train.checkpoint import load_checkpoint, make_ckpt_paths
from asr.train.train_baseline import load_trained_baseline
from asr.train.train_wav2vec2 import load_trained_wav2vec2
from asr.utils.device import resolve_device
from asr.utils.io import ensure_dir


def _env_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "timestamp_unix": int(time.time()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        meta["cuda"] = torch.version.cuda
        meta["cudnn"] = torch.backends.cudnn.version()
        meta["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return meta


def _split_manifests(cfg: dict[str, Any]) -> dict[str, dict[str, Path]]:
    dcfg = cfg["dataset"]
    kind = dcfg["kind"]

    if kind == "synthetic":
        root = Path(dcfg["root"])
        return {
            "train": {"train": root / "train.jsonl"},
            "dev": {"dev": root / "dev.jsonl"},
            "test": {"test": root / "test.jsonl"},
        }

    if kind == "librispeech":
        mdir = Path(dcfg["manifests_dir"])
        dev = {"dev-clean": mdir / "dev-clean.jsonl", "dev-other": mdir / "dev-other.jsonl"}
        test = {"test-clean": mdir / "test-clean.jsonl", "test-other": mdir / "test-other.jsonl"}
        train = {"train-clean-100": mdir / "train-clean-100.jsonl"}
        return {"train": train, "dev": dev, "test": test}

    raise ValueError(f"Unknown dataset.kind: {kind}")


@torch.no_grad()
def _eval_baseline_wer(
    *,
    ckpt_path: Path,
    manifests: dict[str, Path],
    sample_rate: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_utts: int | None,
) -> dict[str, Any]:
    model, vocab, _ = load_trained_baseline(ckpt_path, device=device)

    out: dict[str, Any] = {}
    for name, mp in manifests.items():
        ds = ManifestDataset(mp, max_items=max_utts)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_waveforms(batch, sample_rate=sample_rate),
        )

        refs: list[str] = []
        hyps: list[str] = []

        id2token = vocab.id_to_token()
        blank_id = vocab.blank_id

        for batch in loader:
            wav = batch["waveform"].to(device)
            wav_l = batch["waveform_lengths"].to(device)
            logits, logit_l = model(wav, wav_l)
            logits = logits.detach().cpu()
            for b in range(logit_l.numel()):
                T = int(logit_l[b])
                hyps.append(greedy_decode_logits(logits[:T, b, :], blank_id=blank_id, id2token=id2token, word_delim=" "))
            refs.extend(batch["text"])

        out[name] = {"wer": compute_wer(refs, hyps).wer, "num_utts": len(refs)}

    return out


@torch.no_grad()
def _eval_w2v2_wer(
    *,
    ckpt_path: Path,
    manifests: dict[str, Path],
    sample_rate: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_utts: int | None,
    decoding: list[str],
    lm_arpa_gz: Path | None,
) -> dict[str, Any]:
    model, processor, _ = load_trained_wav2vec2(ckpt_path, device=device)

    blank_id = int(model.config.pad_token_id)
    tok = processor.tokenizer
    id2token = {i: t for i, t in enumerate(tok.convert_ids_to_tokens(range(len(tok))))}

    kenlm_dec = None
    if "kenlm" in decoding:
        if lm_arpa_gz is None or not lm_arpa_gz.exists():
            raise FileNotFoundError(
                "LM decoding requested but LM arpa.gz not found. Download LM resources or disable kenlm decoding."
            )
        # pyctcdecode can read gz in some versions, but we decompress to be safe.
        cache_dir = ensure_dir("data/cache/lm")
        arpa_path = cache_dir / lm_arpa_gz.with_suffix("").name
        if not arpa_path.exists():
            import gzip

            with gzip.open(lm_arpa_gz, "rb") as fin, arpa_path.open("wb") as fout:
                fout.write(fin.read())
        labels = tok.convert_ids_to_tokens(range(len(tok)))
        kenlm_dec = build_kenlm_ctc_decoder(labels=labels, arpa_path=arpa_path)

    out: dict[str, Any] = {}
    for name, mp in manifests.items():
        ds = ManifestDataset(mp, max_items=max_utts)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda batch: _collate_w2v2_for_eval(batch, processor, sample_rate),
        )

        refs: list[str] = []
        hyps_greedy: list[str] = []
        hyps_kenlm: list[str] = []

        for batch in loader:
            input_values = batch["input_values"].to(device)
            attn = batch.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)

            logits = model(input_values=input_values, attention_mask=attn).logits  # (B, T, V)
            logits_t = logits.transpose(0, 1).detach().cpu()  # (T, B, V)

            if attn is None:
                logit_l = torch.full((logits_t.shape[1],), logits_t.shape[0], dtype=torch.long)
            else:
                in_l = attn.sum(dim=1).detach().cpu()
                ratio = logits_t.shape[0] / float(input_values.shape[1])
                logit_l = (in_l.float() * ratio).floor().clamp_min(1).to(torch.long)

            for b in range(logit_l.numel()):
                T = int(logit_l[b])
                hyps_greedy.append(
                    greedy_decode_logits(logits_t[:T, b, :], blank_id=blank_id, id2token=id2token, word_delim="|")
                )

            if kenlm_dec is not None:
                import numpy as _np

                for b in range(logit_l.numel()):
                    T = int(logit_l[b])
                    hyps_kenlm.append(kenlm_dec.decode(_np.asarray(logits_t[:T, b, :], dtype=_np.float32)))

            refs.extend(batch["text"])

        rec: dict[str, Any] = {
            "greedy": {"wer": compute_wer(refs, hyps_greedy).wer, "num_utts": len(refs)},
        }
        if kenlm_dec is not None:
            rec["kenlm"] = {"wer": compute_wer(refs, hyps_kenlm).wer, "num_utts": len(refs)}
        out[name] = rec

    return out


def _collate_w2v2_for_eval(batch, processor, sample_rate: int) -> dict[str, Any]:
    wavs = [load_audio(ex.wav_path, sample_rate).numpy() for ex in batch]
    texts = [ex.text for ex in batch]
    inputs = processor(wavs, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs["text"] = texts
    return inputs


def _streaming_eval_w2v2(
    *,
    ckpt_path: Path,
    manifests: dict[str, Path],
    sample_rate: int,
    device: torch.device,
    chunk_sec: list[float],
    left_context_sec: list[float],
    warmup_chunks: int,
    max_utts: int | None,
) -> dict[str, Any]:
    model, processor, _ = load_trained_wav2vec2(ckpt_path, device=device)

    out: dict[str, Any] = {}
    for name, mp in manifests.items():
        ds = ManifestDataset(mp, max_items=max_utts)
        by_setting: dict[str, Any] = {}

        for cs in chunk_sec:
            for lc in left_context_sec:
                refs: list[str] = []
                hyps: list[str] = []
                rtfs: list[float] = []
                lats: list[float] = []

                for ex in ds:
                    wav = load_audio(ex.wav_path, sample_rate)
                    hyp, metrics = streaming_greedy_decode_wav2vec2(
                        model=model,
                        processor=processor,
                        waveform=wav,
                        text=ex.text,
                        sample_rate=sample_rate,
                        chunk_sec=float(cs),
                        left_context_sec=float(lc),
                        device=device,
                        warmup_chunks=warmup_chunks,
                    )
                    hyps.append(hyp)
                    rtfs.append(float(metrics["rtf"]))
                    lats.append(float(metrics["avg_chunk_ms"]))
                    refs.append(ex.text)

                setting_key = f"chunk_{cs:.3f}_left_{lc:.3f}"
                by_setting[setting_key] = {
                    "chunk_sec": float(cs),
                    "left_context_sec": float(lc),
                    "wer": compute_wer(refs, hyps).wer,
                    "rtf_mean": float(np.mean(rtfs)) if rtfs else float("nan"),
                    "avg_chunk_ms_mean": float(np.nanmean(lats)) if lats else float("nan"),
                    "num_utts": len(hyps),
                }

        out[name] = by_setting

    return out


def _quant_eval(
    *,
    model_kind: str,
    ckpt_path: Path,
    manifests: dict[str, Path],
    sample_rate: int,
    device: torch.device,
    schemes: list[str],
    max_utts: int | None,
) -> dict[str, Any]:
    # Quant evaluation is intentionally simple: greedy WER + RTF + model size.
    out: dict[str, Any] = {}

    if model_kind == "baseline":
        model_fp, vocab, _ = load_trained_baseline(ckpt_path, device=resolve_device("cpu"))
        blank_id = vocab.blank_id
        id2token = vocab.id_to_token()

        for split_name, mp in manifests.items():
            ds = ManifestDataset(mp, max_items=max_utts)
            split_rec: dict[str, Any] = {}

            for scheme in schemes:
                if scheme == "fp32":
                    dev = device
                    model = model_fp.to(dev)
                elif scheme == "int8_dynamic_cpu":
                    model = dynamic_int8_quantize_cpu(model_fp)
                    dev = resolve_device("cpu")
                else:
                    continue

                refs: list[str] = []
                hyps: list[str] = []

                t0 = time.perf_counter()
                audio_sec = 0.0
                for ex in ds:
                    wav = load_audio(ex.wav_path, sample_rate)
                    audio_sec += float(wav.numel()) / float(sample_rate)
                    wav_b = wav.unsqueeze(0).to(dev)
                    wav_l = torch.tensor([wav.numel()], dtype=torch.long).to(dev)
                    logits, logit_l = model(wav_b, wav_l)
                    hyp = greedy_decode_logits(
                        logits[: int(logit_l[0]), 0, :].detach().cpu(), blank_id=blank_id, id2token=id2token, word_delim=" "
                    )
                    hyps.append(hyp)
                    refs.append(ex.text)

                dt = time.perf_counter() - t0
                split_rec[scheme] = {
                    "wer": compute_wer(refs, hyps).wer,
                    "rtf": float(dt) / max(1e-9, audio_sec),
                    "model_size_mb": model_state_size_mb(model),
                    "num_utts": len(refs),
                }

            out[split_name] = split_rec

        return out

    if model_kind in ("wav2vec2_full", "wav2vec2_peft"):
        model_fp, processor, _ = load_trained_wav2vec2(ckpt_path, device=resolve_device("cpu"))

        blank_id = int(model_fp.config.pad_token_id)
        tok = processor.tokenizer
        id2token = {i: t for i, t in enumerate(tok.convert_ids_to_tokens(range(len(tok))))}

        for split_name, mp in manifests.items():
            ds = ManifestDataset(mp, max_items=max_utts)
            split_rec: dict[str, Any] = {}

            for scheme in schemes:
                if scheme == "fp32":
                    dev = device
                    model = model_fp.to(dev)
                    use_amp = False
                elif scheme == "fp16_cuda":
                    dev = resolve_device("cuda")
                    model = model_fp.to(dev)
                    use_amp = dev.type == "cuda"
                elif scheme == "int8_dynamic_cpu":
                    # PEFT LoRA modules are not compatible with torch dynamic quantization:
                    # quantize_dynamic turns the LoRA A/B Linear weights into packed/functional
                    # representations, but PEFT's forward path expects `lora_A.weight.dtype`.
                    if model_kind == "wav2vec2_peft":
                        continue
                    model = dynamic_int8_quantize_cpu(model_fp)
                    dev = resolve_device("cpu")
                    use_amp = False
                else:
                    continue

                refs: list[str] = []
                hyps: list[str] = []

                t0 = time.perf_counter()
                audio_sec = 0.0
                for ex in ds:
                    wav = load_audio(ex.wav_path, sample_rate)
                    audio_sec += float(wav.numel()) / float(sample_rate)
                    inputs = processor([wav.numpy()], sampling_rate=sample_rate, return_tensors="pt", padding=True)
                    iv = inputs["input_values"].to(dev)
                    am = inputs.get("attention_mask")
                    if am is not None:
                        am = am.to(dev)

                    with torch.no_grad():
                        if use_amp:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                logits = model(input_values=iv, attention_mask=am).logits.squeeze(0).detach().cpu()
                        else:
                            logits = model(input_values=iv, attention_mask=am).logits.squeeze(0).detach().cpu()

                    hyp = greedy_decode_logits(logits, blank_id=blank_id, id2token=id2token, word_delim="|")
                    hyps.append(hyp)
                    refs.append(ex.text)

                dt = time.perf_counter() - t0
                split_rec[scheme] = {
                    "wer": compute_wer(refs, hyps).wer,
                    "rtf": float(dt) / max(1e-9, audio_sec),
                    "model_size_mb": model_state_size_mb(model),
                    "num_utts": len(refs),
                }

            out[split_name] = split_rec

        return out

    raise ValueError(f"Unknown model kind for quant eval: {model_kind}")


def run_evaluation(
    *,
    cfg: dict[str, Any],
    only_decoding: str | None = None,
    only_eval: str | None = None,
) -> dict[str, Any]:
    """Run evaluation and return a JSON-serializable dict."""

    run_name = str(cfg.get("run_name") or "run")
    device = resolve_device(cfg.get("training", {}).get("device", "auto"))
    sample_rate = int(cfg.get("dataset", {}).get("sample_rate", 16000))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))

    decoding = list(cfg.get("eval", {}).get("decoding", ["greedy"]))
    if only_decoding is not None:
        decoding = [only_decoding]

    split_group = _split_manifests(cfg)
    splits = list(cfg.get("eval", {}).get("splits", ["dev", "test"]))

    ckpt_base = Path(cfg.get("checkpoints", {}).get("dir", "artifacts/checkpoints"))

    results: dict[str, Any] = {
        "meta": _env_meta(),
        "run_name": run_name,
        "config": cfg,
        "experiments": {},
    }

    def ckpt_path(exp: str) -> Path:
        p = ckpt_base / run_name / exp / "best.pt"
        if not p.exists():
            raise FileNotFoundError(f"Missing checkpoint for {exp}: {p} (run training first)")
        return p

    max_utts_stream = cfg.get("eval", {}).get("streaming", {}).get("max_utts_per_split")
    max_utts_quant = cfg.get("eval", {}).get("quantization", {}).get("max_utts_per_split")

    # Baseline (greedy only)
    if only_eval in (None, "wer"):
        baseline_out: dict[str, Any] = {}
        for s in splits:
            baseline_out[s] = _eval_baseline_wer(
                ckpt_path=ckpt_path("baseline"),
                manifests=split_group[s],
                sample_rate=sample_rate,
                device=device,
                batch_size=int(cfg["training"]["baseline"]["batch_size"]),
                num_workers=num_workers,
                max_utts=None,
            )
        results["experiments"]["baseline"] = {"wer": baseline_out}

        # wav2vec2 full / peft
        lm_arpa_gz = Path(cfg.get("lm", {}).get("arpa_gz")) if cfg.get("lm", {}).get("arpa_gz") else None
        for exp in ("wav2vec2_full", "wav2vec2_peft"):
            out_exp: dict[str, Any] = {}
            for s in splits:
                out_exp[s] = _eval_w2v2_wer(
                    ckpt_path=ckpt_path(exp),
                    manifests=split_group[s],
                    sample_rate=sample_rate,
                    device=device,
                    batch_size=int(cfg["training"][exp]["batch_size"]),
                    num_workers=num_workers,
                    max_utts=None,
                    decoding=decoding,
                    lm_arpa_gz=lm_arpa_gz,
                )
            results["experiments"][exp] = {"wer": out_exp}

    # Streaming eval (wav2vec2)
    if only_eval in (None, "streaming") and bool(cfg.get("eval", {}).get("streaming", {}).get("enabled", False)):
        stream_cfg = cfg["eval"]["streaming"]
        for exp in ("wav2vec2_full", "wav2vec2_peft"):
            by_split: dict[str, Any] = {}
            for s in splits:
                by_split[s] = _streaming_eval_w2v2(
                    ckpt_path=ckpt_path(exp),
                    manifests=split_group[s],
                    sample_rate=sample_rate,
                    device=device,
                    chunk_sec=list(stream_cfg.get("chunk_sec", [0.5])),
                    left_context_sec=list(stream_cfg.get("left_context_sec", [0.0])),
                    warmup_chunks=int(stream_cfg.get("warmup_chunks", 0)),
                    max_utts=int(max_utts_stream) if max_utts_stream is not None else None,
                )
            results["experiments"].setdefault(exp, {})["streaming"] = by_split

    # Quant eval
    if only_eval in (None, "quant") and bool(cfg.get("eval", {}).get("quantization", {}).get("enabled", False)):
        qcfg = cfg["eval"]["quantization"]
        schemes = list(qcfg.get("schemes", ["fp32"]))

        baseline_quant: dict[str, Any] = {}
        for s in splits:
            baseline_quant[s] = _quant_eval(
                model_kind="baseline",
                ckpt_path=ckpt_path("baseline"),
                manifests=split_group[s],
                sample_rate=sample_rate,
                device=device,
                schemes=schemes,
                max_utts=int(max_utts_quant) if max_utts_quant is not None else None,
            )
        results["experiments"]["baseline"]["quantization"] = baseline_quant

        for exp in ("wav2vec2_full", "wav2vec2_peft"):
            qout: dict[str, Any] = {}
            for s in splits:
                qout[s] = _quant_eval(
                    model_kind=exp,
                    ckpt_path=ckpt_path(exp),
                    manifests=split_group[s],
                    sample_rate=sample_rate,
                    device=device,
                    schemes=schemes,
                    max_utts=int(max_utts_quant) if max_utts_quant is not None else None,
                )
            results["experiments"].setdefault(exp, {})["quantization"] = qout

    return results
