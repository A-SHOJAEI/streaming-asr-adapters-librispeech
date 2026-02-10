from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from asr.data.dataset import Example, ManifestDataset, load_audio
from asr.data.vocab import build_char_vocab_from_manifests
from asr.eval.inference import decode_logits_kenlm
from asr.eval.inference import decode_batch_greedy_ctc
from asr.eval.metrics import compute_wer
from asr.models.wav2vec2_ctc import apply_lora, load_wav2vec2_ctc
from asr.train.checkpoint import load_checkpoint, make_ckpt_paths, save_checkpoint
from asr.utils.device import resolve_device
from asr.utils.repro import set_reproducibility


def _write_w2v2_vocab_json(path: Path, *, chars: list[str]) -> None:
    """Write a HF tokenizer vocab.json.

    We use a standard Wav2Vec2 CTC vocabulary layout:
      - <pad> is the CTC blank
      - | is the word delimiter (space)
    """

    vocab: dict[str, int] = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
    next_id = 5
    for ch in chars:
        if ch == " ":
            continue
        if ch in vocab:
            continue
        vocab[ch] = next_id
        next_id += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def build_processor_from_manifests(manifest_paths: list[str | Path], cache_dir: Path) -> tuple[Wav2Vec2Processor, Path]:
    vocab = build_char_vocab_from_manifests(manifest_paths)
    chars = [t for t in vocab.tokens if t not in ("<blank>",)]

    vocab_json = cache_dir / "vocab.json"
    _write_w2v2_vocab_json(vocab_json, chars=chars)

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_json),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        do_lower_case=True,
    )
    feat = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feat, tokenizer=tokenizer), vocab_json


def _collate_w2v2(batch: list[Example], processor: Wav2Vec2Processor, sample_rate: int) -> dict[str, Any]:
    wavs = [load_audio(ex.wav_path, sample_rate).numpy() for ex in batch]
    texts = [ex.text for ex in batch]

    inputs = processor(wavs, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Convert spaces to | for the tokenizer.
    t2 = [t.replace(" ", "|") for t in texts]
    # Avoid `as_target_processor()` (deprecated and has API-edge-cases across versions).
    labels = processor.tokenizer(t2, return_tensors="pt", padding=True)

    # HF expects -100 for padded label tokens.
    label_ids = labels["input_ids"]
    label_mask = labels.get("attention_mask")
    if label_mask is not None:
        label_ids = label_ids.masked_fill(label_mask == 0, -100)

    inputs["labels"] = label_ids
    inputs["text"] = texts
    return inputs


@torch.no_grad()
def evaluate_wav2vec2(
    *,
    model: Wav2Vec2ForCTC,
    loader: DataLoader,
    processor: Wav2Vec2Processor,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()

    refs: list[str] = []
    hyps: list[str] = []

    blank_id = int(model.config.pad_token_id)
    id2token = {i: t for i, t in enumerate(processor.tokenizer.convert_ids_to_tokens(range(model.config.vocab_size)))}

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        input_values = batch["input_values"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = model(input_values=input_values, attention_mask=attention_mask)
        logits = out.logits.transpose(0, 1)  # (T, B, V)

        # Estimate logit lengths from attention mask; fallback to full length.
        if attention_mask is None:
            logit_lengths = torch.full((logits.shape[1],), logits.shape[0], dtype=torch.long)
        else:
            # Wav2vec2 reduces time resolution by a fixed feature extractor stride.
            # We approximate by scaling lengths proportionally.
            in_l = attention_mask.sum(dim=1)
            ratio = logits.shape[0] / float(input_values.shape[1])
            logit_lengths = (in_l.float() * ratio).floor().clamp_min(1).to(torch.long)

        dec = decode_batch_greedy_ctc(
            logits=logits,
            logit_lengths=logit_lengths,
            texts=batch["text"],
            blank_id=blank_id,
            id2token=id2token,
            word_delim="|",
        )
        refs.extend(dec.refs)
        hyps.extend(dec.hyps)

    wer_res = compute_wer(refs, hyps)
    return {"wer": wer_res.wer, "num_utts": len(refs)}


def _make_loader(manifest: str | Path, *, batch_size: int, num_workers: int, processor: Wav2Vec2Processor, sample_rate: int, shuffle: bool) -> DataLoader:
    ds = ManifestDataset(manifest)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: _collate_w2v2(batch, processor, sample_rate),
    )


def _train_one(
    *,
    exp_name: str,
    run_name: str,
    train_manifest: str | Path,
    dev_manifests: dict[str, str | Path],
    cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    checkpoint_dir: str | Path,
    use_peft: bool,
) -> dict[str, Any]:
    seed = int(global_cfg["seed"])
    deterministic = bool(global_cfg.get("training", {}).get("deterministic", True))
    set_reproducibility(seed, deterministic)

    device = resolve_device(global_cfg.get("training", {}).get("device", "auto"))
    sample_rate = int(global_cfg.get("dataset", {}).get("sample_rate", 16000))

    num_workers = int(global_cfg.get("training", {}).get("num_workers", 0))

    ckpt = make_ckpt_paths(checkpoint_dir, run_name, exp_name)

    processor_name = cfg.get("processor_name_or_path")
    if processor_name:
        processor = Wav2Vec2Processor.from_pretrained(processor_name)
        vocab_json = None
    else:
        processor, vocab_json = build_processor_from_manifests([train_manifest], cache_dir=ckpt.dir / "tokenizer")

    model = load_wav2vec2_ctc(cfg["model_name_or_path"]).to(device)

    if use_peft:
        # Freeze all base weights explicitly, then add LoRA trainables.
        for p in model.parameters():
            p.requires_grad = False
        model = apply_lora(model, cfg.get("lora", {})).to(device)

    model.train()

    batch_size = int(cfg["batch_size"])
    train_loader = _make_loader(
        train_manifest,
        batch_size=batch_size,
        num_workers=num_workers,
        processor=processor,
        sample_rate=sample_rate,
        shuffle=True,
    )

    dev_loaders = {
        name: _make_loader(p, batch_size=batch_size, num_workers=num_workers, processor=processor, sample_rate=sample_rate, shuffle=False)
        for name, p in dev_manifests.items()
    }

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["lr"]))

    grad_clip = float(cfg.get("grad_clip", 1.0))
    epochs = int(cfg["epochs"])

    best_key = sorted(dev_loaders.keys())[0]
    best_wer = float("inf")

    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        pbar = tqdm(train_loader, desc=f"{exp_name} epoch {epoch}/{epochs}")
        for batch in pbar:
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch["labels"].to(device)

            out = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            if loss is None:
                raise RuntimeError("Model did not return loss; check label formatting")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        epoch_time = time.time() - start

        # Save last
        save_checkpoint(
            ckpt.last,
            {
                "model_type": "wav2vec2_ctc_peft" if use_peft else "wav2vec2_ctc_full",
                "hf_model_name_or_path": str(cfg["model_name_or_path"]),
                "processor_name_or_path": str(processor_name) if processor_name else None,
                "vocab_json": str(vocab_json) if vocab_json else None,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "global_cfg": global_cfg,
                "train_cfg": cfg,
            },
        )

        dev_metrics: dict[str, Any] = {}
        for name, loader in dev_loaders.items():
            dev_metrics[name] = evaluate_wav2vec2(model=model, loader=loader, processor=processor, device=device)

        key_wer = float(dev_metrics[best_key]["wer"])
        if key_wer < best_wer:
            best_wer = key_wer
            save_checkpoint(
                ckpt.best,
                {
                    "model_type": "wav2vec2_ctc_peft" if use_peft else "wav2vec2_ctc_full",
                    "hf_model_name_or_path": str(cfg["model_name_or_path"]),
                    "processor_name_or_path": str(processor_name) if processor_name else None,
                    "vocab_json": str(vocab_json) if vocab_json else None,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "global_cfg": global_cfg,
                    "train_cfg": cfg,
                    "best_dev_key": best_key,
                    "best_dev_wer": best_wer,
                },
            )

        history.append({"epoch": epoch, "epoch_sec": epoch_time, "dev": dev_metrics})

    return {
        "exp_name": exp_name,
        "checkpoint_best": str(ckpt.best),
        "checkpoint_last": str(ckpt.last),
        "best_dev_wer": best_wer,
        "history": history,
        "device": str(device),
    }


def train_wav2vec2_full(
    *,
    run_name: str,
    train_manifest: str | Path,
    dev_manifests: dict[str, str | Path],
    cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    return _train_one(
        exp_name="wav2vec2_full",
        run_name=run_name,
        train_manifest=train_manifest,
        dev_manifests=dev_manifests,
        cfg=cfg,
        global_cfg=global_cfg,
        checkpoint_dir=checkpoint_dir,
        use_peft=False,
    )


def train_wav2vec2_peft(
    *,
    run_name: str,
    train_manifest: str | Path,
    dev_manifests: dict[str, str | Path],
    cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    return _train_one(
        exp_name="wav2vec2_peft",
        run_name=run_name,
        train_manifest=train_manifest,
        dev_manifests=dev_manifests,
        cfg=cfg,
        global_cfg=global_cfg,
        checkpoint_dir=checkpoint_dir,
        use_peft=True,
    )


def load_trained_wav2vec2(
    path: str | Path,
    *,
    device: torch.device,
) -> tuple[Wav2Vec2ForCTC, Wav2Vec2Processor, dict[str, Any]]:
    ckpt = load_checkpoint(path, map_location="cpu")

    processor_name = ckpt.get("processor_name_or_path")
    if processor_name:
        processor = Wav2Vec2Processor.from_pretrained(processor_name)
    else:
        vocab_json = ckpt.get("vocab_json")
        if not vocab_json:
            raise RuntimeError("Checkpoint missing processor_name_or_path and vocab_json")
        tokenizer = Wav2Vec2CTCTokenizer(
            str(vocab_json),
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
            do_lower_case=True,
        )
        feat = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(feature_extractor=feat, tokenizer=tokenizer)

    model = load_wav2vec2_ctc(ckpt["hf_model_name_or_path"])  # initialize structure

    model_type = str(ckpt.get("model_type") or "")
    if "peft" in model_type:
        # Recreate the adapter structure before loading weights.
        for p in model.parameters():
            p.requires_grad = False
        model = apply_lora(model, (ckpt.get("train_cfg") or {}).get("lora", {}))

    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, processor, ckpt
