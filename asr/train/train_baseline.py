from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from asr.data.dataset import ManifestDataset, collate_ctc_labels, collate_waveforms
from asr.data.vocab import CharVocab, build_char_vocab_from_manifests
from asr.eval.inference import decode_batch_greedy_ctc
from asr.eval.metrics import compute_wer
from asr.models.baseline_bilstm_ctc import BaselineConfig, LogMelBiLSTMCTC
from asr.train.checkpoint import load_checkpoint, make_ckpt_paths, save_checkpoint
from asr.utils.device import resolve_device
from asr.utils.repro import set_reproducibility


def _make_loader(manifest: str | Path, *, batch_size: int, num_workers: int, sample_rate: int, shuffle: bool) -> DataLoader:
    ds = ManifestDataset(manifest)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_waveforms(batch, sample_rate=sample_rate),
    )


@torch.no_grad()
def evaluate_baseline(
    *,
    model: LogMelBiLSTMCTC,
    loader: DataLoader,
    vocab: CharVocab,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    refs: list[str] = []
    hyps: list[str] = []

    blank_id = vocab.blank_id
    id2token = vocab.id_to_token()

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        wav = batch["waveform"].to(device)
        wav_l = batch["waveform_lengths"].to(device)
        logits, logit_l = model(wav, wav_l)
        dec = decode_batch_greedy_ctc(
            logits=logits, logit_lengths=logit_l, texts=batch["text"], blank_id=blank_id, id2token=id2token, word_delim=" "
        )
        refs.extend(dec.refs)
        hyps.extend(dec.hyps)

    wer_res = compute_wer(refs, hyps)
    return {"wer": wer_res.wer, "num_utts": len(refs)}


def train_baseline(
    *,
    run_name: str,
    train_manifest: str | Path,
    dev_manifests: dict[str, str | Path],
    cfg: dict[str, Any],
    global_cfg: dict[str, Any],
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Train from-scratch log-mel + BiLSTM CTC baseline."""

    seed = int(global_cfg["seed"])
    deterministic = bool(global_cfg.get("training", {}).get("deterministic", True))
    set_reproducibility(seed, deterministic)

    device = resolve_device(global_cfg.get("training", {}).get("device", "auto"))

    sample_rate = int(global_cfg.get("dataset", {}).get("sample_rate", 16000))

    batch_size = int(cfg["batch_size"])
    num_workers = int(global_cfg.get("training", {}).get("num_workers", 0))

    vocab = build_char_vocab_from_manifests([train_manifest])

    mcfg = cfg.get("model", {})
    model_cfg = BaselineConfig(
        vocab_size=len(vocab),
        sample_rate=sample_rate,
        n_mels=int(mcfg.get("n_mels", 80)),
        lstm_hidden=int(mcfg.get("lstm_hidden", 512)),
        lstm_layers=int(mcfg.get("lstm_layers", 3)),
    )
    model = LogMelBiLSTMCTC(model_cfg).to(device)

    train_loader = _make_loader(
        train_manifest, batch_size=batch_size, num_workers=num_workers, sample_rate=sample_rate, shuffle=True
    )

    dev_loaders = {
        name: _make_loader(p, batch_size=batch_size, num_workers=num_workers, sample_rate=sample_rate, shuffle=False)
        for name, p in dev_manifests.items()
    }

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    grad_clip = float(cfg.get("grad_clip", 1.0))

    epochs = int(cfg["epochs"])

    ckpt = make_ckpt_paths(checkpoint_dir, run_name, "baseline")
    best_key = sorted(dev_loaders.keys())[0]
    best_wer = float("inf")

    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        pbar = tqdm(train_loader, desc=f"baseline epoch {epoch}/{epochs}")
        for batch in pbar:
            wav = batch["waveform"].to(device)
            wav_l = batch["waveform_lengths"].to(device)
            labels, label_l = collate_ctc_labels(batch["text"], vocab)
            labels = labels.to(device)
            label_l = label_l.to(device)

            logits, logit_l = model(wav, wav_l)  # (T, B, V)
            log_probs = logits.log_softmax(dim=-1)

            loss = ctc(log_probs, labels, logit_l, label_l)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        epoch_time = time.time() - start

        # Save last
        save_checkpoint(
            ckpt.last,
            {
                "model_type": "baseline_bilstm_ctc",
                "model_cfg": asdict(model_cfg),
                "state_dict": model.state_dict(),
                "vocab": vocab.tokens,
                "epoch": epoch,
                "global_cfg": global_cfg,
                "train_cfg": cfg,
            },
        )

        dev_metrics: dict[str, Any] = {}
        for name, loader in dev_loaders.items():
            dev_metrics[name] = evaluate_baseline(model=model, loader=loader, vocab=vocab, device=device)

        key_wer = float(dev_metrics[best_key]["wer"])
        if key_wer < best_wer:
            best_wer = key_wer
            save_checkpoint(
                ckpt.best,
                {
                    "model_type": "baseline_bilstm_ctc",
                    "model_cfg": asdict(model_cfg),
                    "state_dict": model.state_dict(),
                    "vocab": vocab.tokens,
                    "epoch": epoch,
                    "global_cfg": global_cfg,
                    "train_cfg": cfg,
                    "best_dev_key": best_key,
                    "best_dev_wer": best_wer,
                },
            )

        history.append({"epoch": epoch, "epoch_sec": epoch_time, "dev": dev_metrics})

    return {
        "exp_name": "baseline",
        "checkpoint_best": str(ckpt.best),
        "checkpoint_last": str(ckpt.last),
        "best_dev_wer": best_wer,
        "history": history,
        "device": str(device),
    }


def load_trained_baseline(path: str | Path, *, device: torch.device) -> tuple[LogMelBiLSTMCTC, CharVocab, dict[str, Any]]:
    ckpt = load_checkpoint(path, map_location="cpu")
    vocab = CharVocab(tokens=list(ckpt["vocab"]))
    cfg = BaselineConfig(**ckpt["model_cfg"])
    model = LogMelBiLSTMCTC(cfg)
    model.load_state_dict(ckpt["state_dict"])  # type: ignore[arg-type]
    model.to(device)
    return model, vocab, ckpt
