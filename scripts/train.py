from __future__ import annotations

import argparse
from pathlib import Path

from asr.config import load_config
from asr.train.train_baseline import train_baseline
from asr.train.train_wav2vec2 import train_wav2vec2_full, train_wav2vec2_peft


def _manifests_from_cfg(cfg: dict) -> tuple[Path, dict[str, Path]]:
    dcfg = cfg["dataset"]
    kind = dcfg["kind"]

    if kind == "synthetic":
        root = Path(dcfg["root"])
        train = root / "train.jsonl"
        dev = {"dev": root / "dev.jsonl"}
        return train, dev

    if kind == "librispeech":
        mdir = Path(dcfg["manifests_dir"])
        train = mdir / "train-clean-100.jsonl"
        dev = {"dev-clean": mdir / "dev-clean.jsonl", "dev-other": mdir / "dev-other.jsonl"}
        return train, dev

    raise ValueError(f"Unknown dataset.kind: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", choices=["baseline", "wav2vec2_full", "wav2vec2_peft"], default=None)
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    run_name = str(cfg.get("run_name") or "run")

    train_manifest, dev_manifests = _manifests_from_cfg(cfg)

    ckpt_dir = Path(cfg.get("checkpoints", {}).get("dir", "artifacts/checkpoints"))

    tcfg = cfg["training"]

    if args.only in (None, "baseline") and bool(tcfg.get("baseline", {}).get("enabled", False)):
        train_baseline(
            run_name=run_name,
            train_manifest=train_manifest,
            dev_manifests=dev_manifests,
            cfg=tcfg["baseline"],
            global_cfg=cfg,
            checkpoint_dir=ckpt_dir,
        )

    if args.only in (None, "wav2vec2_full") and bool(tcfg.get("wav2vec2_full", {}).get("enabled", False)):
        train_wav2vec2_full(
            run_name=run_name,
            train_manifest=train_manifest,
            dev_manifests=dev_manifests,
            cfg=tcfg["wav2vec2_full"],
            global_cfg=cfg,
            checkpoint_dir=ckpt_dir,
        )

    if args.only in (None, "wav2vec2_peft") and bool(tcfg.get("wav2vec2_peft", {}).get("enabled", False)):
        train_wav2vec2_peft(
            run_name=run_name,
            train_manifest=train_manifest,
            dev_manifests=dev_manifests,
            cfg=tcfg["wav2vec2_peft"],
            global_cfg=cfg,
            checkpoint_dir=ckpt_dir,
        )


if __name__ == "__main__":
    main()
