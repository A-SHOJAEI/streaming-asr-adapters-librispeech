from __future__ import annotations

import argparse
from pathlib import Path

from asr.config import load_config
from asr.data.librispeech import download_librispeech, generate_synthetic_ctc_dataset, prepare_librispeech_manifests
from asr.utils.io import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    dcfg = cfg["dataset"]

    kind = dcfg["kind"]
    sample_rate = int(dcfg.get("sample_rate", 16000))

    if kind == "synthetic":
        root = Path(dcfg["root"])
        ensure_dir(root)
        generate_synthetic_ctc_dataset(
            root=root,
            sample_rate=sample_rate,
            num_train=int(dcfg["num_train"]),
            num_dev=int(dcfg["num_dev"]),
            num_test=int(dcfg["num_test"]),
            min_sec=float(dcfg["min_sec"]),
            max_sec=float(dcfg["max_sec"]),
            vocab=list(dcfg["vocab"]),
            seed=int(cfg["seed"]),
        )
        return

    if kind == "librispeech":
        raw_dir = Path(dcfg["raw_dir"])
        manifests_dir = Path(dcfg["manifests_dir"])
        ensure_dir(raw_dir)
        ensure_dir(manifests_dir)

        # Default plan splits.
        splits = ["train-clean-100", "dev-clean", "dev-other", "test-clean", "test-other"]

        download_librispeech(
            raw_dir=raw_dir,
            splits=splits,
            download_lm=bool(dcfg.get("download_lm", False)),
            verify=True,
        )
        prepare_librispeech_manifests(
            raw_dir=raw_dir,
            manifests_dir=manifests_dir,
            splits=splits,
            sample_rate=sample_rate,
            compute_duration=True,
        )
        return

    raise ValueError(f"Unknown dataset.kind: {kind}")


if __name__ == "__main__":
    main()
