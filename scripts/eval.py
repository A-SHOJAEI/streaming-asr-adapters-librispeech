from __future__ import annotations

import argparse

from asr.config import load_config
from asr.eval.runner import run_evaluation
from asr.utils.io import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="artifacts/results.json")
    ap.add_argument("--only-decoding", choices=["greedy", "kenlm"], default=None)
    ap.add_argument("--only-eval", choices=["wer", "streaming", "quant"], default=None)
    args = ap.parse_args()

    cfg = load_config(args.config).raw
    results = run_evaluation(cfg=cfg, only_decoding=args.only_decoding, only_eval=args.only_eval)
    write_json(args.out, results)


if __name__ == "__main__":
    main()
