from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text(encoding="utf-8"))

    run_name = res.get("run_name", "run")
    meta = res.get("meta", {})
    cfg = res.get("config", {})
    exps = res.get("experiments", {})

    lines: list[str] = []
    lines.append(f"# Report: {run_name}")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(meta, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("This report summarizes:")
    lines.append("- Baseline: log-mel + BiLSTM CTC (trained from scratch; greedy decoding)")
    lines.append("- wav2vec2 full fine-tuning vs LoRA PEFT")
    lines.append("- Optional decoding (greedy vs KenLM beam search, if enabled)")
    lines.append("- Streaming simulation (chunk size vs left-context)")
    lines.append("- Deployability (fp16/fp32 vs int8 dynamic quantization)")
    lines.append("")

    def fmt(x: Any) -> str:
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    # WER tables
    lines.append("## WER")
    lines.append("")

    for exp_name, exp in exps.items():
        wer = exp.get("wer")
        if not wer:
            continue
        for split_group, per_group in wer.items():
            # per_group: subset -> metrics (baseline) OR subset -> {mode -> metrics} (wav2vec2)
            rows = []
            if exp_name == "baseline":
                for subset, m in per_group.items():
                    rows.append([exp_name, split_group, subset, "greedy", fmt(m.get("wer")), str(m.get("num_utts"))])
            else:
                for subset, modes in per_group.items():
                    for mode, m in modes.items():
                        rows.append([exp_name, split_group, subset, mode, fmt(m.get("wer")), str(m.get("num_utts"))])

            if rows:
                lines.append(_md_table(["model", "split", "subset", "decoding", "WER", "N"], rows))
                lines.append("")

    # Streaming
    lines.append("## Streaming (Chunked Inference)")
    lines.append("")
    for exp_name in ["wav2vec2_full", "wav2vec2_peft"]:
        exp = exps.get(exp_name) or {}
        streaming = exp.get("streaming")
        if not streaming:
            continue
        for split_group, per_group in streaming.items():
            rows = []
            for subset, settings in per_group.items():
                for setting_key, m in settings.items():
                    rows.append(
                        [
                            exp_name,
                            split_group,
                            subset,
                            fmt(m.get("chunk_sec")),
                            fmt(m.get("left_context_sec")),
                            fmt(m.get("wer")),
                            fmt(m.get("rtf_mean")),
                            fmt(m.get("avg_chunk_ms_mean")),
                            str(m.get("num_utts")),
                        ]
                    )
            if rows:
                lines.append(_md_table(["model", "split", "subset", "chunk_s", "left_s", "WER", "RTF", "chunk_ms", "N"], rows))
                lines.append("")

    # Quant
    lines.append("## Quantization / Precision")
    lines.append("")
    for exp_name, exp in exps.items():
        q = exp.get("quantization")
        if not q:
            continue
        rows = []
        for split_group, per_group in q.items():
            for subset, schemes in per_group.items():
                for scheme, m in schemes.items():
                    rows.append(
                        [
                            exp_name,
                            split_group,
                            subset,
                            scheme,
                            fmt(m.get("wer")),
                            fmt(m.get("rtf")),
                            fmt(m.get("model_size_mb")),
                            str(m.get("num_utts")),
                        ]
                    )
        if rows:
            lines.append(_md_table(["model", "split", "subset", "scheme", "WER", "RTF", "size_mb", "N"], rows))
            lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
