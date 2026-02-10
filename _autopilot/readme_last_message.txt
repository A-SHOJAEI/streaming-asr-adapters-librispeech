Overwrote `README.md` with a repo-specific writeup grounded in:

- `artifacts/report.md` (all reported tables referenced and the exact values copied into the README)
- `artifacts/results.json` (config highlights and where metrics live in JSON)
- Implemented code in `asr/` and `scripts/` (dataset generation/download, model/training, streaming eval, quantization, optional KenLM)

It includes: problem statement, dataset provenance (synthetic + OpenSLR LibriSpeech/LM), methodology, explicit baselines/ablations, exact results tables with pointers to the canonical tables in `artifacts/report.md`, repro commands (`make all`, partial targets, full LibriSpeech run, KenLM enablement), limitations, and concrete next research steps.