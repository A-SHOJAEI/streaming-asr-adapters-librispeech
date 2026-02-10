PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CONFIG ?= configs/smoke.yaml

.PHONY: setup data train eval report all \
	baseline_train model_train_full model_train_peft decode_lm streaming_eval quant_eval \
	clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	bash scripts/bootstrap_venv.sh
	$(PY) -m scripts.sanity

data: setup
	$(PY) -m scripts.data --config $(CONFIG)

train: setup
	$(PY) -m scripts.train --config $(CONFIG)

eval: setup
	$(PY) -m scripts.eval --config $(CONFIG) --out artifacts/results.json

report: setup
	$(PY) -m scripts.report --results artifacts/results.json --out artifacts/report.md

all: data train eval report

# Plan-aligned convenience targets
baseline_train: setup
	$(PY) -m scripts.train --config $(CONFIG) --only baseline

model_train_full: setup
	$(PY) -m scripts.train --config $(CONFIG) --only wav2vec2_full

model_train_peft: setup
	$(PY) -m scripts.train --config $(CONFIG) --only wav2vec2_peft

decode_lm: setup
	$(PY) -m scripts.eval --config $(CONFIG) --only-decoding kenlm --out artifacts/results.json

streaming_eval: setup
	$(PY) -m scripts.eval --config $(CONFIG) --only-eval streaming --out artifacts/results.json

quant_eval: setup
	$(PY) -m scripts.eval --config $(CONFIG) --only-eval quant --out artifacts/results.json

clean:
	rm -rf $(VENV) artifacts/* data/raw data/manifests data/cache
	mkdir -p artifacts data
	: > artifacts/.gitkeep
	: > data/.gitkeep
