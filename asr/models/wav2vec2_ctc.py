from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


@dataclass(frozen=True)
class Wav2Vec2Config:
    model_name_or_path: str
    processor_name_or_path: str | None


def load_processor(processor_name_or_path: str | None) -> Wav2Vec2Processor | None:
    if processor_name_or_path is None:
        return None
    return Wav2Vec2Processor.from_pretrained(processor_name_or_path)


def load_wav2vec2_ctc(model_name_or_path: str) -> Wav2Vec2ForCTC:
    return Wav2Vec2ForCTC.from_pretrained(model_name_or_path)


def apply_lora(model: torch.nn.Module, lora_cfg: dict[str, Any]) -> torch.nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = list(lora_cfg.get("target_modules") or [])
    if not target_modules:
        raise ValueError("LoRA requires non-empty target_modules")

    # PEFT task types have evolved across versions. Older releases don't have a CTC task type and
    # route audio models through `PeftModelForFeatureExtraction` whose forward() expects `input_ids`.
    # We still want to fine-tune Wav2Vec2ForCTC with `input_values`, so we:
    # 1) configure LoRA with FEATURE_EXTRACTION (the closest available task type)
    # 2) return the underlying `base_model` (a LoraModel wrapper) which preserves the original
    #    forward signature and supports `input_values=...`.
    task_type = getattr(TaskType, "CTC", None) or getattr(TaskType, "FEATURE_EXTRACTION", None)
    if task_type is None:
        raise RuntimeError("Unsupported peft version: missing TaskType.FEATURE_EXTRACTION")

    cfg = LoraConfig(
        task_type=task_type,
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(model, cfg)
    return getattr(peft_model, "base_model", peft_model)
