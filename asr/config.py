from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(RuntimeError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML must be a mapping; got {type(data)}")
    return data


def deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update nested dicts."""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def to_pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True)


@dataclass(frozen=True)
class ExperimentConfig:
    """Thin wrapper around a nested mapping with a few convenience helpers."""

    raw: dict[str, Any]
    path: Path

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.raw:
            raise ConfigError(f"Missing required config key: {key}")
        return self.raw[key]

    def dump(self) -> str:
        return to_pretty_json({"config_path": str(self.path), "config": self.raw})


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    raw = load_yaml(p)

    # Minimal validation for keys we rely on.
    if "seed" not in raw:
        raise ConfigError("Config must define 'seed'")
    if "dataset" not in raw or not isinstance(raw["dataset"], dict):
        raise ConfigError("Config must define 'dataset' mapping")
    if "training" not in raw or not isinstance(raw["training"], dict):
        raise ConfigError("Config must define 'training' mapping")
    if "eval" not in raw or not isinstance(raw["eval"], dict):
        raise ConfigError("Config must define 'eval' mapping")

    return ExperimentConfig(raw=raw, path=p)
