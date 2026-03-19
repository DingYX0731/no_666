"""YAML config loader for ML training pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load yaml file into dictionary."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def get_block(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """Get nested mapping block; return empty dict when missing."""
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config block '{key}' must be a mapping.")
    return value

