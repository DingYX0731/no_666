"""Shared utilities for DRL model artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def pair_to_slug(pair: str) -> str:
    """Convert pair into filesystem-safe slug, e.g. BTC/USD -> BTC_USD."""
    return pair.strip().upper().replace("/", "_").replace(" ", "_")


def save_drl_meta(path: Path, meta: dict[str, Any]) -> None:
    """Save DRL metadata next to checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_drl_meta(path: Path) -> dict[str, Any]:
    """Load DRL metadata from json."""
    return json.loads(path.read_text(encoding="utf-8"))

