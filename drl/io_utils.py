"""Save/load DRL run metadata next to SB3 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def pair_to_slug(pair: str) -> str:
    """BTC/USD -> BTC_USD for filesystem-safe names."""
    return pair.strip().upper().replace("/", "_").replace(" ", "_")


def save_drl_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_drl_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
