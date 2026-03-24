"""Strategy factory backed by YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import BaseStrategy
from .bb_rsi_strategy import BollingerRSIStrategy
from .buy_hold_strategy import BuyAndHoldStrategy
from .drl_strategy import DRLSb3Strategy
from .ma_strategy import MovingAverageCrossStrategy
from .mlp_strategy import MLPCheckpointStrategy

_STRATEGY_REGISTRY = {
    "ma": MovingAverageCrossStrategy,
    "mlp": MLPCheckpointStrategy,
    "drl": DRLSb3Strategy,
    "bb_rsi": BollingerRSIStrategy,
    "buy_hold": BuyAndHoldStrategy,
}


def _default_config_path(strategy_name: str) -> Path:
    return Path("configs/strategies") / f"{strategy_name}.yaml"


def _read_strategy_params(config_path: Path, strategy_name: str) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Strategy config not found: {config_path}. "
            f"Create it or pass --strategy-config."
        )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    declared_name = str(data.get("strategy", strategy_name)).strip().lower()
    if declared_name != strategy_name:
        raise ValueError(
            f"Config strategy mismatch: expected '{strategy_name}', got '{declared_name}' in {config_path}"
        )
    params = data.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"'params' must be a mapping in {config_path}")
    return params


def build_strategy(
    strategy_name: str,
    strategy_config: str = "",
    *,
    pair: str | None = None,
) -> BaseStrategy:
    """Build a concrete strategy instance from yaml.

    ``pair`` is required for ``drl`` (selects checkpoints/drl/<PAIR>_ppo.zip per product).
    """
    strategy_key = strategy_name.strip().lower()
    if strategy_key not in _STRATEGY_REGISTRY:
        raise ValueError(f"Unsupported strategy '{strategy_name}'. Available: {sorted(_STRATEGY_REGISTRY.keys())}")

    config_path = Path(strategy_config) if strategy_config else _default_config_path(strategy_key)
    params = _read_strategy_params(config_path, strategy_key)
    cls = _STRATEGY_REGISTRY[strategy_key]
    if strategy_key == "drl":
        if not pair:
            raise ValueError(
                "Strategy 'drl' requires a trading pair. "
                "Use run_trader with explicit --symbols or backtest with --symbol."
            )
        return cls(pair=pair.strip().upper(), **params)
    return cls(**params)
