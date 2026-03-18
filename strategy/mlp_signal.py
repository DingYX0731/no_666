"""Backward-compatible MLP signal wrapper."""

from __future__ import annotations

from .mlp_strategy import MLPCheckpointStrategy


def generate_mlp_signal(
    prices: list[float],
    position_coin: float,
    ckpt_path: str,
    threshold_buy: float = 0.55,
    threshold_sell: float = 0.45,
) -> str:
    """Generate BUY/SELL/HOLD using MLP strategy class."""
    strategy = MLPCheckpointStrategy(
        ckpt_path=ckpt_path,
        threshold_buy=threshold_buy,
        threshold_sell=threshold_sell,
    )
    return strategy.generate_signal(prices=prices, position_coin=position_coin)
