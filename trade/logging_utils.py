"""Logging utilities for trade and training runtime."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path


def setup_training_logger(
    base_dir: str = "logs/training",
    run_prefix: str = "run",
    run_name: str | None = None,
) -> tuple[logging.Logger, Path]:
    """Create a per-run logger for training (MLP, DRL) writing to a unique file.

    Folder structure:
    logs/training/YYYYMMDD/<run_prefix>_<HHMMSS>/train.log
    """
    now = datetime.utcnow()
    day = now.strftime("%Y%m%d")
    run_name = run_name or f"{run_prefix}_{now.strftime('%H%M%S')}"
    log_dir = Path(base_dir) / day / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"

    logger = logging.getLogger(f"no_666_training_{day}_{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger, log_path


def setup_run_logger(base_dir: str = "logs/trading", run_name: str | None = None) -> tuple[logging.Logger, Path]:
    """Create a per-run logger writing to a unique file.

    Folder structure:
    logs/trading/YYYYMMDD/<run_name>/trader.log
    """
    now = datetime.utcnow()
    day = now.strftime("%Y%m%d")
    run_name = run_name or now.strftime("run_%H%M%S")
    log_dir = Path(base_dir) / day / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "trader.log"

    logger = logging.getLogger(f"no_666_trader_{day}_{run_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger, log_path


def prepare_backtest_artifact_dir(
    *,
    base_dir: str = "logs/backtest",
    symbol: str,
    strategy: str,
) -> Path:
    """Create a per-run folder for backtest CSV + charts (same style as training/trading logs).

    Layout::
        logs/backtest/YYYYMMDD/<HHMMSS>_<PAIRSLUG>_<strategy>/

    Typical files written there by ``backtest.run_backtest``::
        steps.csv, backtest_chart.png
    """
    now = datetime.utcnow()
    day = now.strftime("%Y%m%d")
    ts = now.strftime("%H%M%S")
    sym = symbol.strip().upper().replace("/", "").replace("-", "")
    pair_slug = re.sub(r"[^A-Z0-9]", "", sym) or "PAIR"
    strat_slug = re.sub(r"[^a-z0-9]+", "_", strategy.strip().lower()).strip("_") or "strategy"
    run_name = f"{ts}_{pair_slug}_{strat_slug}"
    out = Path(base_dir) / day / run_name
    out.mkdir(parents=True, exist_ok=True)
    return out
