"""Reusable client for Binance public historical data archives.

Data source:
https://github.com/binance/binance-public-data
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from zipfile import ZipFile

import requests


KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]

AGG_TRADES_COLUMNS = [
    "aggregate_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "is_buyer_maker",
    "is_best_match",
]

TRADES_COLUMNS = [
    "trade_id",
    "price",
    "qty",
    "quote_qty",
    "time",
    "is_buyer_maker",
    "is_best_match",
]


@dataclass
class FetchSummary:
    """Result of one fetch job."""

    source: str
    market: str
    dataset: str
    frequency: str
    symbol: str
    interval: Optional[str]
    periods: list[str]
    downloaded: int
    cache_hits: int
    skipped_missing: int
    zip_files: list[str]
    extracted_csv_files: list[str]
    missing_urls: list[str]

    def to_dict(self) -> dict:
        """Serialize summary for logs or JSON output."""
        return asdict(self)


class BinancePublicDataClient:
    """Client for downloading and reading Binance public historical archives."""

    def __init__(
        self,
        base_url: str = "https://data.binance.vision/data",
        cache_dir: str | Path = "data_cache/binance_public_data",
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = requests.Session()

    @staticmethod
    def normalize_symbol(symbol: str, quote_asset: str = "USDT") -> str:
        """Normalize symbol from formats like BTC/USD, BTCUSDT, or BTC."""
        s = symbol.strip().upper()
        if "/" in s:
            base, quote = s.split("/", 1)
            # Binance spot mainly uses USDT rather than USD suffix.
            if quote == "USD":
                quote = "USDT"
            return f"{base}{quote}"
        if "-" in s:
            base, quote = s.split("-", 1)
            if quote == "USD":
                quote = "USDT"
            return f"{base}{quote}"
        # If plain base asset is provided, append quote asset.
        if s.isalpha() and quote_asset and not s.endswith(quote_asset.upper()):
            return f"{s}{quote_asset.upper()}"
        return s

    @staticmethod
    def _parse_date(value: str) -> date:
        """Parse YYYY-MM-DD."""
        return datetime.strptime(value, "%Y-%m-%d").date()

    @staticmethod
    def _shift_month(d: date, delta: int) -> date:
        """Shift date by N months, preserving month granularity."""
        year = d.year + (d.month - 1 + delta) // 12
        month = (d.month - 1 + delta) % 12 + 1
        return date(year, month, 1)

    def _resolve_periods(
        self,
        frequency: str,
        start_date: Optional[str],
        end_date: Optional[str],
        limit: Optional[int],
    ) -> list[str]:
        """Resolve date windows into daily/monthly archive period strings."""
        if frequency not in {"daily", "monthly"}:
            raise ValueError("frequency must be one of: daily, monthly")

        today = datetime.utcnow().date()
        default_end = today - timedelta(days=1)
        if frequency == "monthly":
            default_end = date(default_end.year, default_end.month, 1)

        end = self._parse_date(end_date) if end_date else default_end
        if frequency == "monthly":
            end = date(end.year, end.month, 1)

        if start_date:
            start = self._parse_date(start_date)
            if frequency == "monthly":
                start = date(start.year, start.month, 1)
        else:
            if limit is None:
                raise ValueError("Please provide start_date or limit.")
            if frequency == "daily":
                start = end - timedelta(days=limit - 1)
            else:
                start = self._shift_month(end, -(limit - 1))

        if start > end:
            raise ValueError("start_date must be <= end_date.")

        periods: list[str] = []
        cursor = start
        while cursor <= end:
            if frequency == "daily":
                periods.append(cursor.strftime("%Y-%m-%d"))
                cursor += timedelta(days=1)
            else:
                periods.append(cursor.strftime("%Y-%m"))
                cursor = self._shift_month(cursor, 1)

        if limit is not None and len(periods) > limit:
            periods = periods[-limit:]
        return periods

    def _build_zip_url_and_path(
        self,
        market: str,
        frequency: str,
        dataset: str,
        symbol: str,
        period: str,
        interval: Optional[str],
    ) -> tuple[str, Path]:
        """Build remote URL and local zip path."""
        if dataset == "klines" and not interval:
            raise ValueError("interval is required when dataset=klines")

        if market == "spot":
            prefix = f"spot/{frequency}/{dataset}/{symbol}"
        else:
            # Binance futures archive path uses "um" (USD-M) and "cm" (COIN-M).
            prefix = f"futures/{market}/{frequency}/{dataset}/{symbol}"

        if dataset == "klines":
            assert interval is not None
            filename = f"{symbol}-{interval}-{period}.zip"
            prefix = f"{prefix}/{interval}"
        else:
            filename = f"{symbol}-{dataset}-{period}.zip"

        url = f"{self.base_url}/{prefix}/{filename}"
        local = self.cache_dir / prefix / filename
        return url, local

    def _download_zip(self, url: str, path: Path) -> bool:
        """Download one zip archive to local path. Return True if downloaded."""
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = self.session.get(url, timeout=self.timeout, stream=True)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        with path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True

    @staticmethod
    def _extract_zip(path: Path) -> list[Path]:
        """Extract zip file and return extracted CSV file paths."""
        extract_dir = path.parent / path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        return sorted(p for p in extract_dir.glob("*.csv"))

    def fetch_history(
        self,
        symbol: str,
        dataset: str,
        frequency: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        interval: Optional[str] = None,
        market: str = "spot",
        quote_asset: str = "USDT",
        extract: bool = True,
        skip_missing: bool = True,
    ) -> FetchSummary:
        """Download historical archives by symbol/data-type/window.

        Args:
            symbol: e.g. BTC/USD, BTCUSDT, BTC
            dataset: one of klines, aggTrades, trades
            frequency: daily or monthly
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            limit: number of periods (daily or monthly based on frequency)
            interval: required for klines (e.g. 1m, 1h, 1d)
            market: spot, um (USD-M futures), or cm (COIN-M futures)
            quote_asset: used when symbol is base-only (e.g. BTC -> BTCUSDT)
            extract: auto-extract csv files from downloaded zip archives
            skip_missing: continue when an archive is missing (404)
        """
        if dataset not in {"klines", "aggTrades", "trades"}:
            raise ValueError("dataset must be one of: klines, aggTrades, trades")
        if market not in {"spot", "um", "cm"}:
            raise ValueError("market must be one of: spot, um, cm")

        normalized_symbol = self.normalize_symbol(symbol, quote_asset=quote_asset)
        periods = self._resolve_periods(
            frequency=frequency,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        downloaded = 0
        cache_hits = 0
        skipped_missing = 0
        zip_files: list[str] = []
        extracted_csv_files: list[str] = []
        missing_urls: list[str] = []

        for period in periods:
            url, local_zip = self._build_zip_url_and_path(
                market=market,
                frequency=frequency,
                dataset=dataset,
                symbol=normalized_symbol,
                period=period,
                interval=interval,
            )

            if local_zip.exists():
                cache_hits += 1
            else:
                ok = self._download_zip(url, local_zip)
                if not ok:
                    missing_urls.append(url)
                    skipped_missing += 1
                    if not skip_missing:
                        raise FileNotFoundError(f"Archive not found: {url}")
                    continue
                downloaded += 1

            zip_files.append(str(local_zip))

            if extract:
                for csv_path in self._extract_zip(local_zip):
                    extracted_csv_files.append(str(csv_path))

        return FetchSummary(
            source="binance_public_data",
            market=market,
            dataset=dataset,
            frequency=frequency,
            symbol=normalized_symbol,
            interval=interval,
            periods=periods,
            downloaded=downloaded,
            cache_hits=cache_hits,
            skipped_missing=skipped_missing,
            zip_files=zip_files,
            extracted_csv_files=extracted_csv_files,
            missing_urls=missing_urls,
        )

    @staticmethod
    def iter_csv_rows(
        csv_paths: Iterable[str | Path],
        columns: Optional[list[str]] = None,
        max_rows: Optional[int] = None,
    ) -> list[dict]:
        """Load CSV rows as dicts for downstream strategy pipelines."""
        rows: list[dict] = []
        for path in csv_paths:
            with Path(path).open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if columns:
                        data = {
                            columns[idx] if idx < len(columns) else f"col_{idx}": value
                            for idx, value in enumerate(row)
                        }
                    else:
                        data = {f"col_{idx}": value for idx, value in enumerate(row)}
                    rows.append(data)
                    if max_rows is not None and len(rows) >= max_rows:
                        return rows
        return rows
