import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _to_bool(value: str, default: bool = False) -> bool:
    """Convert common truthy strings to bool."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime settings loaded from environment variables."""
    base_url: str
    api_key: str
    api_secret: str
    poll_seconds: int
    short_window: int
    long_window: int
    max_position_usd: float
    max_daily_loss_pct: float
    min_notional_usd: float
    max_consecutive_errors: int
    dry_run: bool

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from .env / process environment and validate."""
        load_dotenv()
        base_url = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com").rstrip("/")
        api_key = os.getenv("ROOSTOO_API_KEY", "")
        api_secret = os.getenv("ROOSTOO_API_SECRET", "")

        settings = cls(
            base_url=base_url,
            api_key=api_key,
            api_secret=api_secret,
            poll_seconds=int(os.getenv("POLL_SECONDS", "5")),
            short_window=int(os.getenv("SHORT_WINDOW", "5")),
            long_window=int(os.getenv("LONG_WINDOW", "20")),
            max_position_usd=float(os.getenv("MAX_POSITION_USD", "1000")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05")),
            min_notional_usd=float(os.getenv("MIN_NOTIONAL_USD", "5")),
            max_consecutive_errors=int(os.getenv("MAX_CONSECUTIVE_ERRORS", "5")),
            dry_run=_to_bool(os.getenv("DRY_RUN", "true"), default=True),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        """Basic guardrails to avoid invalid runtime parameters."""
        if not self.api_key or not self.api_secret:
            raise ValueError("Missing ROOSTOO_API_KEY or ROOSTOO_API_SECRET in environment.")
        if self.short_window <= 0 or self.long_window <= 0:
            raise ValueError("SHORT_WINDOW and LONG_WINDOW must be positive.")
        if self.poll_seconds <= 0:
            raise ValueError("POLL_SECONDS must be positive.")
