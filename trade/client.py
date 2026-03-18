"""Roostoo REST client used by trading runtime and CLI tools."""

import hashlib
import hmac
import time
from typing import Any, Dict, Optional, Tuple

import requests


class RoostooClient:
    """Thin API client for Roostoo REST endpoints."""

    def __init__(self, base_url: str, api_key: str, api_secret: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.session = requests.Session()

    @staticmethod
    def _timestamp_ms() -> str:
        """Return a 13-digit millisecond timestamp string."""
        return str(int(time.time() * 1000))

    @staticmethod
    def _sorted_params(payload: Dict[str, Any]) -> str:
        """Sort parameters by key and serialize as k=v&k2=v2."""
        keys = sorted(payload.keys())
        return "&".join(f"{k}={payload[k]}" for k in keys)

    def _sign(self, total_params: str) -> str:
        """Generate HMAC-SHA256 signature for request parameters."""
        return hmac.new(
            self.api_secret.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _signed_headers(self, signature: str, is_post: bool = False) -> Dict[str, str]:
        """Build signed headers required by RCL_TopLevelCheck endpoints."""
        headers = {
            "RST-API-KEY": self.api_key,
            "MSG-SIGNATURE": signature,
        }
        if is_post:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Dict[str, Any]:
        """Send HTTP request and return JSON response or raise RuntimeError."""
        payload = dict(payload or {})
        url = f"{self.base_url}{path}"
        method = method.upper()

        headers = {}
        data = None
        params = None

        if signed:
            payload["timestamp"] = self._timestamp_ms()
            total_params = self._sorted_params(payload)
            signature = self._sign(total_params)
            headers = self._signed_headers(signature, is_post=(method == "POST"))
            if method == "POST":
                data = total_params
            else:
                url = f"{url}?{total_params}"
        else:
            if method == "GET":
                params = payload
            else:
                data = payload

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            response_text = ""
            if exc.response is not None:
                response_text = f" | response={exc.response.text}"
            raise RuntimeError(f"Request failed: {method} {path}{response_text}") from exc

    def get_server_time(self) -> Dict[str, Any]:
        """GET /v3/serverTime."""
        return self._request("GET", "/v3/serverTime", signed=False)

    def get_exchange_info(self) -> Dict[str, Any]:
        """GET /v3/exchangeInfo."""
        return self._request("GET", "/v3/exchangeInfo", signed=False)

    def get_ticker(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """GET /v3/ticker for one pair or all pairs."""
        payload = {}
        if pair:
            payload["pair"] = pair
        payload["timestamp"] = self._timestamp_ms()
        return self._request("GET", "/v3/ticker", payload=payload, signed=False)

    def get_balance(self) -> Dict[str, Any]:
        """GET /v3/balance (signed)."""
        return self._request("GET", "/v3/balance", signed=True)

    def get_pending_count(self) -> Dict[str, Any]:
        """GET /v3/pending_count (signed)."""
        return self._request("GET", "/v3/pending_count", signed=True)

    def place_order(
        self,
        pair: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """POST /v3/place_order."""
        payload: Dict[str, Any] = {
            "pair": pair,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT order requires price.")
            payload["price"] = str(price)
        return self._request("POST", "/v3/place_order", payload=payload, signed=True)

    def query_order(
        self,
        order_id: Optional[str] = None,
        pair: Optional[str] = None,
        pending_only: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """POST /v3/query_order."""
        payload: Dict[str, Any] = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
        return self._request("POST", "/v3/query_order", payload=payload, signed=True)

    def cancel_order(self, order_id: Optional[str] = None, pair: Optional[str] = None) -> Dict[str, Any]:
        """POST /v3/cancel_order."""
        payload: Dict[str, Any] = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair:
            payload["pair"] = pair
        return self._request("POST", "/v3/cancel_order", payload=payload, signed=True)

    @staticmethod
    def parse_last_price(ticker_resp: Dict[str, Any], pair: str) -> float:
        """Extract LastPrice from ticker response for a target pair."""
        data = ticker_resp.get("Data", {})
        pair_data = data.get(pair, {})
        if "LastPrice" not in pair_data:
            raise ValueError(f"Ticker response missing LastPrice for {pair}: {ticker_resp}")
        return float(pair_data["LastPrice"])

    @staticmethod
    def parse_wallet(balance_resp: Dict[str, Any], pair: str) -> Tuple[float, float]:
        """
        Parse free balances for base/quote coins from balance response.

        Roostoo may return either:
        - Wallet (older schema in docs)
        - SpotWallet (current schema observed in live response)
        """
        base_coin, quote_coin = pair.split("/")
        wallet = balance_resp.get("Wallet") or balance_resp.get("SpotWallet") or {}
        base_free = float(wallet.get(base_coin, {}).get("Free", 0.0))
        quote_free = float(wallet.get(quote_coin, {}).get("Free", 0.0))
        return base_free, quote_free
