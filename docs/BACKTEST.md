# Backtest

## Running a backtest

### Synthetic data (smoke test)

```bash
python run_backtest.py \
  --data-source synthetic \
  --strategy ma \
  --save-artifacts
```

### Buy and hold (`buy_hold`)

One-shot BUY on the first bar with available quote, then HOLD:

```bash
python run_backtest.py \
  --data-source synthetic \
  --strategy buy_hold \
  --strategy-config configs/strategies/buy_hold.yaml \
  --save-artifacts
```

**End equity vs start:** Each bar, after trades, equity is `quote + base * close`. **End Equity** uses the **last bar’s close** (mark-to-market). For buy-and-hold, End Equity usually differs from Start Equity because you hold base whose value moves with the final price (and the entry fee reduces equity right after the first BUY). They can be numerically close only if the asset round-trips in price and fees are small.

### Binance historical data (recommended layout)

Use the **same symbol style** as live trading (`BTC/USD` is normalized to Binance `BTCUSDT` when `quote_asset` is USDT). Always set **interval**, **frequency**, **market**, **quote asset**, **date range** (or `--limit`), **strategy**, and optionally **strategy YAML**.

**Moving average (`ma`) — spot, hourly klines, January 2024**

```bash
python run_backtest.py \
  --data-source binance \
  --symbol BTC/USD \
  --interval 1h \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --strategy ma \
  --strategy-config configs/strategies/ma.yaml \
  --save-artifacts
```

**MLP (`mlp`) — same product and window**

```bash
python run_backtest.py \
  --data-source binance \
  --symbol BTC/USD \
  --interval 1h \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --strategy mlp \
  --strategy-config configs/strategies/mlp.yaml \
  --save-artifacts
```

**DRL (`drl`) — requires a trained checkpoint for this pair under `checkpoints/drl/`**

```bash
python run_backtest.py \
  --data-source binance \
  --symbol BTC/USD \
  --interval 1h \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --strategy drl \
  --strategy-config configs/strategies/drl.yaml \
  --save-artifacts
```

**Alternative: date-less window via `--limit`** (fetch last N periods; see `data/market_dataset.py` / fetcher behavior)

```bash
python run_backtest.py \
  --data-source binance \
  --symbol ETH/USD \
  --interval 4h \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --limit 180 \
  --strategy ma \
  --strategy-config configs/strategies/ma.yaml \
  --save-artifacts
```

### CSV file (your own bars)

```bash
python run_backtest.py \
  --data-source csv \
  --csv path/to/your_data.csv \
  --price-col close \
  --strategy ma \
  --strategy-config configs/strategies/ma.yaml \
  --save-artifacts
```

*(Binance-sourced runs attach `features` + `time_ms` for logging; CSV runs log prices only.)*

---

## Artifacts under `logs/` (categorized)

Per-run outputs align with **`logs/training/...`** and **`logs/trading/...`**:

| Root | When | Contents |
|------|------|----------|
| `logs/backtest/YYYYMMDD/<HHMMSS>_<PAIR>_<strategy>/` | `--save-artifacts` | `steps.csv`, `backtest_chart.png` |

- **`--save-artifacts`** — create the directory above under `--artifact-base` (default `logs/backtest`).
- **`--report-dir path/to/dir`** — write the **same filenames** (`steps.csv`, `backtest_chart.png`) under an explicit directory (overrides the auto path for both files).
- **`--step-log`** / **`--plot`** — override **individual** output paths; parent dirs are **not** auto-created (create them yourself or use `--save-artifacts` / `--report-dir`).

Save the terminal summary next to the run:

```bash
python run_backtest.py \
  --data-source binance \
  --symbol BTC/USD \
  --interval 1h \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --start-date 2024-01-01 \
  --end-date 2024-01-15 \
  --strategy drl \
  --strategy-config configs/strategies/drl.yaml \
  --save-artifacts \
  2>&1 | tee logs/backtest_console_$(date -u +%Y%m%d_%H%M%S).log
```

---

## Step log CSV columns (`steps.csv`)

| Column | Meaning |
|--------|---------|
| `step` | Bar index (0-based). |
| `time_ms` | Kline open time (ms) when using Binance dataset; empty for synthetic/CSV. |
| `price` | Close at this bar. |
| `quote` | Free quote balance **before** executing the signal at this bar. |
| `base_coin` | Base position **before** executing the signal. |
| `equity_before` | `quote + base * price` before the trade. |
| `signal` | `BUY` / `SELL` / `HOLD` from the strategy. |
| `executed` | `yes` if a trade was actually filled at this bar; `no` if signal was ignored (e.g. BUY with no cash). |
| `conf_hold`, `conf_buy`, `conf_sell` | Strategy-reported confidences when `evaluate_step` provides them (NaN omitted for simple strategies). |
| `strategy_input` | Short JSON / text summary of inputs (model-dependent). |

---

## Console result block

```
=== Backtest Result ===
Strategy     : drl
Need Prices  : 21
Start Equity : 10000.00  (initial quote cash)
End Price    : 43250.12345678  (last bar, used for MTM)
End Equity   : 10523.45  (= quote + base * End Price)
Return       : 5.23%
Max Drawdown : 3.12%
Trades       : 42
```

---

## Result fields

| Field | Meaning |
|-------|---------|
| **Need Prices** | Minimum number of price bars the strategy needs before it can output BUY/SELL. Before this many bars, the strategy returns HOLD. E.g. MA needs short+long windows; DRL/MLP need lookback+1. |
| **Start Equity** | Initial **quote** cash before bar 0 (default 10,000). Not re-marked at the first open. |
| **End Price** | Close of the **last** bar in the series; used for final mark-to-market. |
| **End Equity** | After processing the last bar: `quote + base * End Price` (same formula as every bar in the equity curve). |
| **Return** | `(End Equity - Start Equity) / Start Equity`. Positive = profit. |
| **Max Drawdown** | Largest peak-to-trough decline during the backtest. `(peak - trough) / peak`. Lower is better; 10% means the worst drop from a prior high was 10%. |
| **Trades** | Number of BUY or SELL executions. |

## Max drawdown

**Max drawdown** measures the worst cumulative loss from a previous high. At each bar, we track the running peak equity; drawdown at that bar is `(peak - current_equity) / peak`. Max drawdown is the maximum of these values over the whole run.

- 0% = no drawdown (equity never fell from a prior high).
- 5% = at some point equity was 5% below a prior peak.
- 20% = at worst, equity dropped 20% from a prior peak.

It is a risk metric: lower max drawdown usually means a smoother equity curve.
