# Backtest

## Running a backtest

```bash
# Synthetic data (quick test)
python run_backtest.py --data-source synthetic --strategy ma

# Binance historical data
python run_backtest.py --data-source binance --symbol BTC/USD --interval 1h \
  --start-date 2024-01-01 --end-date 2024-02-01 --strategy drl

# CSV file
python run_backtest.py --data-source csv --csv your_data.csv --price-col close --strategy mlp
```

## Output and where to find it

Backtest results are printed to **stdout** (terminal). Example:

```
=== Backtest Result ===
Strategy     : drl
Need Prices  : 21
Start Equity : 10000.00
End Equity   : 10523.45
Return       : 5.23%
Max Drawdown : 3.12%
Trades       : 42
```

There is no log file by default. To save output to a file:

```bash
python run_backtest.py --data-source binance --symbol BTC/USD --strategy drl \
  --start-date 2024-01-01 --end-date 2024-02-01 2>&1 | tee logs/backtest_$(date +%Y%m%d_%H%M%S).log
```

## Result fields

| Field | Meaning |
|-------|---------|
| **Need Prices** | Minimum number of price bars the strategy needs before it can output BUY/SELL. Before this many bars, the strategy returns HOLD. E.g. MA needs short+long windows; DRL/MLP need lookback+1. |
| **Start Equity** | Initial capital (default 10,000). |
| **End Equity** | Final portfolio value (cash + position × last price). |
| **Return** | `(End Equity - Start Equity) / Start Equity`. Positive = profit. |
| **Max Drawdown** | Largest peak-to-trough decline during the backtest. `(peak - trough) / peak`. Lower is better; 10% means the worst drop from a prior high was 10%. |
| **Trades** | Number of BUY or SELL executions. |

## Max drawdown

**Max drawdown** measures the worst cumulative loss from a previous high. At each bar, we track the running peak equity; drawdown at that bar is `(peak - current_equity) / peak`. Max drawdown is the maximum of these values over the whole run.

- 0% = no drawdown (equity never fell from a prior high).
- 5% = at some point equity was 5% below a prior peak.
- 20% = at worst, equity dropped 20% from a prior peak.

It is a risk metric: lower max drawdown usually means a smoother equity curve.
