# trade

Trading runtime and exchange integration layer.

## Files

- `client.py`: Roostoo REST API client (signed requests, order operations)
- `trader_engine.py`: multi-symbol trading loop (`single`, `multi`, `all`)
- `logging_utils.py`: per-run log folder/file creation

## Notes

- Use `run_trader.py` as the CLI entrypoint.
- Strategy parameters are loaded from `configs/strategies/<strategy>.yaml`.
- Logs are generated per run under `logs/trading/YYYYMMDD/run_xxx/trader.log`.
