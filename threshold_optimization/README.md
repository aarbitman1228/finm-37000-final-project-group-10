# OBI Threshold Sensitivity Results

This folder contains the daily backtest output and the aggregate threshold
comparison:

- [`daily_results.csv`](daily_results.csv): one row per daily data file and OBI
  threshold, including trades, wins, gross PnL, fees, net PnL, and drawdown.
- [`threshold_summary.csv`](threshold_summary.csv): totals by threshold.
  Daily runs each start with $100,000, so summed PnL is not a compounded
  portfolio return.

## Reproducing the analysis

With the five full CL daily MBP-10 Parquet files for October 5–10, 2026 in
`data/raw/`, run from the project root:

```bash
python src/threshold_analysis.py \
  --start-date 2026-10-05 \
  --end-date 2026-10-10
```

The tool calculates level-0 OBI and runs the existing event-driven backtester
independently on each daily file at 0.75, 0.85, 0.90, and 0.95. Each daily run
uses the backtester's default one-contract quantity, two-tick take-profit and
stop-loss, fees, trading-hour and book-quality filters, and $100,000 initial
cash. It uses the backtester's normal end-of-data liquidation. The shorter
October 4 file is excluded by the date filter. Override `--thresholds` to
compare other valid values.

The sample is exploratory and limited to CL and five dates. No tested threshold
was profitable on every file; these results do not establish a production
threshold or constitute an out-of-sample validation.
