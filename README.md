# Testing Order Book Imbalance in Crude Oil Futures

## What the results show

This project tests whether extreme order book imbalance (OBI) predicts short-term price movements in CME crude oil futures. The completed sensitivity study provides an important negative result: on five October 2026 trading dates, none of the four tested OBI thresholds produced positive net PnL. The default 0.75 threshold generated 30,418 round trips and lost $500,032 after fees. At 0.95, the loss was smaller ($11,564), but the strategy still lost money on all five daily files. These findings do not establish a profitable threshold; they show why transaction costs, execution assumptions, and out-of-sample testing matter.

| OBI threshold | Round trips | Wins | Gross PnL | Fees | Net PnL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 30,418 | 45 | -$378,360 | $121,672 | -$500,032 |
| 0.85 | 10,710 | 27 | -$113,700 | $42,840 | -$156,540 |
| 0.90 | 3,495 | 10 | -$35,430 | $13,980 | -$49,410 |
| 0.95 | 761 | 5 | -$8,520 | $3,044 | -$11,564 |

The study used 12,813,597 book events from five full CL MBP-10 daily files covering UTC dates October 5–10, 2026; the shorter October 4 file was excluded. Each day and threshold was backtested independently with $100,000 starting cash, one contract, two-tick take-profit and stop-loss settings, and $2 in fees per contract per side. Daily PnL is summed across independent runs, not compounded. At 0.75, each daily run reached the equity stop and stopped opening new positions. Raising the threshold reduced trade count and losses, but the result at 0.95 is not evidence that it is a suitable setting.

Detailed daily and aggregate metrics are available in [`threshold_optimization/daily_results.csv`](threshold_optimization/daily_results.csv) and [`threshold_optimization/threshold_summary.csv`](threshold_optimization/threshold_summary.csv). The analysis is exploratory and in-sample: it covers one contract and five dates, and is neither a holdout evaluation nor a statistical significance test. Further research should use additional markets and periods, a time-separated holdout, and realistic execution-cost assumptions before drawing conclusions about predictive value.

## Data and signal

The project uses Databento Market by Price data with ten levels of CME Globex order-book depth. The downloader supports continuous front-month CL, GC, and SI symbols; the stored threshold study specifically evaluates CL. For each event, OBI compares weighted bid and ask displayed sizes:

`OBI = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)`

The default calculation uses the best bid and ask (level 0). The implementation can also combine selected levels from 0 to 9, assigning higher linear weights to levels nearer the top of the book. OBI ranges from -1 to +1. The signal generator labels observations above the configured positive threshold as BUY, below its negative as SELL, and all others as neutral. The default threshold of 0.75 is a configurable research baseline, not a validated trading parameter.

## Event-driven backtest

The Python backtester consumes the book events and generated signals, opens a position only when flat, and closes it at the configured profit or loss distance, when OBI crosses back through zero, near the session close, or at the end of the data. Decisions fill on a later valid event rather than at the quote that generated the signal. Buys cross at the ask and sells at the bid, with configurable exchange and routing fees. Optional entry filters limit spread and require displayed size at the touch.

The engine excludes crossed books and events that do not complete an exchange packet, avoids the CME maintenance break and weekends, and stops entering shortly before the daily close. It records both realized cash flows and mark-to-market equity. These safeguards make the simulation more explicit than a signal-only calculation, but it remains a backtest: displayed liquidity, modeled fills, and configured costs are not a guarantee of live execution.

Backtests write a fill-level trade log and equity curve to `results/`. The reporting module summarizes PnL, win rate, trade count, and drawdown, and can generate an equity chart and a price/OBI chart. A separate spread-and-size analysis measures forward mid-price movement at 100 ms, 1 second, and 10 seconds, compares it with estimated round-trip spread and fees, and writes a CSV summary and chart. These reports are generated when the corresponding commands are run; they are not included as precomputed project outputs.

## Reproducing the threshold study

`src/threshold_analysis.py` reuses the OBI and backtest implementations to compare thresholds by day. The checked-in CSV files are the results described above. The companion [`threshold_optimization/README.md`](threshold_optimization/README.md) documents the study configuration and rerun options. Unit tests cover the backtester, reporting, spread/size analysis, and threshold analysis.

## Running the project

Install dependencies with `pip install -r requirements.txt`. To download data, provide a Databento API key in a root `.env` file as `DATABENTO_API_KEY=your_key_here`. The downloader estimates request size and cost and prompts before fetching unless `--yes` is supplied.

From the project root:

```bash
# Fetch the default CL sample (October 4–10, 2026; end date is exclusive).
python src/fetch_mbp_data.py

# Backtest the available CL data with the default OBI threshold.
python src/backtester.py --symbol CL --imbalance_threshold 0.75

# Reproduce the five-day threshold comparison.
python src/threshold_analysis.py --start-date 2026-10-05 --end-date 2026-10-10

# Summarize a backtest and generate its charts.
python src/reporting.py --symbol CL --imbalance_threshold 0.75

# Run the forward-move spread/size analysis and tests.
python src/spread_size_analysis.py --symbol CL
pytest tests/
```

The backtester also accepts alternative symbols, thresholds, tick targets, fee assumptions, latency, spread/depth filters, input directories, and output directories. Use `--help` on a script for its complete options. The threshold analysis writes daily and aggregate CSVs under `threshold_optimization/`; other commands write to the `results/` and `visualizations/` directories unless configured otherwise.
