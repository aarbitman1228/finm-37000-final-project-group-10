# Latency Re-evaluation Results (Issue #30)

This folder holds the output of `src/latency_analysis.py`, which re-runs the
strategy with realistic delays between CME and the algorithm. The findings are
summarized in the main README under
[Execution Latency](../README.md#execution-latency).

- [`feed_latency.csv`](feed_latency.csv): how far Databento's receive time
  lags the matching engine, and how fast and how stale the old next-event fill
  was, per day and pooled (`data_file = all`). Times are in microseconds.
- [`latency_cost_curve.csv`](latency_cost_curve.csv): for every OBI > 0.75
  entry signal, the price an order pays (and the mid move) if it reaches the
  exchange `latency_us` after Databento received the signal, in ticks
  relative to the quote the strategy saw. `model = next_event` is the
  original approximation.
- [`daily_results.csv`](daily_results.csv): one backtest per daily file,
  latency scenario and OBI threshold.
- [`latency_summary.csv`](latency_summary.csv): the daily backtests pooled by
  scenario and threshold. PnL is summed over independent daily runs, not
  compounded.
- [`latency_analysis.png`](latency_analysis.png): the cost curve and net PnL
  per round trip by scenario.

## Reproducing the analysis

With the five full CL daily MBP-10 Parquet files for October 5–10, 2026 in
`data/raw/`, run from the project root (about two minutes):

```bash
python src/latency_analysis.py --start_date 2026-10-05 --end_date 2026-10-10
```

Each daily file is backtested independently with the backtester defaults
(one contract, two-tick take-profit and stop-loss, $2.00 fees per contract
per side, trading-hour and book-quality filters) at OBI thresholds 0.75,
0.85, 0.90 and 0.95. Runs start with $10,000,000 instead of $100,000 so the
equity stop never ends a day early and every scenario trades the same hours;
since futures cash only moves by PnL, nothing else changes. Use `--scenarios`
and `--thresholds` to run a subset.

## Latency scenarios

One-way delays in microseconds. **Feed** is the delay after Databento's
receive time before the strategy sees an event, **order** is from the
decision until the order reaches the matching engine, and **report** is from
the fill until the strategy receives the fill report.

| Scenario | Feed | Order | Report | Basis |
| --- | ---: | ---: | ---: | --- |
| `next_event` | – | – | – | Original approximation: fill on the next event received |
| `zero` | 0 | 0 | 0 | No delay beyond Databento's own feed delay |
| `colocated` | 0 | 50 | 50 | Software strategy in CME's Aurora data center (backtester default) |
| `chicago_metro` | 290 | 340 | 340 | Colocated plus about 290 µs of fiber from 350 E Cermak |
| `new_york` | 4,000 | 4,050 | 4,050 | Colocated plus about 4 ms of microwave from New Jersey |
| `remote_10ms` | 10,000 | 10,050 | 10,050 | Sensitivity case: strategy far from the exchange |
| `remote_50ms` | 50,000 | 50,050 | 50,050 | Sensitivity case: strategy far from the exchange |

The feed delay of the data itself is not an assumption: Databento captures
CME in Aurora, so `ts_recv - ts_event` is the measured delay to a colocated
receiver (median 0.56 ms, 1.37 ms at entry signals). The order and report
delays cannot be observed in market data and are round-number assumptions
based on public figures:

- Databento's CME capture gateways are in Aurora, and 350 E Cermak is "at
  least 287 microseconds one-way on fiber" away
  ([Databento](https://databento.com/blog/cme-colocation)).
- CME reported a median inbound latency of 52 µs from the router to the
  match engine in 2012
  ([Markets Media](https://www.marketsmedia.com/taking-variability-out-of-velocity/)).
- The fastest Chicago–New Jersey (NY4) route measured 3.962 ms one-way in 2020
  ([Bhattacherjee et al., IMC 2020](https://bdebopam.github.io/papers/imc2020-hft.pdf)).

The results are reported across the whole range, so no single assumption
drives the conclusion.

## Caveats

The sample is CL only and five dates, in-sample. The model is for one-lot
orders that take liquidity: there is no queue position or market impact, and
the order does not change the historical book it trades against. Delays are
constant unless `--latency_jitter_us` is set. An order that would reach the
exchange after the last record of a file is not filled, because the book there
is unknown.
