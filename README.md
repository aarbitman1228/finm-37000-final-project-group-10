# Testing Order Book Imbalance in Crude Oil Futures

## What the results show

This project tests whether extreme order book imbalance (OBI) predicts short-term price movements in CME crude oil futures. The completed study provides an important negative result: on five October 2026 trading dates, none of the four tested OBI thresholds produced positive net PnL. The default 0.75 threshold generated 30,418 round trips and lost $500,032 after fees. At 0.95, the loss was smaller ($11,564), but the strategy still lost money on all five daily files. These findings do not establish a profitable threshold; they show why transaction costs, execution assumptions, and out-of-sample testing matter.

| OBI threshold | Round trips | Wins | Gross PnL | Fees | Net PnL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 30,418 | 45 | -$378,360 | $121,672 | -$500,032 |
| 0.85 | 10,710 | 27 | -$113,700 | $42,840 | -$156,540 |
| 0.90 | 3,495 | 10 | -$35,430 | $13,980 | -$49,410 |
| 0.95 | 761 | 5 | -$8,520 | $3,044 | -$11,564 |

![Net PnL by OBI threshold, split into gross trading loss and fees, and net PnL of every daily run](threshold_optimization/threshold_study.png)

The study used 12,813,597 book events from five full CL MBP-10 daily files covering UTC dates October 5–10, 2026; the shorter October 4 file was excluded. Each day and threshold was backtested independently with $100,000 starting cash, one contract, two-tick take-profit and stop-loss settings, and $2 in fees per contract per side. Daily PnL is summed across independent runs, not compounded. At 0.75, each daily run reached the equity stop and stopped opening new positions. Raising the threshold reduced trade count and losses, but the result at 0.95 is not evidence that it is a suitable setting.

The reason is the size of the edge. After an extreme imbalance, the mid price moves only about 0.3 ticks in the predicted direction, while crossing the spread in and out costs at least one tick plus about 0.4 ticks of fees. With realistic exchange latency (see [Execution latency](#execution-latency)), every threshold loses more, because most of the predicted move happens before an order can reach the exchange. The signal is real, but too weak to pay for taking liquidity.

Detailed daily and aggregate metrics are available in [`threshold_optimization/daily_results.csv`](threshold_optimization/daily_results.csv) and [`threshold_optimization/threshold_summary.csv`](threshold_optimization/threshold_summary.csv). The analysis is exploratory and in-sample: it covers one contract and five dates, and is neither a holdout evaluation nor a statistical significance test. Further research should use additional markets and periods, a time-separated holdout, and realistic execution-cost assumptions before drawing conclusions about predictive value.

## Data and signal

The project uses Databento Market by Price data with ten levels of CME Globex order-book depth. The downloader supports continuous front-month CL, GC, and SI symbols; the stored studies specifically evaluate CL. For each event, OBI compares weighted bid and ask displayed sizes:

`OBI = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)`

The default calculation uses the best bid and ask (level 0). The implementation can also combine selected levels from 0 to 9, assigning higher linear weights to levels nearer the top of the book. OBI ranges from -1 to +1. The signal generator labels observations above the configured positive threshold as BUY, below its negative as SELL, and all others as neutral. The default threshold of 0.75 is a configurable research baseline, not a validated trading parameter.

## Event-driven backtest

The Python backtester (`src/backtester.py`) replays the book events in time order and simulates the strategy on the generated signals. It opens a position only when flat, and closes it at the configured profit or loss distance, when OBI crosses back through zero, near the session close, or at the end of the data. A decision never fills at the quote that generated it. Buys cross at the ask and sells at the bid, with configurable exchange and routing fees. Optional entry filters limit spread and require displayed size at the touch.

By default, orders are filled with a latency model. Every record has two clocks: `ts_event` (CME's matching engine) and `ts_recv` (Databento's capture server, which sits with the matching engine in Aurora). The strategy sees an event after `ts_recv`; its order reaches the matching engine 50 µs after the decision and executes against the book as it stood at that moment in exchange time, often a book the strategy has not received yet. The fill report takes another 50 µs, and only then does the strategy act on its new position. These defaults describe a software strategy colocated with CME; feed, order and report delays and random jitter are configurable. `--fill_model next_event` keeps the original approximation, which fills on the next event received, and reproduces the threshold study above.

The engine excludes crossed books and events that do not complete an exchange packet, avoids the CME maintenance break and weekends, and stops entering shortly before the daily close. A $100,000 futures portfolio records realized cash flows and mark-to-market equity, and new entries stop once equity reaches zero. These safeguards make the simulation more explicit than a signal-only calculation, but it remains a backtest: displayed liquidity, modeled fills, and configured costs are not a guarantee of live execution.

![Equity curve of the default backtest over October 5–9, 2026](figures/equity_curve.png)

Run over the whole week at the default 0.75 threshold, the strategy makes 5,184 round trips with a 0.17% win rate and loses its $100,000 within the first trading day (net −$100,016: −$79,280 gross and $20,736 in fees), after which the equity stop blocks new entries. Almost all trades (4,968 of 5,184) close because level-0 OBI flips sign within milliseconds (median holding 3.6 ms), which locks in the spread.

Backtests write a fill-level trade log and equity curve to `results/`. The reporting module summarizes PnL, win rate, trade count, and drawdown, checks the trade log against the engine's own PnL, and can generate an equity chart and a price/OBI chart.

## Spread and displayed size

`src/spread_size_analysis.py` measures, for every entry signal, how far the mid price moves in the signal's direction after 100 ms, 1 second, and 10 seconds, and compares it with the estimated round-trip cost of crossing the spread plus fees.

![Net edge of OBI signals by bid-ask spread and by size at the touch](figures/spread_size_analysis.png)

The net edge is negative in every bucket and at every horizon. Signals on a 1-tick spread come closest, at about −1.1 ticks, and wider spreads only add cost: the forward move stays near 0.3 ticks while the cost grows with the spread. Displayed size at the touch barely changes the result. The edge is measured from the signal's own quote, as if orders filled instantly, so it is optimistic.

## Execution latency

The original fill rule, filling on the next event Databento received, turned out to be both too fast and stale. On the five full CL days (12.3 million tradable events and 325,785 entry signals at OBI > 0.75), the next event arrived a median 0.21 ms after the signal, faster than any real order can react. And because Databento's capture lags the matching engine by a median 0.56 ms (1.37 ms at entry signals, which arrive in bursts), for 81% of signals the book the old rule filled against had already existed at the exchange before the decision was made.

`src/latency_analysis.py` re-ran every daily file under seven latency scenarios and four thresholds (140 backtests). Order and fill-report delays cannot be observed in market data, so they are round-number assumptions from public figures, reported across a wide range. Full tables, sources, and caveats are in [`latency_analysis/`](latency_analysis/).

| Scenario | One-way delays: feed / order / fill report |
| --- | --- |
| Next event (old model) | fill on the next event received |
| Zero | 0 / 0 / 0 µs (only Databento's own feed delay) |
| Colocated (default) | 0 / 50 / 50 µs |
| Chicago metro | 290 / 340 / 340 µs |
| New York | 4.00 / 4.05 / 4.05 ms |
| Remote 10 ms / 50 ms | 10 / 10.05 / 10.05 ms and 50 / 50.05 / 50.05 ms |

![Entry slippage by latency and net PnL per round trip by scenario](latency_analysis/latency_analysis.png)

The old rule paid 0.12 ticks more than the quote the strategy saw at entry. With the order reaching the exchange the instant Databento receives the signal, an entry pays 0.22 ticks; at 100 ms, 0.33 ticks. Most of that is the mid price moving in the signal's direction before the order arrives, so most of the predicted 0.3-tick move is gone before any order can trade on it.

| Scenario (OBI > 0.75, five days) | Round trips | Net PnL | Net per round trip | Entry slippage (ticks) |
| --- | ---: | ---: | ---: | ---: |
| Next event (old model) | 86,707 | −$1,344,608 | −$15.51 | 0.11 |
| Zero | 87,850 | −$1,500,070 | −$17.08 | 0.22 |
| Colocated (default) | 87,628 | −$1,505,452 | −$17.18 | 0.23 |
| Chicago metro | 84,689 | −$1,476,426 | −$17.43 | 0.25 |
| New York | 69,982 | −$1,235,168 | −$17.65 | 0.27 |
| Remote 10 ms | 63,658 | −$1,136,222 | −$17.85 | 0.28 |
| Remote 50 ms | 53,449 | −$989,526 | −$18.51 | 0.31 |

These runs start with enough cash that the equity stop never ends a day early, so every scenario trades the same hours; that is why the totals are larger than in the threshold study. Every latency scenario loses $1.6 to $5.9 more per round trip than the old approximation, which was optimistic rather than conservative. Slower setups make fewer trades but are no better per trade. None of the 140 daily runs made money. Rerunning the threshold study with the default colocated latency model gives net PnL of −$190,082 at 0.85, −$65,882 at 0.90, and −$17,986 at 0.95, versus the next-event figures in the first table; 0.75 still reaches the equity stop on every day.

The latency model covers one-lot orders that take liquidity: there is no queue position or market impact, and the order does not change the historical book it trades against.

## Time of day and signal timing

The notebook [`Notebook/time_signals.ipynb`](Notebook/time_signals.ipynb) examines when signals occur and how long they last, on the same CL sample.

![Average absolute OBI by hour of day, Chicago time](figures/time_of_day_abs_obi.png)

Signals are not spread evenly through the day. At 1:00 PM Chicago time, about 7.72% of market events generate an active signal, the highest hourly share; the same hour has both the deepest top of book (12.16 contracts on average) and the most imbalanced book (average absolute OBI of 0.368). Signal events also arrive in bursts, averaging about 21 ms since the previous event, compared with 33.78 ms for events without a signal. And signals are short-lived: a Buy or Sell signal lasts a median of two consecutive events. A time-of-day filter, event spacing as a feature, or a minimum persistence requirement are possible next steps; none has been backtested, and the patterns should be checked over more weeks.

## Reproducing the results

`src/threshold_analysis.py` reuses the OBI and backtest implementations to compare thresholds by day; the checked-in CSV files are the results described above, and the companion [`threshold_optimization/README.md`](threshold_optimization/README.md) documents the study configuration and rerun options. `src/latency_analysis.py` produces the latency study in [`latency_analysis/`](latency_analysis/). Unit tests cover the backtester, latency analysis, reporting, spread/size analysis, and threshold analysis.

## Running the project

Install dependencies with `pip install -r requirements.txt`. To download data, provide a Databento API key in a root `.env` file as `DATABENTO_API_KEY=your_key_here`. The downloader estimates request size and cost and prompts before fetching unless `--yes` is supplied.

From the project root:

```bash
# Fetch the default CL sample (October 4–10, 2026; end date is exclusive).
python src/fetch_mbp_data.py

# Backtest the available CL data with the default OBI threshold and latency model.
python src/backtester.py --symbol CL --imbalance_threshold 0.75

# Summarize a backtest and generate its charts.
python src/reporting.py --symbol CL --imbalance_threshold 0.75

# Reproduce the five-day threshold comparison (next-event fills, as in the table above) and its chart.
python src/threshold_analysis.py --start-date 2026-10-05 --end-date 2026-10-10 --fill-model next_event
python src/plot_threshold_study.py

# Run the forward-move spread/size analysis.
python src/spread_size_analysis.py --symbol CL

# Re-evaluate the strategy under each latency scenario (about two minutes).
python src/latency_analysis.py --start_date 2026-10-05 --end_date 2026-10-10

# Run the tests.
pytest tests/
```

The backtester also accepts alternative symbols, thresholds, tick targets, fee assumptions, fill models and latencies, spread/depth filters, input directories, and output directories. Use `--help` on a script for its complete options. The threshold and latency analyses write their CSVs and charts under `threshold_optimization/` and `latency_analysis/`; other commands write to the `results/` and `visualizations/` directories unless configured otherwise. The charts in `figures/` are copies of the `visualizations/` output for the default run and of the time-of-day notebook.
