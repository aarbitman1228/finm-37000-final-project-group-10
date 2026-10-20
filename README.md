# High-Frequency Order Book Imbalance (OBI) Backtester

## Project Goal
The goal of this project is to develop an event-driven backtesting engine in Python that executes a micro-momentum trading strategy based on Order Book Imbalance (OBI) in commodity futures. 

Instead of relying on lagging indicators from executed trades, this project utilizes Level 2 order book depth to test whether the ratio of limit buy orders (bids) to limit sell orders (asks) resting in the book predicts short-term price movements. The current signal cutoff is a provisional heuristic, not an empirically established critical threshold. The strategy tests whether trades entered after an imbalance can capture a small tick-level profit before the book neutralizes.

## Data Source
*   **Provider:** Databento
*   **Schema:** Market by Price (MBP-10) - Top 10 levels of the order book.
*   **Market:** CME Group (NYMEX / COMEX via Globex)
*   **Target Assets:** Crude Oil (CL), Gold (GC), and Silver (SI). *Note: The architecture is modular, allowing the user to select which specific commodity to backtest.*

## Order Book Imbalance Calculation

The Order Book Imbalance (OBI) measures the relative difference between bid-side and ask-side liquidity at each market event.

OBI is calculated as:

`OBI = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)`

The implementation supports configurable order book levels (0–9), with level 0 (best bid and ask) used by default.

When multiple levels are selected, linearly decreasing weights are assigned based on order book depth:
* Level 0 receives the highest weight (10), decreasing to level 9 (1).
* Weights are normalized to sum to 1 across selected levels.
* Weighted bid and ask sizes are calculated before computing OBI.

The resulting OBI ranges from -1 to 1, where positive values indicate greater bid-side liquidity and negative values indicate greater ask-side liquidity.

## Trading Signal Generation

Trading signals are generated based on the calculated OBI using a configurable threshold (default: 0.75):

* **BUY (+1):** OBI > 0.75
* **SELL (-1):** OBI < -0.75
* **NEUTRAL (0):** Otherwise.

The default of 0.75 is a provisional, a priori choice; it has not been established as a statistically or economically meaningful cutoff. See [Critical Threshold Validation](#critical-threshold-validation) for the available sample analysis.

The script processes OBI data from `data/processed/` and saves the resulting signals in `data/signals/`. Previously generated files are reused when available.

## Critical Threshold Validation

**Conclusion:** The current sample does not empirically support 0.75 as a profitable critical threshold, and it does not establish another cutoff as suitable. The default remains unchanged as a configurable research baseline; it must not be interpreted as a validated trading parameter.

### Sample and method

The sensitivity check used the five full CL MBP-10 files for UTC dates October 5–10, 2026 (12,813,597 book events in total). The shorter October 4 file was excluded. OBI was calculated at level 0, as in the default configuration. Each daily file was backtested independently at thresholds 0.75, 0.85, 0.90, and 0.95 using the default one-contract size, two-tick take-profit and stop-loss, $2.00 per-contract-per-side fees, trading-hour/book-quality filters, $100,000 initial cash, and the original next-event fill approximation (`--fill-model next_event`). Each run starts with fresh cash and liquidates any remaining position at that file's end; the daily net PnL figures below are summed, not compounded. Positive round-trip PnL defines a win.

### Results

The detailed reproducible output is in [`threshold_optimization/`](threshold_optimization/): per-day metrics in [`daily_results.csv`](threshold_optimization/daily_results.csv), aggregate comparisons in [`threshold_summary.csv`](threshold_optimization/threshold_summary.csv), and rerun instructions in that folder's README.

| OBI threshold | Round trips | Winning trades | Gross PnL | Fees | Net PnL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.75 | 30,418 | 45 (0.15%) | -$378,360 | $121,672 | -$500,032 |
| 0.85 | 10,710 | 27 (0.25%) | -$113,700 | $42,840 | -$156,540 |
| 0.90 | 3,495 | 10 (0.29%) | -$35,430 | $13,980 | -$49,410 |
| 0.95 | 761 | 5 (0.66%) | -$8,520 | $3,044 | -$11,564 |

Every threshold lost money on every daily file. At 0.75, each daily run reached the equity stop that prevents new entries. Raising the cutoff reduced the number of trades and therefore the fees and losses, but even 0.95 lost money on all five files and produced only five winning round trips. This is evidence against treating 0.75 as a validated profitability threshold; the smaller loss at 0.95 is not evidence that 0.95 is a good threshold.

This is a limited, exploratory in-sample sensitivity check on one contract and five dates, not a holdout test or a statistical significance analysis. The results do not justify tuning the default to 0.95 (or any other tested value). Before selecting a production cutoff, evaluate additional periods and instruments, use a time-separated holdout, and account for selection bias and realistic execution costs.

With realistic message latency (the backtester's default colocated profile, see [Execution Latency](#execution-latency)) every threshold loses more: net PnL is −$190,082 at 0.85 (was −$156,540), −$65,882 at 0.90 (was −$49,410) and −$17,986 at 0.95 (was −$11,564), and 0.75 still reaches the equity stop on every day. Running `src/threshold_analysis.py` without `--fill-model next_event` gives these latency-model figures.


## Backtest Execution Engine

`src/backtester.py` steps through every MBP-10 book event in the order Databento received it and simulates trading on the signals:

* **Entry:** when flat, buy 1 contract on a BUY signal and sell 1 contract on a SELL signal.
* **Exit:** take-profit or stop-loss when the mid price has moved the set number of ticks (default 2) from the mid at entry, or when OBI crosses back through 0.
* **Execution (latency model, default):** every record has two clocks, `ts_event` (CME's matching engine) and `ts_recv` (Databento's capture server, which sits with the matching engine in Aurora). The strategy sees an event `--feed_latency_us` after `ts_recv`. Its order reaches the matching engine `--order_latency_us` after the decision and executes against the book as it stood at that moment in exchange time: the last event with `ts_event` at or before the arrival, often one the strategy has not received yet. The fill report arrives `--response_latency_us` later, and only then does the strategy act on its new position, using the newest book it has seen. Buys pay `ask_px_00` and sells receive `bid_px_00` of that book. The defaults (0 / 50 / 50 µs) describe a software strategy colocated in Aurora; `--latency_jitter_us` adds a random exponential delay to every order and report (seeded by `--seed`). `--fill_model next_event` keeps the original approximation, which fills on the next event received. See [Execution Latency](#execution-latency).
* **Book quality:** only events that complete an exchange packet (Databento `F_LAST` flag) with an uncrossed book are used, so trading halts and pre-open auctions are not traded.
* **Trading hours:** no trading during the CME 16:00–17:00 CT maintenance break or the weekend; positions are flattened and entries stop 1 minute before the 16:00 CT close. Any position left at the end of the data is closed at the last quote.
* **Portfolio:** starts with $100,000. As with futures, cash changes only by realized PnL and fees; equity also includes open-position PnL marked at the bid (long) or ask (short). New entries stop once equity reaches zero.
* **Fees:** $1.50 CME exchange fee plus $0.50 routing fee per contract per fill (`--exchange_fee`, `--routing_fee`). These are approximate and configurable.
* **Spread and size filters (optional):** `--max_spread_ticks N` skips entries when the bid-ask spread is wider than N ticks, and `--min_top_depth N` skips entries when best bid size plus best ask size is below N contracts. An entry also needs the side it crosses to show at least `--quantity` contracts. Exits are never filtered. MBP-10 data does not separate implied (spread-derived) orders, so "size" here means the displayed size at the touch.

Other options include `--stop_loss_ticks`, `--quantity` and `--flatten_minutes`. The trade log has one row per fill: `ts_event` is when the order executed at the exchange, `ts_recv` when the strategy received the fill report, `ts_sent` when it sent the order, and `decision_ts_recv` / `decision_price` give the book event and quote it decided on. The columns `ts_event, side, price, quantity, fee` are the input format for the reporting module.

## Execution Latency

The original engine filled each order on the next book event Databento received after the decision. Measured on the five full CL days (October 5–9, 2026; 12.3 million tradable events and 325,785 entry signals at OBI > 0.75), that approximation was both too fast and stale:

* **Too fast.** The next event arrived a median 0.21 ms after the signal, and within 50 µs for 20% of signals. No real order can react that quickly.
* **Stale.** Databento's capture lags the matching engine by a median 0.56 ms (90th percentile 5.8 ms), and by 1.37 ms at entry signals, which arrive in bursts. So for 81% of signals, the book the old rule filled against had already existed at the exchange *before* the decision was made (a median 1.1 ms earlier). The old trade logs show fills with a `ts_event` earlier than their own decision.

The latency model fixes both: an order executes against the exchange's book at the moment it arrives, and the strategy waits for its fill reports. `src/latency_analysis.py` re-ran every daily file under the scenarios below and four thresholds, starting each run with enough cash that the equity stop never ends a day early. Full tables, assumptions and sources are in [`latency_analysis/`](latency_analysis/).

| Scenario | One-way delays: feed / order / fill report |
| --- | --- |
| Next event (old model) | fill on the next event received |
| Zero | 0 / 0 / 0 µs (only Databento's own feed delay) |
| Colocated (default) | 0 / 50 / 50 µs |
| Chicago metro | 290 / 340 / 340 µs |
| New York | 4.00 / 4.05 / 4.05 ms |
| Remote 10 ms / 50 ms | 10 / 10.05 / 10.05 ms and 50 / 50.05 / 50.05 ms |

![Entry slippage by latency and net PnL per round trip by scenario](latency_analysis/latency_analysis.png)

**Entry cost.** Across all entry signals, the old rule paid 0.12 ticks more than the quote the strategy saw. With the order reaching the exchange the instant Databento receives the signal, it pays 0.22 ticks; at 1 ms, 0.26 ticks; at 10 ms, 0.29; at 100 ms, 0.33. Most of this is the mid price moving in the signal's direction before the order arrives: 0.17 ticks already at zero added latency, out of the roughly 0.3 ticks the signal predicts in total ([#26](https://github.com/aarbitman1228/finm-37000-final-project-group-10/issues/26)). Most of the predicted move happens before any order can reach the exchange.

**Backtests, OBI > 0.75** (five days summed; slippage in ticks per fill):

| Scenario | Round trips | Win rate | Net PnL | Net / round trip | Entry slippage | Exit slippage | Median holding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Next event (old model) | 86,707 | 0.11% | −$1,344,608 | −$15.51 | 0.11 | 0.03 | 5.4 ms |
| Zero | 87,850 | 0.15% | −$1,500,070 | −$17.08 | 0.22 | 0.05 | 5.2 ms |
| Colocated (default) | 87,628 | 0.17% | −$1,505,452 | −$17.18 | 0.23 | 0.05 | 5.3 ms |
| Chicago metro | 84,689 | 0.23% | −$1,476,426 | −$17.43 | 0.25 | 0.08 | 5.8 ms |
| New York | 69,982 | 0.42% | −$1,235,168 | −$17.65 | 0.27 | 0.10 | 18.6 ms |
| Remote 10 ms | 63,658 | 0.56% | −$1,136,222 | −$17.85 | 0.28 | 0.11 | 39.3 ms |
| Remote 50 ms | 53,449 | 0.98% | −$989,526 | −$18.51 | 0.31 | 0.14 | 126.3 ms |

**Net PnL per round trip by threshold:**

| Scenario | 0.75 | 0.85 | 0.90 | 0.95 |
| --- | ---: | ---: | ---: | ---: |
| Next event (old model) | −$15.51 | −$14.62 | −$14.14 | −$15.20 |
| Zero | −$17.08 | −$17.10 | −$17.51 | −$21.07 |
| Colocated (default) | −$17.18 | −$17.19 | −$17.79 | −$21.06 |
| Chicago metro | −$17.43 | −$17.28 | −$17.78 | −$20.83 |
| New York | −$17.65 | −$17.38 | −$17.60 | −$19.38 |
| Remote 10 ms | −$17.85 | −$17.55 | −$17.55 | −$18.73 |
| Remote 50 ms | −$18.51 | −$18.00 | −$17.94 | −$18.56 |

### Conclusions

* **The old approximation was optimistic, not conservative.** Every latency scenario loses more per round trip than the next-event model: $1.6 to $5.9 more even with zero added latency. At 0.75, this holds on each of the five days, and the per-trip loss rises with every step in latency on each day.
* **Most of the damage happens before any order can arrive.** Even a colocated strategy sees a signal a median 1.4 ms after the book event that caused it. Extra latency adds a smaller, steady cost on top.
* **Slower setups trade less but no better.** At 50 ms, the strategy makes 39% fewer round trips and holds for 126 ms instead of 5 ms. Total losses shrink only because there are fewer trades. At 0.95 the per-trip loss is largest at low latency, because the most extreme imbalances are the ones the market has most often already traded through (0.67 ticks of entry slippage). Slower setups enter fewer of them and pay less per entry (0.42–0.45 ticks), which looks like a selection effect rather than a benefit of latency.
* **None of the 140 daily runs** (7 scenarios × 4 thresholds × 5 days) made money. Under realistic latency the conclusion from [#26](https://github.com/aarbitman1228/finm-37000-final-project-group-10/issues/26) is stronger: OBI predicts the direction, but by the time an order arrives too little of the move is left to pay for crossing the spread.

**Limitations.** Order and fill-report delays cannot be observed in market data, so they are assumptions, reported over a wide range. Databento's `ts_recv` stands in for a colocated feed handler. The model covers one-lot orders that take liquidity: no queue position or market impact, and a quantity above the displayed size still fills at the touch. Delays are constant unless jitter is enabled. The sample is CL only and five in-sample days.

## How to Run the Project
1. **Clone the repository:** 
   `git clone https://github.com/aarbitman1228/finm-37000-final-project-group-10.git`
2. **Install dependencies:** 
   `pip install -r requirements.txt` (installs `databento`, `pandas`, `pyarrow`, `python-dotenv`, `matplotlib`, and `pytest`)
3. **Configure API:** 
   Create a `.env` file in the root directory and add your Databento API key: `DATABENTO_API_KEY=your_key_here`.
4. **Fetch Limit Order Data:** 
   Run `python src/fetch_mbp_data.py` from a terminal for a one-week CL sample (October 4–10, 2026; the end date is exclusive). The script checks each day's estimated size and cost, asks once for confirmation, and saves one compressed Parquet file per day in `data/raw/`. Completed daily files are reused if a run is restarted. Use `--symbol GC.c.0` or `--symbol SI.c.0` to select another contract, or `--start` and `--end` to change the date range.
5. **Calculate Order Book Imbalance:**
   Run `python src/order_book_imbalance.py` to process all Parquet files in `data/raw/`. Results are saved in `data/processed/`, with previously processed files reused when available.
6. **Generate Trading Signals:**
   Run `python src/trading_signals.py` to turn the OBI files in `data/processed/` into BUY/SELL/NEUTRAL signals at the default 0.75 threshold. Results are saved in `data/signals/`, with previously generated files reused when available. The backtester also creates any missing OBI and signal files itself.
7. **Run the Backtest Simulation:** 
   Execute `python src/backtester.py --imbalance_threshold 0.75 --take_profit_ticks 2` to run the historical simulation with the default colocated latency. Add, for example, `--feed_latency_us 290 --order_latency_us 340 --response_latency_us 340` for a Chicago-metro setup, or `--fill_model next_event` for the original approximation.
8. **Review Outcomes:** 
   The reporting module reads the backtester output (`results/trade_log.csv` and, if present, `results/equity_curve.csv`):
   `python src/reporting.py --symbol CL --imbalance_threshold 0.75`
   This writes `results_summary.txt` (net PnL, win rate, trade count, max drawdown, plus a cross-check against the engine's `realized_pnl`), `visualizations/equity_curve.png`, and `visualizations/price_and_obi.png`. Use `--trades` / `--equity` to point at other files, and run the tests with `pytest tests/`.
9. **Analyze Spread and Size:**
   Run `python src/spread_size_analysis.py --symbol CL` to measure, for every signal, how far the mid price moves in the signal's direction after 0.1, 1 and 10 seconds (`--horizons_ms`) against the cost of crossing the spread in and out plus fees, grouped by spread and by size at the touch. It writes `results/spread_size_analysis.csv` and `visualizations/spread_size_analysis.png`. Compare runs of `backtester.py` with and without `--max_spread_ticks` / `--min_top_depth` to see the effect on PnL. The edge is measured from the signal's own quote, as if orders filled instantly, so it is optimistic: with realistic latency each entry also pays about 0.22–0.33 ticks of slippage (see [Execution Latency](#execution-latency)).
10. **Evaluate Latency:**
   Run `python src/latency_analysis.py --start_date 2026-10-05 --end_date 2026-10-10` (about two minutes) to measure the feed delay, the entry cost at each latency, and backtests for every latency scenario and threshold. It writes the CSV files and chart in [`latency_analysis/`](latency_analysis/).

## Team Agreement

Each team member confirms they have reviewed this README and the open Issues, and agrees this plan accurately reflects what the team discussed.

| Name | Agreement |
|------|-----------|
| Andrew Arbitman | I agree — @aarbitman1228 |
| Rodrigo Castillo | I agree — @RoCa5tillo |
| Xiaohan Zhu | I agree — @zhucolatee |
| Jessica Xu | I agree — @Jessicakk0711 |

> To register your agreement: edit this table to add your name and GitHub handle, or leave a comment or reaction on the PR that introduced this README.
