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

The sensitivity check used the five full CL MBP-10 files for UTC dates October 5–10, 2026 (12,813,597 book events in total). The shorter October 4 file was excluded. OBI was calculated at level 0, as in the default configuration. Each daily file was backtested independently at thresholds 0.75, 0.85, 0.90, and 0.95 using the default one-contract size, two-tick take-profit and stop-loss, $2.00 per-contract-per-side fees, trading-hour/book-quality filters, and $100,000 initial cash. Each run starts with fresh cash and liquidates any remaining position at that file's end; the daily net PnL figures below are summed, not compounded. Positive round-trip PnL defines a win.

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


## Backtest Execution Engine

`src/backtester.py` steps through every MBP-10 book event in the order Databento received it and simulates trading on the signals:

* **Entry:** when flat, buy 1 contract on a BUY signal and sell 1 contract on a SELL signal.
* **Exit:** take-profit or stop-loss when the mid price has moved the set number of ticks (default 2) from the mid at entry, or when OBI crosses back through 0.
* **Execution:** a decision made on one book event fills on the next valid event, so the strategy never trades at the quote that produced its signal. Buys pay `ask_px_00` and sells receive `bid_px_00`.
* **Book quality:** only events that complete an exchange packet (Databento `F_LAST` flag) with an uncrossed book are used, so trading halts and pre-open auctions are not traded.
* **Trading hours:** no trading during the CME 16:00–17:00 CT maintenance break or the weekend; positions are flattened and entries stop 1 minute before the 16:00 CT close. Any position left at the end of the data is closed at the last quote.
* **Portfolio:** starts with $100,000. As with futures, cash changes only by realized PnL and fees; equity also includes open-position PnL marked at the bid (long) or ask (short). New entries stop once equity reaches zero.
* **Fees:** $1.50 CME exchange fee plus $0.50 routing fee per contract per fill (`--exchange_fee`, `--routing_fee`). These are approximate and configurable.

Other options include `--stop_loss_ticks`, `--quantity`, `--latency_us` (extra order delay) and `--flatten_minutes`. The trade log columns `ts_event, side, price, quantity, fee` are the input format for the reporting module.

## How to Run the Project
1. **Clone the repository:** 
   `git clone https://github.com/aarbitman1228/finm-37000-final-project-group-10.git`
2. **Install dependencies:** 
   `pip install -r requirements.txt` (installs `databento`, `pandas`, `pyarrow`, and `python-dotenv`)
3. **Configure API:** 
   Create a `.env` file in the root directory and add your Databento API key: `DATABENTO_API_KEY=your_key_here`.
4. **Fetch Limit Order Data:** 
   Run `python src/fetch_mbp_data.py` from a terminal for a one-week CL sample (October 4–10, 2026; the end date is exclusive). The script checks each day's estimated size and cost, asks once for confirmation, and saves one compressed Parquet file per day in `data/raw/`. Completed daily files are reused if a run is restarted. Use `--symbol GC.c.0` or `--symbol SI.c.0` to select another contract, or `--start` and `--end` to change the date range.
5. **Calculate Order Book Imbalance:**
   Run `python src/order_book_imbalance.py` to process all Parquet files in `data/raw/`. Results are saved in `data/processed/`, with previously processed files reused when available.
6. **Generate Trading Signals:**
   (keep the existing text from main)
7. **Run the Backtest Simulation:** 
   Execute `python src/backtester.py --imbalance_threshold 0.75 --take_profit_ticks 2` to run the historical simulation.
8. **Review Outcomes:** 
   The reporting module reads the backtester output (`results/trade_log.csv` and, if present, `results/equity_curve.csv`):
   `python src/reporting.py --symbol CL --imbalance_threshold 0.75`
   This writes `results_summary.txt` (net PnL, win rate, trade count, max drawdown, plus a cross-check against the engine's `realized_pnl`), `visualizations/equity_curve.png`, and `visualizations/price_and_obi.png`. Use `--trades` / `--equity` to point at other files, and run the tests with `pytest tests/`.

## Team Agreement

Each team member confirms they have reviewed this README and the open Issues, and agrees this plan accurately reflects what the team discussed.

| Name | Agreement |
|------|-----------|
| Andrew Arbitman | I agree — @aarbitman1228 |
| Rodrigo Castillo | I agree — @RoCa5tillo |
| Xiaohan Zhu | I agree — @zhucolatee |
| Jessica Xu | I agree — @Jessicakk0711 |

> To register your agreement: edit this table to add your name and GitHub handle, or leave a comment or reaction on the PR that introduced this README.
