# High-Frequency Order Book Imbalance (OBI) Backtester

## Project Goal
The goal of this project is to develop an event-driven backtesting engine in Python that executes a micro-momentum trading strategy based on Order Book Imbalance (OBI) in commodity futures. 

Instead of relying on lagging indicators from executed trades, this project utilizes Level 2 order book depth to predict short-term price movements. By analyzing the ratio of limit buy orders (bids) to limit sell orders (asks) resting in the book, the algorithm identifies severe liquidity imbalances. When the imbalance crosses a critical threshold, the strategy executes a momentum trade to capture a small tick-level profit before the book neutralizes. 

## Data Source
*   **Provider:** Databento
*   **Schema:** Market by Price (MBP-10) - Top 10 levels of the order book.
*   **Market:** CME Group (NYMEX / COMEX via Globex)
*   **Target Assets:** Crude Oil (CL), Gold (GC), and Silver (SI). *Note: The architecture is modular, allowing the user to select which specific commodity to backtest.*

## How to Run the Project
1. **Clone the repository:** 
   `git clone https://github.com/aarbitman1228/finm-37000-final-project-group-10.git`
2. **Install dependencies:** 
   `pip install -r requirements.txt` (installs `databento`, `pandas`, `pyarrow`, and `python-dotenv`)
3. **Configure API:** 
   Create a `.env` file in the root directory and add your Databento API key: `DATABENTO_API_KEY=your_key_here`.
4. **Fetch Limit Order Data:** 
   Run `python src/fetch_mbp_data.py` from a terminal for a one-week CL sample (October 4–10, 2026; the end date is exclusive). The script checks each day's estimated size and cost, asks once for confirmation, and saves one compressed Parquet file per day in `data/raw/`. Completed daily files are reused if a run is restarted. Use `--symbol GC.c.0` or `--symbol SI.c.0` to select another contract, or `--start` and `--end` to change the date range.
5. **Run the Backtest Simulation:** 
   Execute `python src/backtester.py --imbalance_threshold 0.75 --take_profit_ticks 2` to run the historical simulation.
6. **Review Outcomes:** 
   The engine will generate a `results_summary.txt` detailing total trades, win rate, and net PnL, along with a `visualizations/` folder containing charts of the order book imbalance overlaid with execution markers.

## Team Agreement

Each team member confirms they have reviewed this README and the open Issues, and agrees this plan accurately reflects what the team discussed.

| Name | Agreement |
|------|-----------|
| Andrew Arbitman | I agree — @aarbitman1228 |
| Rodrigo Castillo | I agree — @RoCa5tillo |
| Xiaohan Zhu | I agree — @zhucolatee |
| Jessica Xu | I agree — @Jessicakk0711 |

> To register your agreement: edit this table to add your name and GitHub handle, or leave a comment or reaction on the PR that introduced this README.
