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
   `pip install -r requirements.txt` (requires `pandas`, `numpy`, `databento`, `matplotlib`)
3. **Configure API:** 
   Create a `.env` file in the root directory and add your Databento API key: `DATABENTO_API_KEY=your_key_here`.
4. **Fetch Limit Order Data:** 
   Run `python src/fetch_mbp_data.py`. This will authenticate with Databento and download the MBP-10 order book snapshots for the specified date range, saving them locally to `data/raw/` to avoid repeated API calls.
5. **Run the Backtest Simulation:** 
   Execute `python src/backtester.py --imbalance_threshold 0.75 --take_profit_ticks 2` to run the historical simulation.
6. **Review Outcomes:** 
   The engine will generate a `results_summary.txt` detailing total trades, win rate, and net PnL, along with a `visualizations/` folder containing charts of the order book imbalance overlaid with execution markers.

## Team Agreement

Each team member confirms they have reviewed this README and the open Issues, and agrees this plan accurately reflects what the team discussed.

| Name | Agreement |
|------|-----------|
| Andrew Arbitman | I agree — @aarbitman1228 |
| Rodrigo Castillo | I agree — @ |
| Xiaohan Zhu | I agree — @ |
| Jessica Xu | I agree — @Jessicakk0711 |

> To register your agreement: edit this table to add your name and GitHub handle, or leave a comment or reaction on the PR that introduced this README.