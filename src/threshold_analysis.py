"""Run a daily OBI threshold sensitivity analysis on raw MBP-10 Parquet files."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

import backtester
import order_book_imbalance
import trading_signals


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT / "threshold_optimization"
DEFAULT_THRESHOLDS = [0.75, 0.85, 0.90, 0.95]
MARKET_COLUMNS = [
    "ts_event",
    "bid_px_00",
    "ask_px_00",
    "bid_sz_00",
    "ask_sz_00",
    "flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare backtest performance across OBI thresholds."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--start-date",
        help="Include files whose start date is on or after YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        help="Exclude files whose start date is on or after YYYY-MM-DD.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    return parser.parse_args()


def select_files(
    raw_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Path]:
    """Select daily CL files by the start date embedded in their filename."""
    if start_date is not None:
        start_date = date.fromisoformat(start_date).isoformat()
    if end_date is not None:
        end_date = date.fromisoformat(end_date).isoformat()

    files = sorted(Path(raw_dir).glob("CL_c_0_*.parquet"))
    if start_date is not None:
        files = [path for path in files if _start_date(path) >= start_date]
    if end_date is not None:
        files = [path for path in files if _start_date(path) < end_date]
    if not files:
        raise FileNotFoundError(
            f"No CL MBP-10 Parquet files found in {raw_dir} for the requested dates."
        )
    return files


def _start_date(file: Path) -> str:
    """Return a validated ISO start date from the standard raw-file name."""
    return date.fromisoformat(file.stem.split("_")[-2]).isoformat()


def analyze_file(file: Path, thresholds: list[float]) -> list[dict]:
    """Backtest one daily file independently at each requested threshold."""
    market = pd.read_parquet(file, columns=MARKET_COLUMNS).reset_index()
    market["OBI"] = order_book_imbalance.get_OBI(
        market[["ts_event", "bid_sz_00", "ask_sz_00"]],
        [0],
    )["OBI"].to_numpy()

    results = []
    for threshold in thresholds:
        signals = trading_signals.generate_signals(
            market[["OBI"]],
            threshold,
        )
        backtest_market = market.copy()
        backtest_market["signal"] = signals["signal"].to_numpy()
        outcome = backtester.run_backtest(
            backtest_market,
            backtester.BacktestConfig(imbalance_threshold=threshold),
        )
        stats = outcome["stats"]
        results.append(
            {
                "data_file": file.name,
                "start_date_utc": _start_date(file),
                "threshold": threshold,
                "book_events": stats["events"],
                "tradable_events": stats["tradable_events"],
                "round_trips": stats["round_trips"],
                "wins": stats["wins"],
                "win_rate": (
                    stats["wins"] / stats["round_trips"]
                    if stats["round_trips"]
                    else 0.0
                ),
                "gross_pnl": float(stats["gross_pnl"]),
                "fees": float(stats["fees"]),
                "net_pnl": float(stats["net_pnl"]),
                "max_drawdown_mtm": stats["max_drawdown_mtm"],
                "entries_blocked_no_equity": stats["entries_blocked_no_equity"],
            }
        )
    return results


def summarize_results(daily_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily results without compounding the independent runs."""
    summary = (
        daily_results.groupby("threshold", as_index=False)
        .agg(
            daily_files=("data_file", "nunique"),
            book_events=("book_events", "sum"),
            tradable_events=("tradable_events", "sum"),
            round_trips=("round_trips", "sum"),
            wins=("wins", "sum"),
            gross_pnl=("gross_pnl", "sum"),
            fees=("fees", "sum"),
            net_pnl=("net_pnl", "sum"),
            max_daily_drawdown_mtm=("max_drawdown_mtm", "max"),
            entries_blocked_no_equity=("entries_blocked_no_equity", "sum"),
        )
        .sort_values("threshold")
        .reset_index(drop=True)
    )
    summary["win_rate"] = summary["wins"] / summary["round_trips"]
    return summary


def main() -> None:
    args = parse_args()
    if args.start_date and args.end_date and args.start_date >= args.end_date:
        raise ValueError("--start-date must be earlier than --end-date.")
    for threshold in args.thresholds:
        backtester.BacktestConfig(imbalance_threshold=threshold)

    files = select_files(args.raw_dir, args.start_date, args.end_date)
    daily_results = pd.DataFrame(
        row
        for file in files
        for row in analyze_file(file, args.thresholds)
    )
    summary = summarize_results(daily_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily_path = args.output_dir / "daily_results.csv"
    summary_path = args.output_dir / "threshold_summary.csv"
    daily_results.to_csv(daily_path, index=False, float_format="%.6f")
    summary.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Daily results written to {daily_path}")
    print(f"Threshold summary written to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
