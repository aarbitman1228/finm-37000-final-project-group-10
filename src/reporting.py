"""Performance metrics and visualizations for the OBI backtester (Issue #18).

Inputs
------
Trade log (``results/trade_log.csv``, one row per fill; written by backtester.py, Issue #17):
    ts_event : timestamp of the fill
    side     : "BUY" / "SELL" (case-insensitive) or +1 / -1
    price    : execution price
    quantity : number of contracts, positive (optional, defaults to 1)
    fee      : total exchange + routing fee for the fill (optional, defaults to 0)
    realized_pnl : optional, engine PnL gross of fees; used only as a cross-check

Equity curve (``results/equity_curve.csv``, optional):
    ts_recv, equity : mark-to-market equity; when present it drives the equity
                      chart, final equity and max drawdown.

Market data (used only for plotting):
    ts_event, OBI, and either ``price`` or both ``bid_px_00`` and ``ask_px_00``
    (the mid price is used when ``price`` is absent).

Outputs
-------
* ``results_summary.txt`` with net PnL, win rate, trade count and max drawdown.
* ``equity_curve.png``: cumulative equity over the backtest.
* ``price_and_obi.png``: zoomed price and OBI subplots with trade markers.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_TRADES_PATH = ROOT / "results" / "trade_log.csv"
DEFAULT_SUMMARY_PATH = ROOT / "results_summary.txt"
DEFAULT_VISUALIZATION_DIR = ROOT / "visualizations"

DEFAULT_INITIAL_CASH = 100_000.0
DEFAULT_WINDOW_MINUTES = 30
MAX_PLOT_POINTS = 20_000

# Dollar value of a 1.00 price move for one contract.
CONTRACT_MULTIPLIERS = {"CL": 1_000.0, "GC": 100.0, "SI": 5_000.0}

ROUND_TRIP_COLUMNS = [
    "entry_time",
    "exit_time",
    "direction",
    "quantity",
    "entry_price",
    "exit_price",
    "gross_pnl",
    "fees",
    "net_pnl",
]


@dataclass
class _Lot:
    """An open slice of position waiting to be closed (FIFO)."""

    sign: int
    quantity: float
    price: float
    fee_per_unit: float
    timestamp: pd.Timestamp


def normalize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Validate a trade log and convert it to a standard signed-quantity format."""
    missing = {"ts_event", "side", "price"} - set(trades.columns)
    if missing:
        raise ValueError(f"Trade log is missing required columns: {sorted(missing)}")

    out = pd.DataFrame(index=trades.index)
    # ISO8601: CSV timestamps on a whole second are written without a fraction.
    out["ts_event"] = pd.to_datetime(trades["ts_event"], utc=True, format="ISO8601")
    out["price"] = trades["price"].astype(float)

    side = trades["side"]
    if pd.api.types.is_numeric_dtype(side):
        sign = np.sign(side.astype(float))
    else:
        mapped = side.astype(str).str.strip().str.upper().map({"BUY": 1.0, "SELL": -1.0})
        sign = mapped
    if sign.isna().any() or (sign == 0).any():
        raise ValueError("Trade side must be BUY/SELL or a non-zero +1/-1 value.")

    quantity = (
        trades["quantity"].astype(float).abs()
        if "quantity" in trades.columns
        else pd.Series(1.0, index=trades.index)
    )
    if (quantity <= 0).any():
        raise ValueError("Trade quantity must be greater than zero.")

    out["signed_qty"] = sign * quantity
    out["fee"] = (
        trades["fee"].astype(float).abs()
        if "fee" in trades.columns
        else pd.Series(0.0, index=trades.index)
    )

    return out.sort_values("ts_event", kind="stable").reset_index(drop=True)


def build_round_trips(
    trades: pd.DataFrame,
    multiplier: float = 1.0,
) -> pd.DataFrame:
    """Match fills into closed round trips using FIFO and compute PnL for each.

    Fees of both the opening and closing fill are charged to the round trip.
    Position still open at the end of the log is not included (no exit yet).
    """
    fills = normalize_trades(trades)
    lots: deque[_Lot] = deque()
    records: list[dict] = []

    for fill in fills.itertuples(index=False):
        sign = 1 if fill.signed_qty > 0 else -1
        remaining = abs(fill.signed_qty)
        fee_per_unit = fill.fee / remaining

        # Close existing lots of the opposite direction first.
        while remaining > 1e-12 and lots and lots[0].sign != sign:
            lot = lots[0]
            matched = min(remaining, lot.quantity)
            gross = lot.sign * (fill.price - lot.price) * matched * multiplier
            fees = matched * (lot.fee_per_unit + fee_per_unit)
            records.append(
                {
                    "entry_time": lot.timestamp,
                    "exit_time": fill.ts_event,
                    "direction": "LONG" if lot.sign > 0 else "SHORT",
                    "quantity": matched,
                    "entry_price": lot.price,
                    "exit_price": fill.price,
                    "gross_pnl": gross,
                    "fees": fees,
                    "net_pnl": gross - fees,
                }
            )
            lot.quantity -= matched
            remaining -= matched
            if lot.quantity <= 1e-12:
                lots.popleft()

        # Any leftover quantity opens (or adds to) a position.
        if remaining > 1e-12:
            lots.append(_Lot(sign, remaining, fill.price, fee_per_unit, fill.ts_event))

    round_trips = pd.DataFrame(records, columns=ROUND_TRIP_COLUMNS)
    round_trips.attrs["open_position"] = float(sum(lot.sign * lot.quantity for lot in lots))
    return round_trips


def build_equity_curve(
    round_trips: pd.DataFrame,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    start_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return realized equity over time, stepping at each round-trip exit."""
    if round_trips.empty:
        times = [start_time] if start_time is not None else []
        return pd.DataFrame({"ts_event": times, "equity": [initial_cash] * len(times)})

    ordered = round_trips.sort_values("exit_time", kind="stable")
    equity = initial_cash + ordered["net_pnl"].cumsum().to_numpy()

    first_time = (
        start_time if start_time is not None else ordered["entry_time"].min()
    )
    return pd.DataFrame(
        {
            "ts_event": [first_time] + list(ordered["exit_time"]),
            "equity": np.concatenate([[initial_cash], equity]),
        }
    )


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Return (largest peak-to-trough drop in dollars, same drop as a fraction of the peak)."""
    if equity.empty:
        return 0.0, 0.0
    running_peak = equity.cummax()
    drawdown = running_peak - equity
    idx = drawdown.idxmax()
    peak = running_peak.loc[idx]
    dollars = float(drawdown.loc[idx])
    fraction = float(dollars / peak) if peak > 0 else 0.0
    return dollars, fraction


def normalize_equity_curve(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Standardize an engine equity curve (``ts_recv`` or ``ts_event`` plus ``equity``)."""
    time_col = "ts_recv" if "ts_recv" in equity_curve.columns else "ts_event"
    if time_col not in equity_curve.columns or "equity" not in equity_curve.columns:
        raise ValueError("Equity curve needs a 'ts_recv' (or 'ts_event') and an 'equity' column.")

    out = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(equity_curve[time_col], utc=True, format="ISO8601"),
            "equity": equity_curve["equity"].astype(float),
        }
    )
    return out.sort_values("ts_event", kind="stable").reset_index(drop=True)


def compute_metrics(
    trades: pd.DataFrame,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    multiplier: float = 1.0,
    equity_curve: pd.DataFrame | None = None,
) -> dict:
    """Compute summary performance metrics from a raw trade log.

    When the engine's mark-to-market ``equity_curve`` is supplied it is used for
    the equity chart, final equity and max drawdown. Otherwise a realized-PnL
    curve is rebuilt from the round trips.

    If the trade log contains the engine's ``realized_pnl`` column (gross of
    fees), it is summed and reported next to the recomputed PnL as a check.
    """
    fills = normalize_trades(trades)
    round_trips = build_round_trips(trades, multiplier=multiplier)

    if equity_curve is not None and len(equity_curve):
        equity_curve = normalize_equity_curve(equity_curve)
        equity_source = "engine mark-to-market"
    else:
        equity_curve = build_equity_curve(
            round_trips,
            initial_cash=initial_cash,
            start_time=fills["ts_event"].min() if not fills.empty else None,
        )
        equity_source = "realized PnL from trade log"

    n_trips = len(round_trips)
    n_wins = int((round_trips["net_pnl"] > 0).sum()) if n_trips else 0
    drawdown, drawdown_pct = max_drawdown(equity_curve["equity"])

    engine_gross_pnl = None
    if "realized_pnl" in trades.columns and len(trades):
        engine_gross_pnl = float(trades["realized_pnl"].astype(float).sum())

    return {
        "equity_source": equity_source,
        "engine_gross_pnl": engine_gross_pnl,
        "initial_cash": float(initial_cash),
        "final_equity": float(equity_curve["equity"].iloc[-1]) if len(equity_curve) else float(initial_cash),
        "total_net_pnl": float(round_trips["net_pnl"].sum()) if n_trips else 0.0,
        "total_gross_pnl": float(round_trips["gross_pnl"].sum()) if n_trips else 0.0,
        "total_fees": float(round_trips["fees"].sum()) if n_trips else 0.0,
        "total_fills": int(len(fills)),
        "total_trades": int(n_trips),
        "winning_trades": n_wins,
        "losing_trades": n_trips - n_wins,
        "win_rate": (n_wins / n_trips) if n_trips else None,
        "max_drawdown": drawdown,
        "max_drawdown_pct": drawdown_pct,
        "open_position": round_trips.attrs.get("open_position", 0.0),
        "round_trips": round_trips,
        "equity_curve": equity_curve,
    }


def format_summary(metrics: dict, symbol: str | None = None) -> str:
    """Render the metrics dictionary as human-readable text."""
    win_rate = metrics["win_rate"]
    win_rate_text = "N/A (no completed trades)" if win_rate is None else f"{win_rate:.2%}"
    title = "Backtest Results Summary" + (f" - {symbol}" if symbol else "")

    lines = [
        title,
        "=" * len(title),
        f"Initial cash:            ${metrics['initial_cash']:,.2f}",
        f"Final equity:            ${metrics['final_equity']:,.2f}",
        f"Total net PnL:           ${metrics['total_net_pnl']:,.2f}",
        f"Total gross PnL:         ${metrics['total_gross_pnl']:,.2f}",
        f"Total fees:              ${metrics['total_fees']:,.2f}",
        f"Total trades:            {metrics['total_trades']} (completed round trips)",
        f"Total fills:             {metrics['total_fills']}",
        f"Winning / losing:        {metrics['winning_trades']} / {metrics['losing_trades']}",
        f"Win rate:                {win_rate_text}",
        f"Max drawdown:            ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']:.2%})",
        f"Open position at end:    {metrics['open_position']:g} contracts",
        f"Equity curve source:     {metrics['equity_source']}",
    ]
    engine_gross = metrics.get("engine_gross_pnl")
    if engine_gross is not None:
        diff = metrics["total_gross_pnl"] - engine_gross
        lines.append(
            f"Engine gross PnL check:  ${engine_gross:,.2f} (difference ${diff:,.2f})"
        )
    return "\n".join(lines) + "\n"


def write_summary(metrics: dict, path: Path, symbol: str | None = None) -> Path:
    """Write ``results_summary.txt``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_summary(metrics, symbol), encoding="utf-8")
    return path


def plot_equity_curve(
    equity_curve: pd.DataFrame,
    output_path: Path,
    symbol: str | None = None,
) -> Path:
    """Save the cumulative equity curve as a PNG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if len(equity_curve) > 1:
        ax.step(equity_curve["ts_event"], equity_curve["equity"], where="post", color="tab:blue")
    else:
        ax.axhline(
            equity_curve["equity"].iloc[0] if len(equity_curve) else DEFAULT_INITIAL_CASH,
            color="tab:blue",
        )
        ax.text(0.5, 0.5, "No completed trades", transform=ax.transAxes, ha="center")

    ax.set_title("Cumulative Equity Curve" + (f" - {symbol}" if symbol else ""))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equity ($)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _market_price(market: pd.DataFrame) -> pd.Series:
    """Return a price series from ``price`` or the best bid/ask mid."""
    if "price" in market.columns:
        return market["price"].astype(float)
    if {"bid_px_00", "ask_px_00"} <= set(market.columns):
        return (market["bid_px_00"].astype(float) + market["ask_px_00"].astype(float)) / 2
    raise ValueError("Market data needs a 'price' column or 'bid_px_00' and 'ask_px_00'.")


def select_window(
    market_times: pd.Series,
    fill_times: pd.Series,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pick the time window containing the most fills (or the data start if no fills)."""
    width = pd.Timedelta(minutes=window_minutes)
    if fill_times.empty:
        start = market_times.min()
        return start, start + width

    times = pd.DatetimeIndex(fill_times).sort_values()
    ends = times.searchsorted(times + width, side="right")
    best = int(np.argmax(ends - np.arange(len(times))))
    start = times[best]
    return start, start + width


def plot_price_and_obi(
    market: pd.DataFrame,
    trades: pd.DataFrame,
    output_path: Path,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
    obi_threshold: float | None = None,
    symbol: str | None = None,
) -> Path:
    """Save a zoomed price chart with the OBI score below and trade markers."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "OBI" not in market.columns or "ts_event" not in market.columns:
        raise ValueError("Market data needs 'ts_event' and 'OBI' columns.")

    data = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(market["ts_event"], utc=True),
            "price": _market_price(market).to_numpy(),
            "OBI": market["OBI"].astype(float).to_numpy(),
        }
    )
    fills = normalize_trades(trades) if len(trades) else pd.DataFrame(
        columns=["ts_event", "price", "signed_qty", "fee"]
    )

    if window_start is None or window_end is None:
        window_start, window_end = select_window(
            data["ts_event"], fills["ts_event"], window_minutes
        )
    window_start = pd.to_datetime(window_start, utc=True)
    window_end = pd.to_datetime(window_end, utc=True)

    in_window = data[(data["ts_event"] >= window_start) & (data["ts_event"] <= window_end)]
    if len(in_window) > MAX_PLOT_POINTS:
        step = int(np.ceil(len(in_window) / MAX_PLOT_POINTS))
        in_window = in_window.iloc[::step]
    window_fills = fills[(fills["ts_event"] >= window_start) & (fills["ts_event"] <= window_end)]

    fig, (ax_price, ax_obi) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]}
    )

    ax_price.plot(in_window["ts_event"], in_window["price"], color="black", linewidth=1, label="Price")
    buys = window_fills[window_fills["signed_qty"] > 0]
    sells = window_fills[window_fills["signed_qty"] < 0]
    ax_price.scatter(buys["ts_event"], buys["price"], marker="^", color="green", s=70, zorder=3, label="Buy")
    ax_price.scatter(sells["ts_event"], sells["price"], marker="v", color="red", s=70, zorder=3, label="Sell")
    ax_price.set_ylabel("Price")
    ax_price.set_title("Price and Order Book Imbalance" + (f" - {symbol}" if symbol else ""))
    ax_price.legend(loc="upper left")
    ax_price.grid(alpha=0.3)

    ax_obi.plot(in_window["ts_event"], in_window["OBI"], color="tab:purple", linewidth=0.8)
    ax_obi.axhline(0.0, color="gray", linewidth=0.8)
    if obi_threshold is not None:
        ax_obi.axhline(obi_threshold, color="green", linestyle="--", linewidth=0.8)
        ax_obi.axhline(-obi_threshold, color="red", linestyle="--", linewidth=0.8)
    ax_obi.set_ylim(-1.05, 1.05)
    ax_obi.set_ylabel("OBI")
    ax_obi.set_xlabel("Time (UTC)")
    ax_obi.grid(alpha=0.3)
    ax_obi.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def generate_report(
    trades: pd.DataFrame,
    market: pd.DataFrame | None = None,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    visualization_dir: Path = DEFAULT_VISUALIZATION_DIR,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    multiplier: float = 1.0,
    symbol: str | None = None,
    obi_threshold: float | None = None,
    window_minutes: float = DEFAULT_WINDOW_MINUTES,
    equity_curve: pd.DataFrame | None = None,
) -> dict:
    """Compute metrics and write the summary file and both charts."""
    metrics = compute_metrics(
        trades,
        initial_cash=initial_cash,
        multiplier=multiplier,
        equity_curve=equity_curve,
    )

    write_summary(metrics, summary_path, symbol)
    visualization_dir = Path(visualization_dir)
    plot_equity_curve(
        metrics["equity_curve"], visualization_dir / "equity_curve.png", symbol
    )
    if market is not None:
        plot_price_and_obi(
            market,
            trades,
            visualization_dir / "price_and_obi.png",
            window_minutes=window_minutes,
            obi_threshold=obi_threshold,
            symbol=symbol,
        )
    return metrics


def load_trades(path: Path) -> pd.DataFrame:
    """Load a trade log from a Parquet or CSV file."""
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_market_data(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Join raw top-of-book prices with the cached OBI files (same row order)."""
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    obi_files = sorted(
        path
        for path in processed_dir.glob("*_OBI_levels_*.parquet")
        if "_signals_" not in path.name
    )
    if not obi_files:
        raise FileNotFoundError(f"No processed OBI files found in {processed_dir}")

    frames = []
    for obi_path in obi_files:
        raw_path = raw_dir / f"{obi_path.stem.split('_OBI_levels_')[0]}.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file for {obi_path.name} not found: {raw_path}")

        prices = pd.read_parquet(raw_path, columns=["ts_event", "bid_px_00", "ask_px_00"])
        obi = pd.read_parquet(obi_path, columns=["OBI"])
        if len(prices) != len(obi):
            raise ValueError(f"Row count mismatch between {raw_path.name} and {obi_path.name}")

        frame = prices.reset_index(drop=True)
        frame["OBI"] = obi["OBI"].to_numpy()
        frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
        if start is not None:
            frame = frame[frame["ts_event"] >= pd.to_datetime(start, utc=True)]
        if end is not None:
            frame = frame[frame["ts_event"] <= pd.to_datetime(end, utc=True)]
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate performance metrics and charts from a backtest trade log."
    )
    parser.add_argument("--trades", type=Path, default=DEFAULT_TRADES_PATH,
                        help="Trade log written by backtester.py (.csv or .parquet).")
    parser.add_argument("--equity", type=Path, default=None,
                        help="Engine equity curve; defaults to equity_curve.csv next to the trade log.")
    parser.add_argument("--symbol", choices=sorted(CONTRACT_MULTIPLIERS), default="CL")
    parser.add_argument("--initial_cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--summary_path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_VISUALIZATION_DIR)
    parser.add_argument("--imbalance_threshold", type=float, default=None,
                        help="Draw +/- this OBI threshold on the OBI subplot.")
    parser.add_argument("--window_minutes", type=float, default=DEFAULT_WINDOW_MINUTES,
                        help="Width of the zoomed price/OBI chart.")
    parser.add_argument("--skip_market_plot", action="store_true",
                        help="Skip the price/OBI chart (no market data needed).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trades = load_trades(args.trades)

    equity_path = args.equity or args.trades.with_name("equity_curve.csv")
    equity_curve = load_trades(equity_path) if equity_path.exists() else None

    market = None
    if not args.skip_market_plot:
        fills = normalize_trades(trades)
        pad = pd.Timedelta(minutes=args.window_minutes)
        market = load_market_data(
            args.raw_dir,
            args.processed_dir,
            start=fills["ts_event"].min() - pad if len(fills) else None,
            end=fills["ts_event"].max() + pad if len(fills) else None,
        )

    metrics = generate_report(
        trades,
        market,
        summary_path=args.summary_path,
        visualization_dir=args.output_dir,
        initial_cash=args.initial_cash,
        multiplier=CONTRACT_MULTIPLIERS[args.symbol],
        symbol=args.symbol,
        obi_threshold=args.imbalance_threshold,
        window_minutes=args.window_minutes,
        equity_curve=equity_curve,
    )
    print(format_summary(metrics, args.symbol))
    print(f"Summary written to {args.summary_path}")
    print(f"Charts written to {args.output_dir}")


if __name__ == "__main__":
    main()
