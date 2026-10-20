"""How bid-ask spread and displayed size affect the OBI strategy's edge (Issue #29).

For every BUY/SELL signal on a tradable book event this measures how far the
mid price moves in the signal's direction over a short horizon, and compares it
with the cost of crossing the spread in and out plus fees. Signals are then
grouped by the spread (in ticks) and by the size shown at the touch, showing
where the signal pays for itself, if anywhere.

Outputs
-------
* ``results/spread_size_analysis.csv``: one row per horizon, dimension and bucket.
* ``visualizations/spread_size_analysis.png``: net edge by spread and by depth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import backtester  # noqa: E402
import order_book_imbalance  # noqa: E402
import trading_signals  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_DIR = ROOT / "results"
DEFAULT_VISUALIZATION_DIR = ROOT / "visualizations"

DEFAULT_HORIZONS_MS = [100.0, 1_000.0, 10_000.0]
MAX_SPREAD_BUCKET = 3  # spreads of this many ticks or more share one bucket
DEPTH_QUANTILES = 4

TABLE_COLUMNS = [
    "ts_recv",
    "direction",
    "spread_ticks",
    "top_depth",
    "move_ticks",
    "cost_ticks",
    "net_edge_ticks",
]

SUMMARY_COLUMNS = [
    "horizon_ms",
    "dimension",
    "bucket",
    "signals",
    "mean_move_ticks",
    "mean_cost_ticks",
    "mean_net_edge_ticks",
    "share_move_above_cost",
]


def build_signal_table(
    market: pd.DataFrame,
    config: backtester.BacktestConfig,
    horizon_ms: float,
) -> pd.DataFrame:
    """Measure the forward mid move and round-trip cost after every signal.

    ``move_ticks`` is the mid change ``horizon_ms`` after the signal, in the
    signal's direction. ``cost_ticks`` is the spread paid across entry and exit
    (assuming the exit spread equals the entry spread) plus both fees, so
    ``net_edge_ticks = move_ticks - cost_ticks`` is what a trade would keep
    before any further slippage. Signals whose horizon runs past the last
    tradable event are dropped.
    """
    tradable, closing = backtester.tradable_events(market, config.flatten_minutes)
    if len(tradable) == 0:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    book = market.iloc[tradable]
    recv_ns = pd.DatetimeIndex(book["ts_recv"]).as_unit("ns").asi8
    bid = book["bid_px_00"].to_numpy(dtype=float)
    ask = book["ask_px_00"].to_numpy(dtype=float)
    mid = (bid + ask) / 2
    signal = book["signal"].to_numpy()

    horizon_ns = int(horizon_ms * 1_000_000)
    candidates = np.flatnonzero((signal != 0) & ~closing[tradable] & (recv_ns + horizon_ns <= recv_ns[-1]))
    if len(candidates) == 0:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    # Book as it stands at signal time + horizon: the last event at or before it.
    future = np.searchsorted(recv_ns, recv_ns[candidates] + horizon_ns, side="right") - 1
    direction = np.sign(signal[candidates]).astype(int)
    tick = config.tick_size

    spread_ticks = (ask[candidates] - bid[candidates]) / tick
    fee_ticks = 2 * config.fee_per_contract / (tick * config.multiplier)
    move_ticks = direction * (mid[future] - mid[candidates]) / tick
    cost_ticks = spread_ticks + fee_ticks

    return pd.DataFrame(
        {
            "ts_recv": book["ts_recv"].to_numpy()[candidates],
            "direction": direction,
            "spread_ticks": np.round(spread_ticks),
            "top_depth": (book["bid_sz_00"].to_numpy()[candidates] + book["ask_sz_00"].to_numpy()[candidates]),
            "move_ticks": move_ticks,
            "cost_ticks": cost_ticks,
            "net_edge_ticks": move_ticks - cost_ticks,
        },
        columns=TABLE_COLUMNS,
    )


def _aggregate(table: pd.DataFrame, bucket: pd.Series) -> pd.DataFrame:
    grouped = table.groupby(bucket, observed=True)
    out = pd.DataFrame(
        {
            "signals": grouped.size(),
            "mean_move_ticks": grouped["move_ticks"].mean(),
            "mean_cost_ticks": grouped["cost_ticks"].mean(),
            "mean_net_edge_ticks": grouped["net_edge_ticks"].mean(),
            "share_move_above_cost": grouped["net_edge_ticks"].apply(lambda s: float((s > 0).mean())),
        }
    )
    out.index = out.index.astype(str)
    return out.rename_axis("bucket").reset_index()


def summarize_table(table: pd.DataFrame, horizon_ms: float) -> pd.DataFrame:
    """Group a signal table by spread bucket and by touch-depth quartile."""
    if table.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    def spread_label(ticks: int) -> str:
        if ticks >= MAX_SPREAD_BUCKET:
            return f"{MAX_SPREAD_BUCKET}+ ticks"
        return "1 tick" if ticks == 1 else f"{ticks} ticks"

    spread_bucket = table["spread_ticks"].clip(lower=1, upper=MAX_SPREAD_BUCKET).astype(int).map(spread_label)
    by_spread = _aggregate(table, spread_bucket).assign(dimension="spread")

    # Displayed sizes are small integers with many ties, so split by rank into
    # equal-count quartiles and label each with the sizes it contains.
    quartile = pd.qcut(table["top_depth"].rank(method="first"), DEPTH_QUANTILES, labels=False)
    ranges = table.groupby(quartile)["top_depth"].agg(["min", "max"])
    depth_bucket = quartile.map(
        lambda q: f"Q{q + 1} (size {ranges.loc[q, 'min']:g}-{ranges.loc[q, 'max']:g})"
    )
    by_depth = _aggregate(table, depth_bucket).assign(dimension="top_depth")

    summary = pd.concat([by_spread, by_depth], ignore_index=True)
    summary["horizon_ms"] = horizon_ms
    return summary[SUMMARY_COLUMNS]


def run_analysis(
    market: pd.DataFrame,
    config: backtester.BacktestConfig,
    horizons_ms: list[float],
) -> pd.DataFrame:
    """Summarize signal edge by spread and depth for each horizon."""
    if not horizons_ms:
        raise ValueError("At least one horizon is required.")
    frames = [
        summarize_table(build_signal_table(market, config, horizon), horizon)
        for horizon in horizons_ms
    ]
    return pd.concat(frames, ignore_index=True)


def format_horizon(horizon_ms: float) -> str:
    return f"{horizon_ms / 1_000:g} s" if horizon_ms >= 1_000 else f"{horizon_ms:g} ms"


def plot_analysis(summary: pd.DataFrame, output_path: Path, symbol: str | None = None) -> Path:
    """Save net edge (move minus cost) by spread bucket and by depth quartile."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    titles = {"spread": "Net edge by bid-ask spread", "top_depth": "Net edge by size at the touch (bid + ask)"}

    for ax, dimension in zip(axes, ("spread", "top_depth")):
        subset = summary[summary["dimension"] == dimension]
        horizons = list(dict.fromkeys(subset["horizon_ms"]))
        buckets = list(dict.fromkeys(subset["bucket"]))
        width = 0.8 / max(len(horizons), 1)
        for k, horizon in enumerate(horizons):
            rows = subset[subset["horizon_ms"] == horizon].set_index("bucket").reindex(buckets)
            ax.bar(
                np.arange(len(buckets)) + k * width,
                rows["mean_net_edge_ticks"],
                width=width,
                label=format_horizon(horizon),
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(buckets)) + width * (len(horizons) - 1) / 2)
        ax.set_xticklabels(buckets, rotation=15 if dimension == "top_depth" else 0)
        ax.set_title(titles[dimension] + (f" - {symbol}" if symbol else ""))
        ax.set_ylabel("Mean (move - cost), ticks")
        ax.grid(alpha=0.3, axis="y")
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * 0.2)  # headroom so the legend clears the bars
        ax.legend(title="Horizon", loc="upper right", ncol=len(horizons))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure OBI signal edge by spread and displayed size.")
    parser.add_argument("--symbol", choices=sorted(backtester.CONTRACT_SPECS), default="CL")
    parser.add_argument("--imbalance_threshold", type=float, default=trading_signals.DEFAULT_THRESHOLD)
    parser.add_argument("--exchange_fee", type=float, default=1.50)
    parser.add_argument("--routing_fee", type=float, default=0.50)
    parser.add_argument("--horizons_ms", type=float, nargs="+", default=DEFAULT_HORIZONS_MS,
                        help="Forward-return horizons in milliseconds.")
    parser.add_argument("--flatten_minutes", type=float, default=1.0)
    parser.add_argument("--levels", type=int, nargs="+", default=order_book_imbalance.DEFAULT_LEVELS)
    parser.add_argument("--raw_dir", type=Path, default=backtester.DEFAULT_RAW_DIR)
    parser.add_argument("--processed_dir", type=Path, default=backtester.DEFAULT_PROCESSED_DIR)
    parser.add_argument("--signals_dir", type=Path, default=backtester.DEFAULT_SIGNALS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--visualization_dir", type=Path, default=DEFAULT_VISUALIZATION_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = backtester.BacktestConfig(
        symbol=args.symbol,
        imbalance_threshold=args.imbalance_threshold,
        exchange_fee=args.exchange_fee,
        routing_fee=args.routing_fee,
        flatten_minutes=args.flatten_minutes,
    )
    signal_files = backtester.ensure_signal_files(
        config.symbol, config.imbalance_threshold, args.levels,
        args.raw_dir, args.processed_dir, args.signals_dir,
    )
    market = backtester.load_market_data(signal_files, args.raw_dir)

    summary = run_analysis(market, config, args.horizons_ms)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "spread_size_analysis.csv"
    summary.to_csv(csv_path, index=False)
    chart_path = plot_analysis(summary, args.visualization_dir / "spread_size_analysis.png", config.symbol)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"Summary written to {csv_path}")
    print(f"Chart written to {chart_path}")


if __name__ == "__main__":
    main()
