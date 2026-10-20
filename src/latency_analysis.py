"""Re-evaluate the OBI strategy with realistic message latency (Issue #30).

The backtester used to fill every order on the next book event Databento
received. This script measures how fast and how stale that approximation was,
and re-runs the strategy with explicit exchange-to-strategy latency:

1. Feed delay: how far Databento's receive time (``ts_recv``) lags CME's
   matching engine (``ts_event``) at tradable book events, and when the book
   that the next-event rule filled against had actually existed.
2. Latency cost curve: for every entry signal, the price an order pays if it
   reaches the exchange L microseconds after Databento received the signal,
   compared with the quote the strategy saw, and the mid move over the same
   time. Lookups use the backtester's own latency model.
3. Backtests: every daily file for each latency scenario and OBI threshold.

Outputs (``latency_analysis/``)
-------------------------------
* ``feed_latency.csv``: feed delay and next-event statistics per day and pooled.
* ``latency_cost_curve.csv``: entry slippage and mid move by latency.
* ``daily_results.csv``: one backtest per day, scenario and threshold.
* ``latency_summary.csv``: the daily backtests pooled by scenario and threshold.
* ``latency_analysis.png``: the cost curve and net PnL per round trip.
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
import threshold_analysis  # noqa: E402
import trading_signals  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "latency_analysis"
DEFAULT_THRESHOLDS = [0.75, 0.85, 0.90, 0.95]

# Databento flag for a record whose ts_recv is not a real receive time (the
# snapshot that opens each file); left out of the timing measurements.
F_BAD_TS_RECV = 8

# One-way delays in microseconds: (feed, order, fill report). A colocated
# software strategy reaches CME's matching engine through the order gateway in
# about 50 us. Other locations add their one-way network delay to every leg:
# about 290 us by fiber from 350 E Cermak (Chicago) and 4 ms by microwave from
# New Jersey. The remote rows are sensitivity cases for a strategy run far
# from the exchange. None is the original next-event approximation.
SCENARIOS = {
    "next_event": None,
    "zero": (0.0, 0.0, 0.0),
    "colocated": (0.0, 50.0, 50.0),
    "chicago_metro": (290.0, 340.0, 340.0),
    "new_york": (4_000.0, 4_050.0, 4_050.0),
    "remote_10ms": (10_000.0, 10_050.0, 10_050.0),
    "remote_50ms": (50_000.0, 50_050.0, 50_050.0),
}
SCENARIO_LABELS = {
    "next_event": "Next event\n(old model)",
    "zero": "Zero",
    "colocated": "Colocated",
    "chicago_metro": "Chicago\nmetro",
    "new_york": "New York",
    "remote_10ms": "Remote\n10 ms",
    "remote_50ms": "Remote\n50 ms",
}

COST_CURVE_LATENCIES_US = [
    0, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000,
]

# Large enough that the equity stop never ends a run early, so every scenario
# trades the same hours. Futures cash only moves by PnL, so nothing else changes.
ANALYSIS_CASH = 10_000_000.0

POOLED = "all"

COST_CURVE_COLUMNS = [
    "data_file",
    "model",
    "latency_us",
    "signals",
    "mean_slippage_ticks",
    "mean_mid_move_ticks",
    "share_same_quote",
    "share_worse_quote",
    "share_better_quote",
]

# Chart colors: surface, ink and the first categorical slots of the dataviz palette.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def load_day(path: Path) -> pd.DataFrame:
    """Read a raw daily MBP-10 file and add level-0 OBI; ``set_signals`` adds the signal."""
    market = pd.read_parquet(path, columns=threshold_analysis.MARKET_COLUMNS).reset_index()
    market["OBI"] = order_book_imbalance.get_OBI(
        market[["ts_event", "bid_sz_00", "ask_sz_00"]], [0]
    )["OBI"].to_numpy()
    return market


def set_signals(market: pd.DataFrame, threshold: float) -> pd.DataFrame:
    market["signal"] = trading_signals.generate_signals(market[["OBI"]], threshold)["signal"].to_numpy()
    return market


def _clocks(market: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    recv_ns = pd.DatetimeIndex(market["ts_recv"]).as_unit("ns").asi8
    event_ns = pd.DatetimeIndex(market["ts_event"]).as_unit("ns").asi8
    return recv_ns, event_ns


def entry_signals(market: pd.DataFrame, config: backtester.BacktestConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return (tradable records, entry-signal records).

    Entry signals are tradable records with a BUY/SELL signal outside the
    flatten window and with a genuine receive time.
    """
    tradable, closing = backtester.tradable_events(market, config.flatten_minutes)
    valid = tradable[(market["flags"].to_numpy()[tradable] & F_BAD_TS_RECV) == 0]
    signals = valid[(market["signal"].to_numpy()[valid] != 0) & ~closing[valid]]
    return tradable, signals


def feed_samples(market: pd.DataFrame, config: backtester.BacktestConfig) -> dict[str, np.ndarray]:
    """Timing samples in microseconds.

    * ``feed_delay``: ``ts_recv - ts_event`` at every tradable record.
    * ``signal_feed_delay``: the same at entry signals.
    * ``next_event_gap``: time from a signal to the next tradable record
      received, which is when the next-event approximation filled.
    * ``next_event_book_age``: how long before the decision (the signal's
      ``ts_recv``) that fill book existed at the exchange; positive means the
      order filled against a book older than the decision itself.
    """
    tradable, signals = entry_signals(market, config)
    recv_ns, event_ns = _clocks(market)
    valid = tradable[(market["flags"].to_numpy()[tradable] & F_BAD_TS_RECV) == 0]

    next_position = np.searchsorted(recv_ns[tradable], recv_ns[signals] + 1)
    has_next = next_position < len(tradable)
    filled = tradable[next_position[has_next]]
    decided = signals[has_next]
    return {
        "feed_delay": (recv_ns[valid] - event_ns[valid]) / 1_000,
        "signal_feed_delay": (recv_ns[signals] - event_ns[signals]) / 1_000,
        "next_event_gap": (recv_ns[filled] - recv_ns[decided]) / 1_000,
        "next_event_book_age": (recv_ns[decided] - event_ns[filled]) / 1_000,
    }


def summarize_feed(samples: dict[str, np.ndarray]) -> dict:
    feed, signal_feed = samples["feed_delay"], samples["signal_feed_delay"]
    gap, age = samples["next_event_gap"], samples["next_event_book_age"]
    return {
        "tradable_events": len(feed),
        "entry_signals": len(signal_feed),
        "feed_delay_p50_us": np.percentile(feed, 50),
        "feed_delay_p90_us": np.percentile(feed, 90),
        "feed_delay_p99_us": np.percentile(feed, 99),
        "signal_feed_delay_p50_us": np.percentile(signal_feed, 50),
        "signal_feed_delay_p90_us": np.percentile(signal_feed, 90),
        "next_event_gap_p10_us": np.percentile(gap, 10),
        "next_event_gap_p50_us": np.percentile(gap, 50),
        "next_event_gap_p90_us": np.percentile(gap, 90),
        "next_event_within_50_us": float((gap <= 50).mean()),
        "next_event_book_before_decision": float((age > 0).mean()),
        "next_event_book_age_p50_us": np.percentile(age, 50),
    }


def cost_curve_sums(
    market: pd.DataFrame,
    config: backtester.BacktestConfig,
    latencies_us: list[float],
) -> pd.DataFrame:
    """Slippage and mid-move totals over all entry signals, for each latency.

    ``latency_us`` is the time from Databento receiving the signal to the order
    reaching the exchange (feed plus order latency). The order executes against
    the book the backtester's latency model gives for that arrival time; the
    ``next_event`` row uses the original approximation instead. Slippage is
    the crossed price (ask for buys, bid for sells) relative to the signal's
    quote and the mid move is measured in the signal's direction, both in
    ticks; positive slippage is worse for the strategy.
    """
    tradable, signals = entry_signals(market, config)
    recv_ns, event_ns = _clocks(market)
    exchange_ns = backtester.exchange_clock(event_ns, recv_ns)
    bid = market["bid_px_00"].to_numpy(dtype=float)
    ask = market["ask_px_00"].to_numpy(dtype=float)
    mid = (bid + ask) / 2
    direction = np.sign(market["signal"].to_numpy()[signals])
    quote = np.where(direction > 0, ask[signals], bid[signals])

    def totals(model: str, latency_us: float, fill_position: np.ndarray) -> dict:
        ok = fill_position < len(tradable)
        j = tradable[fill_position[ok]]
        sign = direction[ok]
        slippage = np.round(sign * (np.where(sign > 0, ask[j], bid[j]) - quote[ok]) / config.tick_size, 6)
        return {
            "model": model,
            "latency_us": latency_us,
            "signals": int(ok.sum()),
            "slippage_ticks": float(slippage.sum()),
            "mid_move_ticks": float((sign * (mid[j] - mid[signals[ok]]) / config.tick_size).sum()),
            "same_quote": int((slippage == 0).sum()),
            "worse_quote": int((slippage > 0).sum()),
            "better_quote": int((slippage < 0).sum()),
        }

    rows = [totals("next_event", np.nan, np.searchsorted(recv_ns[tradable], recv_ns[signals] + 1))]
    for latency_us in latencies_us:
        arrival_ns = recv_ns[signals] + round(latency_us * 1_000)
        rows.append(totals("latency", latency_us, backtester.book_at(exchange_ns, tradable, arrival_ns)))
    return pd.DataFrame(rows)


def finish_cost_curve(sums: pd.DataFrame) -> pd.DataFrame:
    """Turn per-day totals (with a ``data_file`` column) into means, adding pooled rows."""
    keys = ["model", "latency_us"]
    pooled = sums.groupby(keys, sort=False, dropna=False).sum(numeric_only=True).reset_index()
    pooled["data_file"] = POOLED
    table = pd.concat([sums, pooled], ignore_index=True)
    signals = table["signals"].replace(0, np.nan)
    table["mean_slippage_ticks"] = table["slippage_ticks"] / signals
    table["mean_mid_move_ticks"] = table["mid_move_ticks"] / signals
    table["share_same_quote"] = table["same_quote"] / signals
    table["share_worse_quote"] = table["worse_quote"] / signals
    table["share_better_quote"] = table["better_quote"] / signals
    return table[COST_CURVE_COLUMNS]


def scenario_config(scenario: str, threshold: float) -> backtester.BacktestConfig:
    delays = SCENARIOS[scenario]
    if delays is None:
        return backtester.BacktestConfig(
            imbalance_threshold=threshold, initial_cash=ANALYSIS_CASH, fill_model="next_event"
        )
    feed, order, response = delays
    return backtester.BacktestConfig(
        imbalance_threshold=threshold,
        initial_cash=ANALYSIS_CASH,
        feed_latency_us=feed,
        order_latency_us=order,
        response_latency_us=response,
    )


# Trade-log columns needed to pool round-trip statistics across days.
POOLED_TRADE_COLUMNS = ["ts_event", "side", "price", "decision_price", "reason", "fee", "realized_pnl", "position_after"]


def backtest(market: pd.DataFrame, file: Path, scenario: str, threshold: float) -> tuple[dict, pd.DataFrame]:
    """Run one daily backtest; return its result row and its trade log."""
    config = scenario_config(scenario, threshold)
    result = backtester.run_backtest(market, config)
    stats = result["stats"]
    row = {
        "data_file": file.name,
        "start_date_utc": file.stem.split("_")[-2],
        "scenario": scenario,
        "threshold": threshold,
        "round_trips": stats["round_trips"],
        "wins": stats["wins"],
        "gross_pnl": float(stats["gross_pnl"]),
        "fees": float(stats["fees"]),
        "net_pnl": float(stats["net_pnl"]),
        "net_pnl_per_round_trip": float(stats["net_pnl"]) / stats["round_trips"] if stats["round_trips"] else np.nan,
        "entry_slippage_ticks": stats["entry_slippage_ticks"],
        "exit_slippage_ticks": stats["exit_slippage_ticks"],
        "median_holding_ms": stats["median_holding_ms"],
        "max_drawdown_mtm": stats["max_drawdown_mtm"],
        "entries_blocked_no_equity": stats["entries_blocked_no_equity"],
    }
    return row, result["trades"][POOLED_TRADE_COLUMNS]


def summarize_backtests(daily: pd.DataFrame, trades: dict[tuple[str, float], list[pd.DataFrame]]) -> pd.DataFrame:
    """Pool the independent daily runs by scenario and threshold (PnL is summed, not compounded)."""
    rows = []
    for (scenario, threshold), group in daily.groupby(["scenario", "threshold"], sort=False):
        # Every daily run ends flat, so the logs can be joined into one.
        stats = backtester.trade_stats(
            pd.concat(trades[(scenario, threshold)], ignore_index=True),
            backtester.CONTRACT_SPECS["CL"]["tick_size"],
        )
        round_trips = int(group["round_trips"].sum())
        rows.append(
            {
                "scenario": scenario,
                "threshold": threshold,
                "daily_files": group["data_file"].nunique(),
                "round_trips": round_trips,
                "wins": int(group["wins"].sum()),
                "win_rate": group["wins"].sum() / round_trips if round_trips else np.nan,
                "gross_pnl": group["gross_pnl"].sum(),
                "fees": group["fees"].sum(),
                "net_pnl": group["net_pnl"].sum(),
                "gross_pnl_per_round_trip": group["gross_pnl"].sum() / round_trips if round_trips else np.nan,
                "net_pnl_per_round_trip": group["net_pnl"].sum() / round_trips if round_trips else np.nan,
                "entry_slippage_ticks": stats["entry_slippage_ticks"],
                "exit_slippage_ticks": stats["exit_slippage_ticks"],
                "median_holding_ms": stats["median_holding_ms"],
                "max_daily_drawdown_mtm": group["max_drawdown_mtm"].max(),
                "losing_days": int((group["net_pnl"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0)


def plot_analysis(curve: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> Path:
    """Save the latency cost curve and net PnL per round trip by scenario."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_curve, ax_pnl) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)

    pooled = curve[curve["data_file"] == POOLED]
    latency = pooled[pooled["model"] == "latency"].dropna(subset=["mean_slippage_ticks"]).sort_values("latency_us")
    old = pooled[pooled["model"] == "next_event"].iloc[0]
    x = latency["latency_us"].to_numpy()
    for column, label, color in (
        ("mean_slippage_ticks", "Entry price vs. signal quote", SERIES[0]),
        ("mean_mid_move_ticks", "Mid move in signal direction", SERIES[1]),
    ):
        y = latency[column].to_numpy()
        ax_curve.plot(x, y, color=color, linewidth=2, marker="o", markersize=6,
                      markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, solid_capstyle="round")
        ax_curve.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                          va="center", color=INK, fontsize=9)
    ax_curve.axhline(old["mean_slippage_ticks"], color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax_curve.annotate(f"Next-event fill (old model): {old['mean_slippage_ticks']:.2f}",
                      (x[1], old["mean_slippage_ticks"]), xytext=(0, -14), textcoords="offset points",
                      color=INK_SECONDARY, fontsize=9)
    for scenario in ("colocated", "chicago_metro", "new_york", "remote_10ms", "remote_50ms"):
        feed, order, _ = SCENARIOS[scenario]
        ax_curve.axvline(feed + order, color=GRID, linewidth=1, zorder=0)
        ax_curve.annotate(SCENARIO_LABELS[scenario].replace("\n", " "), (feed + order, 1.0),
                          xycoords=("data", "axes fraction"), xytext=(3, -4), textcoords="offset points",
                          rotation=90, va="top", ha="left", color=INK_MUTED, fontsize=8)
    ax_curve.set_xscale("symlog", linthresh=10, linscale=0.5)
    ax_curve.set_xlim(0, x.max() * 1.6)
    ax_curve.set_ylim(0, max(latency["mean_slippage_ticks"].max(), old["mean_slippage_ticks"]) * 1.25)
    ax_curve.set_xticks([0, 10, 100, 1_000, 10_000, 100_000])
    ax_curve.set_xticklabels(["0", "10 us", "100 us", "1 ms", "10 ms", "100 ms"])
    ax_curve.set_xlabel("Time from Databento receiving the signal to the order reaching CME", color=INK_SECONDARY)
    ax_curve.set_ylabel("Mean per entry signal (ticks)", color=INK_SECONDARY)
    ax_curve.set_title("Entry slippage grows with latency (OBI > 0.75 signals)", color=INK, loc="left", fontsize=12)
    _style(ax_curve)
    ax_curve.legend(loc="lower right", frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    scenarios = list(dict.fromkeys(summary["scenario"]))
    positions = np.arange(len(scenarios))
    thresholds = sorted(summary["threshold"].unique())
    for k, threshold in enumerate(thresholds):
        rows = summary[summary["threshold"] == threshold].set_index("scenario").reindex(scenarios)
        y = rows["net_pnl_per_round_trip"].to_numpy()
        color = SERIES[k % len(SERIES)]
        ax_pnl.plot(positions, y, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"OBI > {threshold:g}")
        # The lines converge on the right, so label them where they are apart.
        ax_pnl.annotate(f"{threshold:g}", (positions[0], y[0]), xytext=(-9, 0), textcoords="offset points",
                        va="center", ha="right", color=INK, fontsize=9)
    values = summary["net_pnl_per_round_trip"]
    padding = (values.max() - values.min()) * 0.12
    ax_pnl.set_ylim(values.min() - padding, values.max() + padding)
    ax_pnl.yaxis.set_major_formatter(lambda value, _: f"-${-value:,.0f}" if value < 0 else f"${value:,.0f}")
    ax_pnl.set_xticks(positions)
    ax_pnl.set_xticklabels([SCENARIO_LABELS.get(name, name) for name in scenarios], fontsize=9)
    ax_pnl.set_xlim(-0.9, len(scenarios) - 0.6)
    ax_pnl.set_ylabel("Net PnL per round trip (fees included)", color=INK_SECONDARY)
    ax_pnl.set_title("Every scenario loses; the old model understated the loss", color=INK, loc="left", fontsize=12)
    _style(ax_pnl)
    ax_pnl.legend(title="Entry threshold", loc="upper right", frameon=False, labelcolor=INK_SECONDARY,
                  fontsize=9, title_fontsize=9)

    fig.text(0.01, 0.01, "CL front month, 5 daily MBP-10 files (Oct 5-9, 2026). 2-tick take-profit / stop-loss, "
             "$2.00 fees per contract per side.", color=INK_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate the OBI strategy under latency scenarios.")
    parser.add_argument("--raw_dir", type=Path, default=backtester.DEFAULT_RAW_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start_date", help="Include files whose start date is on or after YYYY-MM-DD.")
    parser.add_argument("--end_date", help="Exclude files whose start date is on or after YYYY-MM-DD.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = threshold_analysis.select_files(args.raw_dir, args.start_date, args.end_date)
    default = backtester.BacktestConfig()

    feed_rows, pooled_samples, curve_sums, daily_rows = [], [], [], []
    trades: dict[tuple[str, float], list[pd.DataFrame]] = {}
    for file in files:
        print(f"Processing {file.name}", flush=True)
        market = set_signals(load_day(file), default.imbalance_threshold)
        samples = feed_samples(market, default)
        pooled_samples.append(samples)
        feed_rows.append({"data_file": file.name, **summarize_feed(samples)})
        curve_sums.append(cost_curve_sums(market, default, COST_CURVE_LATENCIES_US).assign(data_file=file.name))

        for threshold in args.thresholds:
            set_signals(market, threshold)
            for scenario in args.scenarios:
                row, log = backtest(market, file, scenario, threshold)
                daily_rows.append(row)
                trades.setdefault((scenario, threshold), []).append(log)

    pooled = {key: np.concatenate([s[key] for s in pooled_samples]) for key in pooled_samples[0]}
    feed = pd.DataFrame(feed_rows + [{"data_file": POOLED, **summarize_feed(pooled)}])
    curve = finish_cost_curve(pd.concat(curve_sums, ignore_index=True))
    daily = pd.DataFrame(daily_rows)
    summary = summarize_backtests(daily, trades)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feed_latency.csv": feed,
        "latency_cost_curve.csv": curve,
        "daily_results.csv": daily,
        "latency_summary.csv": summary,
    }
    for name, table in outputs.items():
        table.to_csv(args.output_dir / name, index=False, float_format="%.6f")
    chart = plot_analysis(curve, summary, args.output_dir / "latency_analysis.png")

    columns = ["scenario", "threshold", "round_trips", "win_rate", "net_pnl", "net_pnl_per_round_trip",
               "entry_slippage_ticks", "exit_slippage_ticks", "median_holding_ms"]
    print(summary[columns].to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"Results written to {args.output_dir} (chart: {chart.name})")


if __name__ == "__main__":
    main()
