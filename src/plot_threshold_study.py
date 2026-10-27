"""Chart the OBI threshold study from its checked-in CSV results.

Reads ``threshold_optimization/threshold_summary.csv`` and ``daily_results.csv``
(written by ``threshold_analysis.py``) and writes
``threshold_optimization/threshold_study.png``: the gross loss and fees behind
each threshold's net PnL, and the net PnL of every daily run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from latency_analysis import BASELINE, INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE, _style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "threshold_optimization"


def dollars(value: float) -> str:
    return f"-${abs(value):,.0f}" if value < 0 else f"${value:,.0f}"


def plot_threshold_study(summary: pd.DataFrame, daily: pd.DataFrame, output_path: Path) -> Path:
    fig, (ax_total, ax_daily) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)
    labels = [f"{t:g}" for t in summary["threshold"]]
    x = np.arange(len(summary))

    # Net PnL = gross PnL - fees; stack the two losses downward from zero.
    gross = summary["gross_pnl"].to_numpy()
    fees = -summary["fees"].to_numpy()
    ax_total.bar(x, gross, width=0.6, color=SERIES[0], label="Gross trading loss")
    ax_total.bar(x, fees, width=0.6, bottom=gross, color=SERIES[1], label="Fees")
    for xi, net, trips in zip(x, summary["net_pnl"], summary["round_trips"]):
        ax_total.annotate(f"{dollars(net)}\n{trips:,} round trips", (xi, net), xytext=(0, -6),
                          textcoords="offset points", ha="center", va="top", color=INK, fontsize=9)
    ax_total.axhline(0, color=BASELINE, linewidth=1)
    ax_total.set_ylim(summary["net_pnl"].min() * 1.25, abs(summary["net_pnl"].min()) * 0.05)
    ax_total.set_xticks(x, labels)
    ax_total.set_xlabel("OBI entry threshold", color=INK_SECONDARY)
    ax_total.set_ylabel("Net PnL, five daily runs summed", color=INK_SECONDARY)
    ax_total.yaxis.set_major_formatter(lambda v, _: dollars(v))
    ax_total.set_title("Higher thresholds lose less only by trading less", color=INK, loc="left", fontsize=12)
    ax_total.legend(loc="lower right", frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    days = sorted(daily["start_date_utc"].unique())
    day_colors = SERIES + [INK_MUTED]
    for k, day in enumerate(days):
        rows = daily[daily["start_date_utc"] == day].sort_values("threshold")
        ax_daily.plot(x, rows["net_pnl"], color=day_colors[k % len(day_colors)], linewidth=2, marker="o",
                      markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                      label=pd.Timestamp(day).strftime("%b %d"))
    ax_daily.axhline(0, color=BASELINE, linewidth=1)
    ax_daily.set_xticks(x, labels)
    ax_daily.set_xlabel("OBI entry threshold", color=INK_SECONDARY)
    ax_daily.set_ylabel("Net PnL per daily run", color=INK_SECONDARY)
    ax_daily.yaxis.set_major_formatter(lambda v, _: dollars(v))
    ax_daily.set_title("No threshold made money on any day", color=INK, loc="left", fontsize=12)
    ax_daily.legend(title="Trading date (UTC)", loc="lower right", frameon=False, labelcolor=INK_SECONDARY,
                    fontsize=9, title_fontsize=9)

    for ax in (ax_total, ax_daily):
        _style(ax)
        ax.grid(False, axis="x")

    # "\$" keeps matplotlib from reading the text between two dollar signs as math.
    fig.text(0.01, 0.01, "CL front month, 5 daily MBP-10 files (Oct 5-9, 2026), \\$100,000 per daily run "
             "(0.75 hits the equity stop every day). 2-tick take-profit / stop-loss, \\$2.00 fees per contract "
             "per side, next-event fills.",
             color=INK_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chart the OBI threshold study results.")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    summary = pd.read_csv(args.results_dir / "threshold_summary.csv").sort_values("threshold")
    daily = pd.read_csv(args.results_dir / "daily_results.csv")
    path = plot_threshold_study(summary, daily, args.results_dir / "threshold_study.png")
    print(f"Chart written to {path}")


if __name__ == "__main__":
    main()
