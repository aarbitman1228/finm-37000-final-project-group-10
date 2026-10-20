import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtester import BacktestConfig  # noqa: E402
from spread_size_analysis import (  # noqa: E402
    SUMMARY_COLUMNS,
    TABLE_COLUMNS,
    build_signal_table,
    plot_analysis,
    run_analysis,
    summarize_table,
)
from test_backtester import make_market  # noqa: E402

# Default CL fees: 2 * $2.00 / ($0.01 tick * 1000 multiplier) = 0.4 ticks per round trip.
FEE_TICKS = 0.4


def rising_market(n=30, spread=0.01) -> pd.DataFrame:
    """Mid rises 1 tick per millisecond; every event signals BUY."""
    rows = []
    for k in range(n):
        bid = round(100 + 0.01 * k, 2)
        rows.append((bid, round(bid + spread, 2), 0.9, 1))
    return make_market(rows)


def test_forward_move_and_cost_are_measured_in_ticks():
    table = build_signal_table(rising_market(), BacktestConfig(), horizon_ms=3)

    assert list(table.columns) == TABLE_COLUMNS
    assert table["move_ticks"].tolist() == pytest.approx([3.0] * len(table))
    assert (table["spread_ticks"] == 1).all()
    assert table["cost_ticks"].iloc[0] == pytest.approx(1 + FEE_TICKS)
    assert table["net_edge_ticks"].iloc[0] == pytest.approx(3 - 1 - FEE_TICKS)


def test_signals_without_a_full_horizon_are_dropped():
    # 30 events 1 ms apart; with a 3 ms horizon only events 0..26 have a future book.
    table = build_signal_table(rising_market(30), BacktestConfig(), horizon_ms=3)
    assert len(table) == 27


def test_move_is_signed_by_signal_direction():
    rows = []
    for k in range(20):
        bid = round(100 - 0.01 * k, 2)         # falling market, SELL signals
        rows.append((bid, round(bid + 0.01, 2), -0.9, -1))
    table = build_signal_table(make_market(rows), BacktestConfig(), horizon_ms=2)

    assert (table["direction"] == -1).all()
    assert table["move_ticks"].tolist() == pytest.approx([2.0] * len(table))


def test_wider_spread_has_a_lower_net_edge():
    tight = rising_market(spread=0.01)
    wide = rising_market(spread=0.03)
    config = BacktestConfig()

    tight_edge = build_signal_table(tight, config, 3)["net_edge_ticks"].mean()
    wide_edge = build_signal_table(wide, config, 3)["net_edge_ticks"].mean()

    assert tight_edge - wide_edge == pytest.approx(2.0)


def test_summary_groups_by_spread_and_depth():
    market = rising_market(40)
    market["ask_px_00"] = (market["bid_px_00"] + np.where(np.arange(40) % 2 == 0, 0.01, 0.02)).round(2)
    market["bid_sz_00"] = np.where(np.arange(40) < 20, 2, 10)

    summary = run_analysis(market, BacktestConfig(), [2])

    assert list(summary.columns) == SUMMARY_COLUMNS
    spread_rows = summary[summary["dimension"] == "spread"]
    assert set(spread_rows["bucket"]) == {"1 tick", "2 ticks"}
    assert spread_rows["signals"].sum() == len(build_signal_table(market, BacktestConfig(), 2))
    depth_rows = summary[summary["dimension"] == "top_depth"]
    assert len(depth_rows) == 4
    assert depth_rows["signals"].sum() == spread_rows["signals"].sum()


def test_empty_inputs():
    empty = build_signal_table(make_market([(100.00, 100.01, 0.0, 0)] * 5), BacktestConfig(), 1)
    assert empty.empty
    assert summarize_table(empty, 1).empty
    with pytest.raises(ValueError):
        run_analysis(rising_market(), BacktestConfig(), [])


def test_plot_is_written(tmp_path):
    summary = run_analysis(rising_market(40), BacktestConfig(), [1, 3])
    path = plot_analysis(summary, tmp_path / "out" / "spread_size_analysis.png", "CL")

    assert path.stat().st_size > 0
