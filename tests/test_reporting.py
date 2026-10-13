"""Tests for src/reporting.py using a hand-built (mock) trade log.

The trade log below stands in for the output of the execution engine
(Issue #17), so these tests do not depend on that issue being finished.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import reporting  # noqa: E402


def make_trades() -> pd.DataFrame:
    """Three round trips: long win (+1.0), short loss (-0.5), long win (+0.25)."""
    times = pd.date_range("2026-10-05 14:00:00", periods=6, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "ts_event": times,
            "side": ["BUY", "SELL", "SELL", "BUY", "BUY", "SELL"],
            "price": [70.00, 71.00, 71.00, 71.50, 70.00, 70.25],
            "quantity": [1, 1, 1, 1, 1, 1],
            "fee": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        }
    )


def test_round_trips_and_pnl():
    trips = reporting.build_round_trips(make_trades(), multiplier=1000.0)

    assert len(trips) == 3
    assert list(trips["direction"]) == ["LONG", "SHORT", "LONG"]
    assert list(trips["gross_pnl"]) == pytest.approx([1000.0, -500.0, 250.0])
    # Each round trip pays 2.0 on entry and 2.0 on exit.
    assert list(trips["fees"]) == pytest.approx([4.0, 4.0, 4.0])
    assert list(trips["net_pnl"]) == pytest.approx([996.0, -504.0, 246.0])


def test_metrics():
    metrics = reporting.compute_metrics(
        make_trades(), initial_cash=100_000.0, multiplier=1000.0
    )

    assert metrics["total_trades"] == 3
    assert metrics["total_fills"] == 6
    assert metrics["winning_trades"] == 2
    assert metrics["win_rate"] == pytest.approx(2 / 3)
    assert metrics["total_net_pnl"] == pytest.approx(738.0)
    assert metrics["final_equity"] == pytest.approx(100_738.0)
    assert metrics["open_position"] == 0


def test_max_drawdown():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0, 130.0])
    dollars, fraction = reporting.max_drawdown(equity)

    assert dollars == pytest.approx(40.0)  # 120 -> 80
    assert fraction == pytest.approx(40.0 / 120.0)


def test_max_drawdown_of_metrics():
    metrics = reporting.compute_metrics(
        make_trades(), initial_cash=100_000.0, multiplier=1000.0
    )
    # Equity: 100000 -> 100996 -> 100492 -> 100738; the dip is the short loss.
    assert metrics["max_drawdown"] == pytest.approx(504.0)


def test_partial_close_and_flip():
    times = pd.date_range("2026-10-05 14:00:00", periods=3, freq="1min", tz="UTC")
    trades = pd.DataFrame(
        {
            "ts_event": times,
            "side": ["BUY", "SELL", "SELL"],
            "price": [100.0, 101.0, 102.0],
            "quantity": [2, 3, 1],
        }
    )
    trips = reporting.build_round_trips(trades, multiplier=1.0)

    # Sell 3 closes the long 2 and opens a short 1; the final sell adds to the short.
    assert len(trips) == 1
    assert trips.loc[0, "quantity"] == 2
    assert trips.loc[0, "net_pnl"] == pytest.approx(2.0)
    assert trips.attrs["open_position"] == -2


def test_no_trades():
    trades = pd.DataFrame(columns=["ts_event", "side", "price", "quantity", "fee"])
    metrics = reporting.compute_metrics(trades)

    assert metrics["total_trades"] == 0
    assert metrics["win_rate"] is None
    assert metrics["total_net_pnl"] == 0.0
    assert "N/A" in reporting.format_summary(metrics)


def test_invalid_trade_log():
    with pytest.raises(ValueError):
        reporting.normalize_trades(pd.DataFrame({"ts_event": [], "price": []}))

    bad = make_trades()
    bad.loc[0, "side"] = "HOLD"
    with pytest.raises(ValueError):
        reporting.normalize_trades(bad)


def test_numeric_side_is_accepted():
    trades = make_trades()
    trades["side"] = [1, -1, -1, 1, 1, -1]
    assert len(reporting.build_round_trips(trades)) == 3


def test_generate_report_writes_files(tmp_path):
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-10-05 13:55:00", periods=1200, freq="1s", tz="UTC")
    market = pd.DataFrame(
        {
            "ts_event": times,
            "bid_px_00": 70 + np.cumsum(rng.normal(0, 0.01, len(times))),
            "OBI": rng.uniform(-1, 1, len(times)),
        }
    )
    market["ask_px_00"] = market["bid_px_00"] + 0.01

    summary = tmp_path / "results_summary.txt"
    out_dir = tmp_path / "visualizations"
    reporting.generate_report(
        make_trades(),
        market,
        summary_path=summary,
        visualization_dir=out_dir,
        multiplier=1000.0,
        symbol="CL",
        obi_threshold=0.75,
    )

    text = summary.read_text(encoding="utf-8")
    assert "Total net PnL" in text and "Win rate" in text and "Max drawdown" in text
    assert (out_dir / "equity_curve.png").stat().st_size > 0
    assert (out_dir / "price_and_obi.png").stat().st_size > 0


def test_select_window_prefers_busiest_period():
    market_times = pd.Series(pd.date_range("2026-10-05", periods=100, freq="1min", tz="UTC"))
    fill_times = pd.Series(
        pd.to_datetime(
            ["2026-10-05 00:05", "2026-10-05 01:00", "2026-10-05 01:02", "2026-10-05 01:10"],
            utc=True,
        )
    )
    start, end = reporting.select_window(market_times, fill_times, window_minutes=15)

    assert start == pd.Timestamp("2026-10-05 01:00", tz="UTC")
    assert end == start + pd.Timedelta(minutes=15)
