import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtester import BacktestConfig, Portfolio, run_backtest, session_masks  # noqa: E402

# Tuesday 2026-10-06 10:00 CT, regular trading hours.
START = pd.Timestamp("2026-10-06 15:00:00", tz="UTC")
FEE = 2.0  # default exchange 1.50 + routing 0.50 per contract per side


def make_market(rows, start=START, step_ms=1):
    """rows: tuples (bid, ask, OBI, signal[, flags])."""
    records = []
    for k, row in enumerate(rows):
        bid, ask, obi, signal = row[:4]
        flags = row[4] if len(row) > 4 else 128
        ts = start + pd.Timedelta(milliseconds=k * step_ms)
        records.append(
            {
                "ts_recv": ts,
                "ts_event": ts,
                "bid_px_00": bid,
                "ask_px_00": ask,
                "bid_sz_00": 5,
                "ask_sz_00": 5,
                "flags": flags,
                "OBI": obi,
                "signal": signal,
            }
        )
    return pd.DataFrame(records)


def run(rows, start=START, step_ms=1, **config):
    return run_backtest(make_market(rows, start, step_ms), BacktestConfig(**config))


def test_long_take_profit_crosses_spread_and_pays_fees():
    result = run([
        (100.00, 100.01, 0.9, 1),   # signal: buy
        (100.01, 100.02, 0.9, 1),   # entry fills at ask 100.02 (mid 100.015)
        (100.03, 100.04, 0.9, 1),   # mid +2 ticks -> take profit
        (100.03, 100.04, 0.9, 1),   # exit fills at bid 100.03
    ])
    trades = result["trades"]
    assert list(trades["side"]) == ["BUY", "SELL"]
    assert list(trades["price"]) == [100.02, 100.03]
    assert list(trades["reason"]) == ["entry_long", "take_profit"]
    assert trades["fee"].tolist() == [FEE, FEE]
    assert trades["realized_pnl"].iloc[1] == pytest.approx(10.0)
    assert result["portfolio"].cash == pytest.approx(100_000 + 10.0 - 2 * FEE)
    assert result["portfolio"].position == 0
    assert result["stats"]["wins"] == 1


def test_short_stop_loss_buys_back_at_ask():
    result = run([
        (100.00, 100.01, -0.9, -1),  # signal: sell
        (100.00, 100.01, -0.9, -1),  # entry fills at bid 100.00 (mid 100.005)
        (100.02, 100.03, -0.9, -1),  # mid +2 ticks against the short -> stop
        (100.02, 100.03, -0.9, -1),  # exit fills at ask 100.03
    ])
    trades = result["trades"]
    assert list(trades["side"]) == ["SELL", "BUY"]
    assert list(trades["price"]) == [100.00, 100.03]
    assert trades["reason"].iloc[1] == "stop_loss"
    assert trades["realized_pnl"].iloc[1] == pytest.approx(-30.0)
    assert result["portfolio"].cash == pytest.approx(100_000 - 30.0 - 2 * FEE)


def test_wide_spread_alone_does_not_trigger_stop():
    result = run([
        (100.00, 100.02, 0.9, 1),
        (100.00, 100.02, 0.9, 1),   # entry at 100.02; bid is 2 ticks below entry
        (100.00, 100.02, 0.9, 1),   # unchanged mid: no exit
        (100.00, 100.02, -0.1, 0),  # OBI crosses 0 -> exit
        (100.00, 100.02, -0.1, 0),
    ])
    assert list(result["trades"]["reason"]) == ["entry_long", "obi_neutral"]


def test_obi_neutral_exit_and_reentry():
    result = run([
        (100.00, 100.01, 0.9, 1),
        (100.00, 100.01, 0.9, 1),   # long at 100.01
        (100.00, 100.01, 0.0, 0),   # OBI reaches 0 -> exit
        (100.00, 100.01, 0.8, 1),   # exit fills at 100.00; same event re-signals
        (100.00, 100.01, 0.8, 1),   # re-entry fills
    ])
    assert list(result["trades"]["reason"]) == ["entry_long", "obi_neutral", "entry_long", "end_of_data"]


def test_fills_skip_crossed_books_and_incomplete_packets():
    result = run([
        (100.00, 100.01, 0.9, 1),
        (100.02, 100.01, 0.9, 1),        # crossed book (halt): not fillable
        (100.00, 100.05, 0.9, 1, 0),     # not the last record of its packet
        (100.00, 100.02, 0.9, 1),        # entry fills here at 100.02
    ])
    assert result["trades"]["price"].iloc[0] == 100.02
    assert result["stats"]["tradable_events"] == 2


def test_latency_delays_fill():
    result = run(
        [(100.00, 100.01, 0.9, 1), (100.00, 100.02, 0.9, 1), (100.00, 100.03, 0.9, 1)],
        latency_us=1_500,
    )
    assert result["trades"]["price"].iloc[0] == 100.03


def test_flatten_before_close_and_no_trading_in_break():
    close = pd.Timestamp("2026-10-06 20:59:59.990", tz="UTC")  # 15:59:59.990 CT
    result = run(
        [
            (100.00, 100.01, 0.9, 1),   # inside flatten window: no entry
            (100.00, 100.01, 0.9, 1),
            (100.00, 100.01, 0.9, 1),   # 16:00:00.010 CT, maintenance break
        ],
        start=close,
        step_ms=10,
    )
    assert result["trades"].empty

    before = pd.Timestamp("2026-10-06 20:58:59.990", tz="UTC")  # 15:58:59.990 CT
    result = run(
        [
            (100.00, 100.01, 0.9, 1),   # entry signal before the window
            (100.00, 100.01, 0.9, 1),   # entry fills at 15:59:00.000 -> already closing
            (100.00, 100.01, 0.9, 1),
            (100.00, 100.01, 0.9, 1),
        ],
        start=before,
        step_ms=10,
    )
    assert result["stats"]["entries_cancelled"] == 1
    assert result["trades"].empty


def test_session_close_exit():
    before = pd.Timestamp("2026-10-06 20:58:59.980", tz="UTC")
    result = run(
        [
            (100.00, 100.01, 0.9, 1),
            (100.00, 100.01, 0.9, 1),   # 15:58:59.990 CT entry fills
            (100.00, 100.01, 0.9, 1),   # 15:59:00.000 CT -> flatten
            (100.00, 100.01, 0.9, 1),
        ],
        start=before,
        step_ms=10,
    )
    assert list(result["trades"]["reason"]) == ["entry_long", "session_close"]


def test_session_masks_globex_hours():
    ts = pd.Series(pd.to_datetime([
        "2026-10-04 21:59:00",  # Sun 16:59 CT  closed
        "2026-10-04 22:00:00",  # Sun 17:00 CT  open
        "2026-10-06 20:59:30",  # Tue 15:59:30 CT open, closing
        "2026-10-06 21:30:00",  # Tue 16:30 CT  break
        "2026-10-09 21:00:00",  # Fri 16:00 CT  closed
        "2026-10-10 15:00:00",  # Sat           closed
    ], utc=True))
    in_session, closing = session_masks(ts, flatten_minutes=1)
    assert in_session.tolist() == [False, True, True, False, False, False]
    assert closing.tolist() == [False, False, True, False, False, False]


def test_end_of_data_liquidates_open_position():
    result = run([(100.00, 100.01, 0.9, 1), (100.00, 100.01, 0.9, 1), (100.01, 100.02, 0.9, 1)])
    trades = result["trades"]
    assert list(trades["reason"]) == ["entry_long", "end_of_data"]
    assert trades["price"].iloc[1] == 100.01
    assert result["portfolio"].position == 0


def test_entries_stop_when_equity_exhausted():
    result = run(
        [
            (100.00, 100.01, 0.9, 1),
            (100.00, 100.01, 0.9, 1),    # entry fills; fee pushes cash below 0
            (100.00, 100.01, -0.1, 0),   # exit signal
            (100.00, 100.01, 0.9, 1),    # exit fills; new entry is blocked
            (100.00, 100.01, 0.9, 1),    # still blocked
        ],
        initial_cash=1.0,
    )
    assert list(result["trades"]["reason"]) == ["entry_long", "obi_neutral"]
    assert result["stats"]["entries_blocked_no_equity"] == 2


def test_trade_log_matches_reporting_inputs():
    result = run([(100.00, 100.01, 0.9, 1), (100.00, 100.01, 0.9, 1), (100.00, 100.01, 0.9, 1)])
    trades = result["trades"]
    assert {"ts_event", "side", "price", "quantity", "fee"} <= set(trades.columns)
    assert set(trades["side"]) <= {"BUY", "SELL"}
    assert (trades["quantity"] > 0).all()


def test_portfolio_add_flip_and_mark_to_market():
    p = Portfolio(100_000, multiplier=1_000, fee_per_contract=2.0)
    p.fill(1, 1, 100.00)
    p.fill(1, 1, 100.02)
    assert p.position == 2 and p.avg_price == pytest.approx(100.01)
    assert p.unrealized_pnl(bid=100.03, ask=100.04) == pytest.approx(40.0)
    p.fill(-1, 3, 100.05)  # close 2 (+80) and open 1 short at 100.05
    assert p.position == -1 and p.avg_price == 100.05
    assert p.realized_pnl == pytest.approx(80.0)
    assert p.fees_paid == pytest.approx(10.0)
    assert p.equity(bid=100.03, ask=100.04) == pytest.approx(100_000 + 80 - 10 + 10)


def test_spread_filter_skips_wide_entries():
    rows = [
        (100.00, 100.02, 0.9, 1),   # 2-tick spread: too wide
        (100.00, 100.02, 0.9, 1),
        (100.00, 100.02, 0.9, 1),
    ]
    filtered = run(rows, max_spread_ticks=1)
    assert filtered["trades"].empty
    assert filtered["stats"]["entries_skipped_spread"] == 3

    unfiltered = run(rows, max_spread_ticks=2)
    assert not unfiltered["trades"].empty
    assert unfiltered["stats"]["entries_skipped_spread"] == 0


def test_spread_filter_never_blocks_exits():
    result = run(
        [
            (100.00, 100.01, 0.9, 1),
            (100.00, 100.01, 0.9, 1),    # entry fills at 100.01
            (100.00, 100.03, -0.1, 0),   # spread widens and OBI crosses 0: exit anyway
            (100.00, 100.03, -0.1, 0),
        ],
        max_spread_ticks=1,
    )
    assert list(result["trades"]["reason"]) == ["entry_long", "obi_neutral"]


def test_depth_filter_skips_thin_books():
    market = make_market([(100.00, 100.01, 0.9, 1)] * 3)
    market[["bid_sz_00", "ask_sz_00"]] = 2   # 4 contracts at the touch

    thin = run_backtest(market, BacktestConfig(min_top_depth=5))
    assert thin["trades"].empty
    assert thin["stats"]["entries_skipped_depth"] == 3

    enough = run_backtest(market, BacktestConfig(min_top_depth=4))
    assert not enough["trades"].empty


def test_entry_needs_displayed_size_for_quantity():
    market = make_market([(100.00, 100.01, 0.9, 1)] * 3)
    market["ask_sz_00"] = 2     # a buy of 3 contracts cannot be filled at the touch

    result = run_backtest(market, BacktestConfig(quantity=3))
    assert result["trades"].empty
    assert result["stats"]["entries_skipped_depth"] == 3

    assert not run_backtest(market, BacktestConfig(quantity=2))["trades"].empty


def test_invalid_config():
    with pytest.raises(ValueError):
        BacktestConfig(symbol="NG")
    with pytest.raises(ValueError):
        BacktestConfig(imbalance_threshold=1.5)
    with pytest.raises(ValueError):
        BacktestConfig(max_spread_ticks=0.5)
    with pytest.raises(ValueError):
        BacktestConfig(min_top_depth=-1)
