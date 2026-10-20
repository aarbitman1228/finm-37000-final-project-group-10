import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtester import BacktestConfig, Portfolio, book_at, exchange_clock, run_backtest, session_masks  # noqa: E402

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
    """Backtest rows spaced ``step_ms`` apart.

    Unless overridden, an order takes one event spacing to reach the exchange
    and its fill report is instant, so every order fills on the next event.
    """
    config.setdefault("order_latency_us", step_ms * 1_000)
    config.setdefault("response_latency_us", 0.0)
    return run_backtest(make_market(rows, start, step_ms), BacktestConfig(**config))


def at(ms):
    return START + pd.Timedelta(microseconds=round(ms * 1_000))


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
    # The order reached the exchange at 1 ms and waited for the valid book at 3 ms.
    assert result["trades"]["ts_event"].iloc[0] == at(3)
    assert result["stats"]["tradable_events"] == 2


def test_order_fills_against_book_when_it_reaches_exchange():
    result = run(
        [(100.00, 100.01, 0.9, 1), (100.00, 100.02, 0.9, 1), (100.00, 100.03, 0.9, 1)],
        order_latency_us=1_500,
        response_latency_us=200,
    )
    entry = result["trades"].iloc[0]
    # Arrives at 1.5 ms: the book is still the one from 1 ms, not the next event at 2 ms.
    assert entry["price"] == 100.02
    assert entry["decision_price"] == 100.01
    assert entry["ts_sent"] == at(0)
    assert entry["ts_event"] == at(1.5)
    assert entry["ts_recv"] == at(1.7)


def test_zero_latency_fills_at_decision_quote():
    result = run(
        [(100.00, 100.01, 0.9, 1), (100.00, 100.02, 0.9, 1), (100.00, 100.03, 0.9, 1)],
        order_latency_us=0,
    )
    assert result["trades"]["price"].iloc[0] == 100.01
    assert result["stats"]["entry_slippage_ticks"] == pytest.approx(0.0)


def test_order_meets_exchange_book_the_strategy_has_not_received():
    market = make_market([
        (100.00, 100.01, 0.9, 1),   # happened at 0.0 ms, received at 0.5 ms: signal
        (100.00, 100.02, 0.9, 1),   # happened at 0.1 ms, received at 0.6 ms
        (100.00, 100.03, 0.9, 1),   # happened at 0.3 ms, received at 0.8 ms
        (100.00, 100.04, 0.9, 1),   # happened at 0.9 ms, received at 1.4 ms
    ], step_ms=0.1)
    market["ts_recv"] = [at(0.5), at(0.6), at(0.8), at(1.4)]
    market["ts_event"] = [at(0.0), at(0.1), at(0.3), at(0.9)]

    # Sent at 0.5 ms, arrives at 0.7 ms: the exchange book is the 0.3 ms one,
    # which the strategy only receives at 0.8 ms.
    latency = run_backtest(market, BacktestConfig(order_latency_us=200, response_latency_us=0))
    assert latency["trades"]["price"].iloc[0] == 100.03

    # The next-event approximation fills on the 0.1 ms book, which existed
    # before the decision was made.
    next_event = run_backtest(market, BacktestConfig(fill_model="next_event"))
    first = next_event["trades"].iloc[0]
    assert first["price"] == 100.02
    assert first["ts_event"] < first["ts_sent"]


def test_strategy_waits_for_fill_report_then_uses_newest_book():
    result = run(
        [
            (100.00, 100.01, 0.9, 1),    # 0 ms: entry, fills at 0.1 ms; report at 2.6 ms
            (100.00, 100.01, -0.1, 0),   # 1 ms: exit condition, but no fill report yet
            (100.00, 100.01, -0.1, 0),   # 2 ms: newest book when the report arrives
            (99.98, 99.99, -0.1, 0),     # 3 ms
            (99.98, 99.99, -0.1, 0),
        ],
        order_latency_us=100,
        response_latency_us=2_500,
    )
    entry, exit_ = result["trades"].iloc[0], result["trades"].iloc[1]
    assert entry["ts_recv"] == at(2.6)
    assert exit_["reason"] == "obi_neutral"
    assert exit_["decision_ts_recv"] == at(2)
    assert exit_["ts_sent"] == at(2.6)
    assert exit_["ts_event"] == at(2.7)
    assert exit_["price"] == 100.00


def test_feed_latency_delays_what_the_strategy_sees():
    rows = [(100.00, 100.01, 0.9, 1), (100.00, 100.02, 0.9, 1), (100.00, 100.03, 0.9, 1)]
    result = run(rows, feed_latency_us=1_000, order_latency_us=0)
    entry = result["trades"].iloc[0]
    # The 0 ms signal is seen at 1 ms, when the exchange book is already the 1 ms one.
    assert entry["ts_sent"] == at(1)
    assert entry["decision_ts_recv"] == at(0)
    assert entry["price"] == 100.02


def test_fill_report_does_not_reevaluate_a_book_already_acted_on():
    market = make_market([
        (100.00, 100.01, 0.9, 1),   # happened at 0.0 ms, received at 0.5 ms: entry
        (100.02, 100.03, 0.9, 1),   # happened at 0.1 ms, received at 5 ms: the order fills here
        (100.02, 100.03, 0.9, 1),
    ])
    market["ts_recv"] = [at(0.5), at(5), at(6.5)]
    market["ts_event"] = [at(0.0), at(0.1), at(6)]
    result = run_backtest(market, BacktestConfig(order_latency_us=200, response_latency_us=100))

    # The report arrives at 0.8 ms, before the 0.1 ms book is received. Judging the
    # fill against the old 0 ms book would show a 2-tick loss and trigger the stop.
    assert list(result["trades"]["reason"]) == ["entry_long", "end_of_data"]


def test_fill_report_without_new_book_waits_for_next_event():
    result = run(
        [
            (100.00, 100.01, 0.9, 1),    # 0 ms: entry fills at once; report at 1 ms
            (100.00, 100.01, -0.1, 0),   # 10 ms: first new book -> exit
            (100.00, 100.01, -0.1, 0),
        ],
        step_ms=10,
        order_latency_us=0,
        response_latency_us=1_000,
    )
    exit_ = result["trades"].iloc[1]
    assert exit_["reason"] == "obi_neutral"
    assert exit_["ts_sent"] == at(10)


def test_order_arriving_after_data_ends_is_not_filled():
    result = run(
        [(100.00, 100.01, 0.0, 0), (100.00, 100.01, 0.0, 0), (100.00, 100.01, 0.9, 1)],
        order_latency_us=50,
    )
    assert result["trades"].empty


def test_fill_is_kept_when_its_report_arrives_after_data_ends():
    result = run(
        [(100.00, 100.01, 0.9, 1), (100.01, 100.02, 0.9, 1)],
        order_latency_us=0,
        response_latency_us=10_000,
    )
    trades = result["trades"]
    assert list(trades["reason"]) == ["entry_long", "end_of_data"]
    assert list(trades["price"]) == [100.01, 100.01]
    assert result["portfolio"].position == 0


def churning_market(n=200):
    """OBI alternates between a buy signal and neutral, so the strategy trades all the time."""
    rows = [(100.00, 100.01, 0.9, 1) if k % 2 == 0 else (100.00, 100.01, -0.1, 0) for k in range(n)]
    return make_market(rows, step_ms=0.2)


def test_latency_jitter_is_seeded_and_only_adds_delay():
    config = dict(order_latency_us=100, response_latency_us=100, latency_jitter_us=300)
    first = run_backtest(churning_market(), BacktestConfig(seed=7, **config))["trades"]
    again = run_backtest(churning_market(), BacktestConfig(seed=7, **config))["trades"]
    other = run_backtest(churning_market(), BacktestConfig(seed=8, **config))["trades"]

    assert len(first) > 10
    assert first.equals(again)
    assert not first.equals(other)
    filled = first[first["reason"] != "end_of_data"]
    assert ((filled["ts_event"] - filled["ts_sent"]) >= pd.Timedelta(microseconds=100)).all()
    assert ((filled["ts_recv"] - filled["ts_event"]) >= pd.Timedelta(microseconds=100)).all()
    assert (filled["ts_event"] - filled["ts_sent"]).nunique() > 1


def test_one_order_at_a_time():
    trades = run_backtest(churning_market(), BacktestConfig(order_latency_us=300, response_latency_us=300))["trades"]
    filled = trades[trades["reason"] != "end_of_data"]
    # Each order is sent only after the previous fill report arrived.
    assert (filled["ts_sent"].iloc[1:].to_numpy() >= filled["ts_recv"].iloc[:-1].to_numpy()).all()
    assert trades["ts_event"].is_monotonic_increasing


def test_next_event_model_ignores_latency_settings():
    rows = [(100.00, 100.01, 0.9, 1), (100.00, 100.02, 0.9, 1), (100.00, 100.03, 0.9, 1)]
    result = run(rows, fill_model="next_event", order_latency_us=1_500, feed_latency_us=500)
    entry = result["trades"].iloc[0]
    assert entry["price"] == 100.02
    assert entry["ts_event"] == entry["ts_recv"] == at(1)


def test_slippage_and_holding_time_stats():
    result = run([
        (100.00, 100.01, 0.9, 1),   # decide to buy at 100.01
        (100.01, 100.02, 0.9, 1),   # fill at 100.02: 1 tick of slippage
        (100.03, 100.04, 0.9, 1),   # decide to sell at 100.03
        (100.03, 100.04, 0.9, 1),   # fill at 100.03: no slippage
    ])
    stats = result["stats"]
    assert stats["entry_slippage_ticks"] == pytest.approx(1.0)
    assert stats["exit_slippage_ticks"] == pytest.approx(0.0)
    assert stats["median_holding_ms"] == pytest.approx(2.0)


def test_exchange_clock_and_book_lookup():
    recv = np.array([10, 20, 30, 40])
    event = np.array([5, 4, 25, 50])     # out of order, and one after its receive time
    clock = exchange_clock(event, recv)
    assert clock.tolist() == [5, 5, 25, 40]

    tradable = np.array([0, 2, 3])       # record 1 is not tradable
    assert book_at(clock, tradable, 7).item() == 1     # last record at 7 is 1 -> next tradable is 2
    assert book_at(clock, tradable, 25).item() == 1    # record 2 itself
    assert book_at(clock, tradable, 41).item() == 3    # after the last record: no book
    assert book_at(clock, tradable, np.array([5, 30])).tolist() == [1, 1]


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
    with pytest.raises(ValueError):
        BacktestConfig(fill_model="instant")
    with pytest.raises(ValueError):
        BacktestConfig(order_latency_us=-1)
    with pytest.raises(ValueError):
        BacktestConfig(latency_jitter_us=-5)
