import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import latency_analysis as la  # noqa: E402
from backtester import BacktestConfig, trade_stats  # noqa: E402
from test_backtester import churning_market, make_market  # noqa: E402

CONFIG = BacktestConfig()


def rising_market(feed_delay_ms, n=20):
    """Ask rises 1 tick per event, events 1 ms apart, every event a BUY signal.

    Each event happens at the exchange ``feed_delay_ms`` before Databento receives it.
    """
    rows = [(round(100 + 0.01 * k, 2), round(100.01 + 0.01 * k, 2), 0.9, 1) for k in range(n)]
    market = make_market(rows)
    market["ts_event"] = market["ts_recv"] - pd.Timedelta(microseconds=round(feed_delay_ms * 1_000))
    return market


def curve_row(curve, model, latency_us=None):
    rows = curve[(curve["model"] == model) & (curve["data_file"] == la.POOLED)]
    if latency_us is not None:
        rows = rows[rows["latency_us"] == latency_us]
    return rows.iloc[0]


def test_cost_curve_follows_the_exchange_book_at_arrival():
    # Events reach Databento 0.5 ms after they happen, 1 ms apart: an order
    # arriving L ms after the signal meets the book floor(L + 0.5) events later.
    sums = la.cost_curve_sums(rising_market(0.5), CONFIG, [0, 500, 1_500]).assign(data_file="day")
    curve = la.finish_cost_curve(sums)

    assert curve_row(curve, "latency", 0)["mean_slippage_ticks"] == pytest.approx(0.0)
    assert curve_row(curve, "latency", 500)["mean_slippage_ticks"] == pytest.approx(1.0)
    assert curve_row(curve, "latency", 1_500)["mean_slippage_ticks"] == pytest.approx(2.0)
    assert curve_row(curve, "latency", 1_500)["mean_mid_move_ticks"] == pytest.approx(2.0)
    assert curve_row(curve, "latency", 1_500)["share_worse_quote"] == pytest.approx(1.0)
    # The old rule filled on the next event received.
    assert curve_row(curve, "next_event")["mean_slippage_ticks"] == pytest.approx(1.0)
    # Orders arriving after the last record's exchange time (18.5 ms) are dropped:
    # the book there is unknown. Even at zero latency, the last signal (received
    # at 19 ms) arrives too late.
    assert curve_row(curve, "latency", 0)["signals"] == 19
    assert curve_row(curve, "latency", 1_500)["signals"] == 18


def test_cost_curve_pools_days_by_signal_count():
    day_one = la.cost_curve_sums(rising_market(0.5, n=10), CONFIG, [500]).assign(data_file="one")
    day_two = la.cost_curve_sums(rising_market(1.5, n=30), CONFIG, [500]).assign(data_file="two")
    curve = la.finish_cost_curve(pd.concat([day_one, day_two], ignore_index=True))

    one = curve[(curve["data_file"] == "one") & (curve["model"] == "latency")].iloc[0]
    two = curve[(curve["data_file"] == "two") & (curve["model"] == "latency")].iloc[0]
    pooled = curve_row(curve, "latency", 500)
    assert pooled["signals"] == one["signals"] + two["signals"]
    expected = (
        one["mean_slippage_ticks"] * one["signals"] + two["mean_slippage_ticks"] * two["signals"]
    ) / pooled["signals"]
    assert pooled["mean_slippage_ticks"] == pytest.approx(expected)
    assert list(curve.columns) == la.COST_CURVE_COLUMNS


def test_feed_samples_measure_delay_and_stale_next_event_fills():
    fresh = la.summarize_feed(la.feed_samples(rising_market(0.5), CONFIG))
    assert fresh["feed_delay_p50_us"] == pytest.approx(500)
    assert fresh["next_event_gap_p50_us"] == pytest.approx(1_000)
    # The next book happened 0.5 ms after the decision: not stale.
    assert fresh["next_event_book_before_decision"] == 0.0

    stale = la.summarize_feed(la.feed_samples(rising_market(1.5), CONFIG))
    # With a 1.5 ms feed delay the next event received happened 0.5 ms before the decision.
    assert stale["next_event_book_before_decision"] == 1.0
    assert stale["next_event_book_age_p50_us"] == pytest.approx(500)


def test_records_with_bad_receive_time_are_not_measured():
    market = rising_market(0.5)
    market.loc[0, "flags"] = 128 | la.F_BAD_TS_RECV
    samples = la.feed_samples(market, CONFIG)
    assert len(samples["feed_delay"]) == 19
    assert len(samples["signal_feed_delay"]) == 19


def test_scenarios_build_valid_configs():
    for scenario in la.SCENARIOS:
        config = la.scenario_config(scenario, 0.9)
        assert config.imbalance_threshold == 0.9
        assert config.initial_cash == la.ANALYSIS_CASH
    assert la.scenario_config("next_event", 0.75).fill_model == "next_event"
    new_york = la.scenario_config("new_york", 0.75)
    assert (new_york.feed_latency_us, new_york.order_latency_us) == (4_000, 4_050)


def test_daily_backtests_pool_into_one_summary():
    daily_rows, trades = [], {}
    for name in ("CL_c_0_mbp-10_2026-10-05_2026-10-06.parquet", "CL_c_0_mbp-10_2026-10-06_2026-10-07.parquet"):
        market = churning_market()
        for scenario in ("next_event", "colocated"):
            row, log = la.backtest(market, Path(name), scenario, 0.75)
            daily_rows.append(row)
            trades.setdefault((scenario, 0.75), []).append(log)
    daily = pd.DataFrame(daily_rows)
    summary = la.summarize_backtests(daily, trades)

    assert set(summary["scenario"]) == {"next_event", "colocated"}
    assert daily["start_date_utc"].tolist()[:2] == ["2026-10-05", "2026-10-05"]
    for scenario, group in daily.groupby("scenario"):
        pooled = summary[summary["scenario"] == scenario].iloc[0]
        assert pooled["daily_files"] == 2
        assert pooled["round_trips"] == group["round_trips"].sum() > 0
        assert pooled["net_pnl"] == pytest.approx(group["net_pnl"].sum())
        assert pooled["net_pnl_per_round_trip"] == pytest.approx(group["net_pnl"].sum() / group["round_trips"].sum())
        joined = pd.concat(trades[(scenario, 0.75)], ignore_index=True)
        assert pooled["entry_slippage_ticks"] == pytest.approx(trade_stats(joined, 0.01)["entry_slippage_ticks"])


def test_plot_is_written(tmp_path):
    curve = la.finish_cost_curve(
        la.cost_curve_sums(rising_market(0.5), CONFIG, la.COST_CURVE_LATENCIES_US).assign(data_file="day")
    )
    # The 20 ms sample is too short for the longest latencies, which have no signals.
    assert curve["mean_slippage_ticks"].isna().any()
    rows = []
    for k, scenario in enumerate(la.SCENARIOS):
        for threshold in (0.75, 0.9):
            rows.append({"scenario": scenario, "threshold": threshold, "net_pnl_per_round_trip": -15.0 - k - threshold})
    path = la.plot_analysis(curve, pd.DataFrame(rows), tmp_path / "out" / "latency_analysis.png")

    assert path.stat().st_size > 0
