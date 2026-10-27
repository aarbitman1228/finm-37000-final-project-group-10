"""Event-driven execution engine for the OBI micro-momentum strategy.

The engine steps through MBP-10 book events in exchange-receive order and
simulates trading on the signals produced by ``trading_signals.py``:

* Entry: go long on signal +1 and short on signal -1 when flat.
* Exit: take-profit or stop-loss measured in ticks of mid-price movement
  since entry, or the OBI crossing back through 0.
* New entries stop once account equity is at or below zero.
* Execution: messages between the exchange and the strategy take time (see
  below). Buys pay ``ask_px_00`` and sells receive ``bid_px_00`` of the book
  the order executes against.
* Only events that complete an exchange packet (``F_LAST`` flag) with an
  uncrossed book are used, so halts and pre-open auctions are not traded.
* No trading during the CME daily maintenance break or the weekend; open
  positions are flattened shortly before the 16:00 CT close.
* Every fill pays a per-contract exchange fee and routing fee.

Latency model (``fill_model="latency"``, the default)
-----------------------------------------------------
Every record carries two clocks: ``ts_event``, when CME's matching engine
processed the event, and ``ts_recv``, when Databento's capture server in the
CME data center received it. The strategy

1. sees a book event ``feed_latency_us`` after ``ts_recv``;
2. sends an order that reaches the matching engine ``order_latency_us`` after
   the decision. It executes against the book as it stands at that moment in
   exchange time: the last event with ``ts_event`` at or before the arrival,
   which may be one the strategy has not received yet;
3. receives the fill report ``response_latency_us`` after the execution, and
   only then acts on its new position, using the newest book it has seen.

``latency_jitter_us`` adds a random exponential delay with this mean to every
order and fill report. ``fill_model="next_event"`` keeps the original
approximation: an order fills on the next book event Databento received, and
the strategy knows about the fill at once.

Outputs a trade log (one row per fill) and a mark-to-market equity curve in
``results/``. The trade log columns ``ts_event, side, price, quantity, fee``
match the input expected by the reporting module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import order_book_imbalance
import trading_signals


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_SIGNALS_DIR = ROOT / "data" / "signals"
DEFAULT_OUTPUT_DIR = ROOT / "results"

# Dollar value of a 1.00 price move for one contract, and minimum price increment.
CONTRACT_SPECS = {
    "CL": {"multiplier": 1_000.0, "tick_size": 0.01},
    "GC": {"multiplier": 100.0, "tick_size": 0.10},
    "SI": {"multiplier": 5_000.0, "tick_size": 0.005},
}

# Databento flag marking the last record of an exchange packet; the book is
# only consistent after this record.
F_LAST = 128

EXCHANGE_TZ = "America/Chicago"
SESSION_CLOSE_MINUTE = 16 * 60
SESSION_OPEN_MINUTE = 17 * 60

FILL_MODELS = ("latency", "next_event")

TRADE_LOG_COLUMNS = [
    "ts_event",
    "ts_recv",
    "side",
    "price",
    "quantity",
    "fee",
    "reason",
    "bid_px_00",
    "ask_px_00",
    "OBI",
    "decision_ts_recv",
    "decision_price",
    "ts_sent",
    "position_after",
    "realized_pnl",
    "cash_after",
]


@dataclass
class BacktestConfig:
    symbol: str = "CL"
    imbalance_threshold: float = trading_signals.DEFAULT_THRESHOLD
    take_profit_ticks: float = 2.0
    stop_loss_ticks: float = 2.0
    quantity: int = 1
    initial_cash: float = 100_000.0
    exchange_fee: float = 1.50
    routing_fee: float = 0.50
    fill_model: str = "latency"
    # One-way delays in microseconds. The defaults are a software strategy
    # colocated with CME in Aurora, where Databento also captures the data.
    feed_latency_us: float = 0.0
    order_latency_us: float = 50.0
    response_latency_us: float = 50.0
    latency_jitter_us: float = 0.0
    seed: int = 0
    flatten_minutes: float = 1.0
    max_spread_ticks: float | None = None
    min_top_depth: int = 0

    def __post_init__(self) -> None:
        if self.symbol not in CONTRACT_SPECS:
            raise ValueError(f"Unsupported symbol {self.symbol!r}; use one of {sorted(CONTRACT_SPECS)}")
        if not 0 < self.imbalance_threshold <= 1:
            raise ValueError("imbalance_threshold must be between 0 and 1.")
        if self.take_profit_ticks <= 0 or self.stop_loss_ticks <= 0:
            raise ValueError("take_profit_ticks and stop_loss_ticks must be positive.")
        if self.quantity < 1:
            raise ValueError("quantity must be at least 1 contract.")
        if self.fill_model not in FILL_MODELS:
            raise ValueError(f"fill_model must be one of {FILL_MODELS}.")
        latencies = (self.feed_latency_us, self.order_latency_us, self.response_latency_us, self.latency_jitter_us)
        if min(self.exchange_fee, self.routing_fee, self.flatten_minutes, *latencies) < 0:
            raise ValueError("Fees, latencies and flatten_minutes cannot be negative.")
        if self.max_spread_ticks is not None and self.max_spread_ticks < 1:
            raise ValueError("max_spread_ticks must be at least 1 tick (or None to disable).")
        if self.min_top_depth < 0:
            raise ValueError("min_top_depth cannot be negative.")

    @property
    def multiplier(self) -> float:
        return CONTRACT_SPECS[self.symbol]["multiplier"]

    @property
    def tick_size(self) -> float:
        return CONTRACT_SPECS[self.symbol]["tick_size"]

    @property
    def fee_per_contract(self) -> float:
        return self.exchange_fee + self.routing_fee


class Portfolio:
    """Cash, open position and fill log for a single futures contract.

    Futures do not exchange notional, so cash moves only by realized PnL and
    fees; equity adds the unrealized PnL of the open position marked at the
    price it could be closed at (bid for longs, ask for shorts).
    """

    def __init__(self, initial_cash: float, multiplier: float, fee_per_contract: float) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.multiplier = multiplier
        self.fee_per_contract = fee_per_contract
        self.position = 0
        self.avg_price = 0.0
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.fills: list[dict] = []

    def fill(self, side: int, quantity: int, price: float, **details) -> dict:
        """Apply a fill (side +1 buy, -1 sell) and record it in the log."""
        if side not in (1, -1) or quantity <= 0:
            raise ValueError("Fill needs side +1/-1 and a positive quantity.")

        fee = quantity * self.fee_per_contract
        realized = 0.0
        signed_qty = side * quantity

        if self.position != 0 and np.sign(self.position) != side:
            closed = min(quantity, abs(self.position))
            realized = np.sign(self.position) * (price - self.avg_price) * closed * self.multiplier

        new_position = self.position + signed_qty
        if new_position == 0:
            self.avg_price = 0.0
        elif self.position == 0 or np.sign(new_position) != np.sign(self.position):
            self.avg_price = price
        elif np.sign(self.position) == side:
            self.avg_price = (
                self.avg_price * abs(self.position) + price * quantity
            ) / abs(new_position)

        self.position = new_position
        self.realized_pnl += realized
        self.fees_paid += fee
        self.cash += realized - fee

        record = {
            **details,
            "side": "BUY" if side > 0 else "SELL",
            "price": price,
            "quantity": quantity,
            "fee": fee,
            "position_after": self.position,
            "realized_pnl": realized,
            "cash_after": self.cash,
        }
        self.fills.append(record)
        return record

    def unrealized_pnl(self, bid: float, ask: float) -> float:
        if self.position > 0:
            return (bid - self.avg_price) * self.position * self.multiplier
        if self.position < 0:
            return (self.avg_price - ask) * -self.position * self.multiplier
        return 0.0

    def equity(self, bid: float, ask: float) -> float:
        return self.cash + self.unrealized_pnl(bid, ask)


def session_masks(ts_recv: pd.Series, flatten_minutes: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (in_session, closing) masks for CME Globex hours in Chicago time.

    Globex trades Sunday 17:00 CT to Friday 16:00 CT with a daily break from
    16:00 to 17:00 CT. ``closing`` marks the last ``flatten_minutes`` before
    each 16:00 CT close, when positions are flattened and entries blocked.
    """
    local = pd.DatetimeIndex(ts_recv).tz_convert(EXCHANGE_TZ)
    minute = local.hour.to_numpy() * 60 + local.minute.to_numpy() + local.second.to_numpy() / 60
    weekday = local.weekday.to_numpy()

    in_break = (minute >= SESSION_CLOSE_MINUTE) & (minute < SESSION_OPEN_MINUTE)
    weekend = (
        (weekday == 5)
        | ((weekday == 4) & (minute >= SESSION_CLOSE_MINUTE))
        | ((weekday == 6) & (minute < SESSION_OPEN_MINUTE))
    )
    in_session = ~in_break & ~weekend
    closing = (
        in_session
        & (weekday <= 4)
        & (minute >= SESSION_CLOSE_MINUTE - flatten_minutes)
        & (minute < SESSION_CLOSE_MINUTE)
    )
    return in_session, closing


def tradable_events(market: pd.DataFrame, flatten_minutes: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices of tradable book events, closing mask over all events).

    An event is tradable when it completes an exchange packet, the book is
    uncrossed with size on both sides, and it falls inside Globex hours.
    """
    bid = market["bid_px_00"].to_numpy(dtype=float)
    ask = market["ask_px_00"].to_numpy(dtype=float)
    valid_book = (
        ((market["flags"].to_numpy() & F_LAST) > 0)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & (bid < ask)
        & (market["bid_sz_00"].to_numpy() > 0)
        & (market["ask_sz_00"].to_numpy() > 0)
    )
    in_session, closing = session_masks(market["ts_recv"], flatten_minutes)
    return np.flatnonzero(valid_book & in_session), closing


def exchange_clock(event_ns: np.ndarray, recv_ns: np.ndarray) -> np.ndarray:
    """Exchange time of each record, sorted so the book can be looked up by time.

    ``ts_event`` is made non-decreasing and capped at ``ts_recv``, since an
    event cannot happen after Databento received it.
    """
    return np.minimum(np.maximum.accumulate(event_ns), recv_ns)


def book_at(exchange_ns: np.ndarray, tradable: np.ndarray, arrival_ns):
    """Position in ``tradable`` of the book an order reaching the exchange at
    ``arrival_ns`` executes against.

    That is the last record at or before the arrival or, if that book is not
    tradable (crossed, mid-packet or outside trading hours), the next tradable
    one. Returns ``len(tradable)`` when there is none or the arrival is after
    the last record, where the book is unknown. Accepts scalars or arrays.
    """
    last = np.searchsorted(exchange_ns, arrival_ns, side="right") - 1
    position = np.searchsorted(tradable, last)
    return np.where(arrival_ns > exchange_ns[-1], len(tradable), position)


def run_backtest(market: pd.DataFrame, config: BacktestConfig) -> dict:
    """Simulate the strategy over book events.

    ``market`` needs ``ts_recv, ts_event, bid_px_00, ask_px_00, bid_sz_00,
    ask_sz_00, flags, OBI, signal`` in receive-time order.

    Entries can be restricted to tight spreads (``max_spread_ticks``) and to
    books showing enough size at the touch (``min_top_depth`` contracts on
    bid plus ask). An entry also needs the side it crosses to show at least
    ``quantity`` contracts. Exits are never filtered.

    The strategy has at most one order working: after sending one it waits
    for the fill report, then decides on the newest book it has seen. Book
    events seen while waiting are not evaluated one by one.
    """
    required = {"ts_recv", "ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "flags", "OBI", "signal"}
    missing = required - set(market.columns)
    if missing:
        raise ValueError(f"Market data is missing columns: {sorted(missing)}")
    if not market["ts_recv"].is_monotonic_increasing:
        raise ValueError("Market data must be sorted by ts_recv.")

    bid = market["bid_px_00"].to_numpy(dtype=float)
    ask = market["ask_px_00"].to_numpy(dtype=float)
    obi = market["OBI"].to_numpy(dtype=float)
    signal = market["signal"].to_numpy()
    recv_ns = pd.DatetimeIndex(market["ts_recv"]).as_unit("ns").asi8
    event_ns = pd.DatetimeIndex(market["ts_event"]).as_unit("ns").asi8

    bid_sz = market["bid_sz_00"].to_numpy(dtype=float)
    ask_sz = market["ask_sz_00"].to_numpy(dtype=float)
    tradable, closing = tradable_events(market, config.flatten_minutes)

    mid = (bid + ask) / 2
    spread_ticks = (ask - bid) / config.tick_size
    portfolio = Portfolio(config.initial_cash, config.multiplier, config.fee_per_contract)
    tp_distance = config.take_profit_ticks * config.tick_size
    sl_distance = config.stop_loss_ticks * config.tick_size
    # Float tolerance so prices exactly on the target tick level trigger.
    eps = config.tick_size * 1e-6

    latency_model = config.fill_model == "latency"
    exchange_ns = exchange_clock(event_ns, recv_ns)
    # When the strategy sees each event; the next-event approximation has no delays.
    seen_ns = recv_ns + (round(config.feed_latency_us * 1_000) if latency_model else 0)
    tradable_recv_ns = recv_ns[tradable]
    rng = np.random.default_rng(config.seed)

    pending: dict | None = None
    entry_mid = 0.0
    seen = evaluated = -1
    equity_points: list[tuple] = []
    last_minute = None
    counts = {
        "entries_cancelled": 0,
        "entries_blocked_no_equity": 0,
        "entries_skipped_spread": 0,
        "entries_skipped_depth": 0,
    }

    def record_equity(ts_ns: int, i: int) -> None:
        equity_points.append((ts_ns, portfolio.position, portfolio.cash, portfolio.equity(bid[i], ask[i])))

    def delay_ns(latency_us: float) -> int:
        """One message delay, plus random jitter when configured."""
        if config.latency_jitter_us > 0:
            latency_us += rng.exponential(config.latency_jitter_us)
        return round(latency_us * 1_000)

    def route(i: int, sent_ns: int) -> tuple[int, int, int] | None:
        """(book record filled against, exchange fill time, fill report time) of an
        order decided on record ``i`` and sent at ``sent_ns``; None if it never fills."""
        if latency_model:
            arrival_ns = sent_ns + delay_ns(config.order_latency_us)
            position = int(book_at(exchange_ns, tradable, arrival_ns))
            if position == len(tradable):
                return None
            j = tradable[position]
            # If the book was not tradable on arrival, the order waits for the next valid book.
            fill_ns = max(arrival_ns, int(exchange_ns[j]))
            return j, fill_ns, fill_ns + delay_ns(config.response_latency_us)
        position = int(np.searchsorted(tradable_recv_ns, recv_ns[i] + 1))
        if position == len(tradable):
            return None
        j = tradable[position]
        return j, int(event_ns[j]), int(recv_ns[j])

    def decide(i: int, now_ns: int) -> dict | None:
        """Apply the exit or entry rules to book record ``i`` and return the order sent, if any."""
        position = portfolio.position
        if position != 0:
            # Take-profit / stop-loss compare mid to mid, so the spread paid on
            # entry does not trigger the stop on its own; fills still cross the spread.
            direction = 1 if position > 0 else -1
            move = (mid[i] - entry_mid) * direction
            if closing[i]:
                reason = "session_close"
            elif move >= tp_distance - eps:
                reason = "take_profit"
            elif move <= -sl_distance + eps:
                reason = "stop_loss"
            elif obi[i] * direction <= 0:
                reason = "obi_neutral"
            else:
                return None
            side, quantity = -direction, abs(position)
        elif signal[i] != 0 and not closing[i]:
            crossed_size = ask_sz[i] if signal[i] > 0 else bid_sz[i]
            if portfolio.equity(bid[i], ask[i]) <= 0:
                counts["entries_blocked_no_equity"] += 1
                return None
            if config.max_spread_ticks is not None and spread_ticks[i] > config.max_spread_ticks + 1e-6:
                counts["entries_skipped_spread"] += 1
                return None
            if crossed_size < config.quantity or bid_sz[i] + ask_sz[i] < config.min_top_depth:
                counts["entries_skipped_depth"] += 1
                return None
            side, quantity = int(signal[i]), config.quantity
            reason = "entry_long" if side > 0 else "entry_short"
        else:
            return None

        fill = route(i, now_ns)
        return {
            "side": side,
            "quantity": quantity,
            "reason": reason,
            "is_entry": position == 0,
            "decision": i,
            "decision_price": ask[i] if side > 0 else bid[i],
            "sent_ns": now_ns,
            "fill": fill,
            "report_ns": np.inf if fill is None else fill[2],
        }

    def complete(order: dict) -> bool:
        """Apply an executed order to the portfolio. An entry that reaches the
        exchange in the flatten window is dropped instead; returns whether it filled."""
        nonlocal entry_mid
        j, fill_ns, report_ns = order["fill"]
        if order["is_entry"] and closing[j]:
            counts["entries_cancelled"] += 1
            return False
        side = order["side"]
        portfolio.fill(
            side,
            order["quantity"],
            ask[j] if side > 0 else bid[j],
            ts_event=fill_ns,
            ts_recv=report_ns,
            reason=order["reason"],
            bid_px_00=bid[j],
            ask_px_00=ask[j],
            OBI=obi[j],
            decision_ts_recv=recv_ns[order["decision"]],
            decision_price=order["decision_price"],
            ts_sent=order["sent_ns"],
        )
        if order["is_entry"]:
            entry_mid = mid[j]
        return True

    for i in tradable:
        now_ns = seen_ns[i]
        # Fill reports that reach the strategy before it sees this event.
        while pending is not None and pending["report_ns"] <= now_ns:
            order, pending = pending, None
            j, _, report_ns = order["fill"]
            if complete(order):
                record_equity(report_ns, j)
            # React to the report using the newest book seen, unless that book was
            # already evaluated. The next-event approximation decides on the fill event.
            if latency_model and seen > evaluated:
                evaluated = seen
                pending = decide(seen, report_ns)
        seen = i

        minute = now_ns // 60_000_000_000
        if minute != last_minute:
            record_equity(now_ns, i)
            last_minute = minute

        if pending is None:
            evaluated = i
            if portfolio.position != 0 or signal[i] != 0:
                pending = decide(i, now_ns)

    if pending is not None and pending["fill"] is not None:
        # Executed at the exchange, but the fill report arrives after the data ends.
        complete(pending)

    if len(tradable):
        last = tradable[-1]
        if portfolio.position != 0:
            side = -int(np.sign(portfolio.position))
            price = ask[last] if side > 0 else bid[last]
            portfolio.fill(
                side,
                abs(portfolio.position),
                price,
                ts_event=max(int(event_ns[last]), int(portfolio.fills[-1]["ts_event"])),
                ts_recv=recv_ns[last],
                reason="end_of_data",
                bid_px_00=bid[last],
                ask_px_00=ask[last],
                OBI=obi[last],
                decision_ts_recv=recv_ns[last],
                decision_price=price,
                ts_sent=recv_ns[last],
            )
        record_equity(seen_ns[last], last)

    trades = pd.DataFrame(portfolio.fills, columns=TRADE_LOG_COLUMNS)
    for column in ("ts_event", "ts_recv", "decision_ts_recv", "ts_sent"):
        trades[column] = pd.to_datetime(trades[column], utc=True)
    equity_curve = pd.DataFrame(equity_points, columns=["ts_recv", "position", "cash", "equity"])
    equity_curve["ts_recv"] = pd.to_datetime(equity_curve["ts_recv"], utc=True)
    equity_curve = equity_curve.drop_duplicates("ts_recv", keep="last").reset_index(drop=True)

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "portfolio": portfolio,
        "stats": summarize(trades, equity_curve, portfolio, config, len(market), len(tradable), counts),
    }


def trade_stats(trades: pd.DataFrame, tick_size: float) -> dict:
    """Round-trip, slippage and holding-time statistics of a trade log.

    Slippage is the price paid relative to the quote the strategy saw when it
    decided (ask for buys, bid for sells), in ticks; positive means worse.
    """
    # Fills alternate entry, exit, so each exit's entry is on the previous row.
    is_exit = trades["position_after"] == 0
    exits = trades[is_exit]
    entries = trades.shift(1)[is_exit]
    round_trip_pnl = exits["realized_pnl"] - exits["fee"] - entries["fee"]
    holding = exits["ts_event"] - entries["ts_event"]

    direction = np.where(trades["side"] == "BUY", 1.0, -1.0)
    slippage = direction * (trades["price"] - trades["decision_price"]) / tick_size
    traded_exit = is_exit & (trades["reason"] != "end_of_data")
    return {
        "round_trips": len(exits),
        "wins": int((round_trip_pnl > 0).sum()),
        "exit_reasons": exits["reason"].value_counts().to_dict() if len(exits) else {},
        "entry_slippage_ticks": float(slippage[~is_exit].mean()) if (~is_exit).any() else float("nan"),
        "exit_slippage_ticks": float(slippage[traded_exit].mean()) if traded_exit.any() else float("nan"),
        "median_holding_ms": float(holding.median() / pd.Timedelta(milliseconds=1)) if len(exits) else float("nan"),
    }


def summarize(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    portfolio: Portfolio,
    config: BacktestConfig,
    n_events: int,
    n_tradable: int,
    counts: dict,
) -> dict:
    equity = equity_curve["equity"]
    drawdown = (equity.cummax() - equity).max() if len(equity) else 0.0
    return {
        "events": n_events,
        "tradable_events": n_tradable,
        "fills": len(trades),
        **trade_stats(trades, config.tick_size),
        "entries_cancelled": counts["entries_cancelled"],
        "entries_blocked_no_equity": counts["entries_blocked_no_equity"],
        "entries_skipped_spread": counts["entries_skipped_spread"],
        "entries_skipped_depth": counts["entries_skipped_depth"],
        "gross_pnl": portfolio.realized_pnl,
        "fees": portfolio.fees_paid,
        "net_pnl": portfolio.cash - portfolio.initial_cash,
        "final_cash": portfolio.cash,
        "max_drawdown_mtm": float(drawdown),
    }


def describe_execution(config: BacktestConfig) -> str:
    if config.fill_model == "next_event":
        return "next-event approximation (fills on the next book event received)"
    return (
        f"latency model - feed +{config.feed_latency_us:g} us, order {config.order_latency_us:g} us, "
        f"fill report {config.response_latency_us:g} us, jitter {config.latency_jitter_us:g} us"
    )


def format_stats(stats: dict, config: BacktestConfig) -> str:
    win_rate = stats["wins"] / stats["round_trips"] if stats["round_trips"] else float("nan")
    lines = [
        f"Backtest - {config.symbol}, OBI threshold {config.imbalance_threshold}, "
        f"TP {config.take_profit_ticks:g} / SL {config.stop_loss_ticks:g} ticks, "
        f"{config.quantity} contract(s), fees ${config.fee_per_contract:.2f}/contract/side",
        f"Execution:          {describe_execution(config)}",
        f"Book events:        {stats['events']:,} ({stats['tradable_events']:,} tradable)",
        f"Fills:              {stats['fills']:,}",
        f"Round trips:        {stats['round_trips']:,} (win rate {win_rate:.2%})",
        f"Exit reasons:       {stats['exit_reasons']}",
        f"Median holding:     {stats['median_holding_ms']:,.3f} ms",
        f"Slippage vs. quote at decision: entries {stats['entry_slippage_ticks']:.3f} ticks, "
        f"exits {stats['exit_slippage_ticks']:.3f} ticks",
        f"Entries cancelled at close: {stats['entries_cancelled']}",
        f"Entry signals skipped (equity <= 0): {stats['entries_blocked_no_equity']:,}",
        f"Entry signals skipped (spread > {config.max_spread_ticks} ticks): {stats['entries_skipped_spread']:,}",
        f"Entry signals skipped (thin book, depth < {config.min_top_depth} or size < {config.quantity}): "
        f"{stats['entries_skipped_depth']:,}",
        f"Gross PnL:          ${stats['gross_pnl']:,.2f}",
        f"Fees:               ${stats['fees']:,.2f}",
        f"Net PnL:            ${stats['net_pnl']:,.2f}",
        f"Final cash:         ${stats['final_cash']:,.2f}",
        f"Max drawdown (MTM): ${stats['max_drawdown_mtm']:,.2f}",
    ]
    return "\n".join(lines)


def ensure_signal_files(
    symbol: str,
    threshold: float,
    levels: list[int],
    raw_dir: Path,
    processed_dir: Path,
    signals_dir: Path,
) -> list[Path]:
    """Return signal files for the symbol, generating OBI and signals if missing."""
    levels_label = "-".join(str(level) for level in levels)
    threshold_label = str(threshold).replace(".", "_")
    pattern = f"{symbol}_c_0_*_OBI_levels_{levels_label}_signals_{threshold_label}.parquet"
    raw_files = sorted(Path(raw_dir).glob(f"{symbol}_c_0_*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No raw {symbol} Parquet files in {raw_dir}; run fetch_mbp_data.py first.")

    if len(sorted(Path(signals_dir).glob(pattern))) < len(raw_files):
        order_book_imbalance.process_parquet_files(raw_dir, processed_dir, levels)
        trading_signals.process_signal_files(processed_dir, signals_dir, threshold)
    return sorted(Path(signals_dir).glob(pattern))


def load_market_data(signal_files: list[Path], raw_dir: Path) -> pd.DataFrame:
    """Join each signal file with its raw MBP-10 file (same row order)."""
    frames = []
    for signal_path in signal_files:
        raw_path = Path(raw_dir) / f"{signal_path.stem.split('_OBI_levels_')[0]}.parquet"
        raw = pd.read_parquet(
            raw_path,
            columns=["ts_event", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "flags"],
        ).reset_index()
        signals = pd.read_parquet(signal_path, columns=["ts_event", "OBI", "signal"])
        if len(raw) != len(signals) or not (
            raw["ts_event"].to_numpy() == signals["ts_event"].to_numpy()
        ).all():
            raise ValueError(f"{signal_path.name} does not line up with {raw_path.name}")
        raw["OBI"] = signals["OBI"].to_numpy()
        raw["signal"] = signals["signal"].to_numpy()
        frames.append(raw)

    market = pd.concat(frames, ignore_index=True)
    return market.sort_values("ts_recv", kind="stable").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Event-driven OBI strategy backtest.")
    parser.add_argument("--symbol", choices=sorted(CONTRACT_SPECS), default="CL")
    parser.add_argument("--imbalance_threshold", type=float, default=trading_signals.DEFAULT_THRESHOLD)
    parser.add_argument("--take_profit_ticks", type=float, default=2.0)
    parser.add_argument("--stop_loss_ticks", type=float, default=2.0)
    parser.add_argument("--quantity", type=int, default=1, help="Contracts per trade.")
    parser.add_argument("--initial_cash", type=float, default=100_000.0)
    parser.add_argument("--exchange_fee", type=float, default=1.50,
                        help="CME exchange + clearing fee per contract per side (default: %(default)s).")
    parser.add_argument("--routing_fee", type=float, default=0.50,
                        help="Broker routing fee per contract per side (default: %(default)s).")
    parser.add_argument("--fill_model", choices=FILL_MODELS, default="latency",
                        help="'latency' models message delays; 'next_event' is the original "
                             "approximation (default: %(default)s).")
    parser.add_argument("--feed_latency_us", type=float, default=0.0,
                        help="Delay from Databento's receive time until the strategy sees an event "
                             "(default: %(default)s).")
    parser.add_argument("--order_latency_us", type=float, default=50.0,
                        help="Delay from a decision until the order reaches the matching engine "
                             "(default: %(default)s).")
    parser.add_argument("--response_latency_us", type=float, default=50.0,
                        help="Delay from a fill at the exchange until the strategy receives the report "
                             "(default: %(default)s).")
    parser.add_argument("--latency_jitter_us", type=float, default=0.0,
                        help="Mean of a random exponential delay added to every order and fill report.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the latency jitter.")
    parser.add_argument("--max_spread_ticks", type=float, default=None,
                        help="Only enter when the bid-ask spread is at most this many ticks (default: no limit).")
    parser.add_argument("--min_top_depth", type=int, default=0,
                        help="Only enter when best bid size plus best ask size is at least this many contracts.")
    parser.add_argument("--flatten_minutes", type=float, default=1.0,
                        help="Flatten and stop entering this many minutes before the 16:00 CT close.")
    parser.add_argument("--levels", type=int, nargs="+", default=order_book_imbalance.DEFAULT_LEVELS)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed_dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--signals_dir", type=Path, default=DEFAULT_SIGNALS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BacktestConfig(
        symbol=args.symbol,
        imbalance_threshold=args.imbalance_threshold,
        take_profit_ticks=args.take_profit_ticks,
        stop_loss_ticks=args.stop_loss_ticks,
        quantity=args.quantity,
        initial_cash=args.initial_cash,
        exchange_fee=args.exchange_fee,
        routing_fee=args.routing_fee,
        fill_model=args.fill_model,
        feed_latency_us=args.feed_latency_us,
        order_latency_us=args.order_latency_us,
        response_latency_us=args.response_latency_us,
        latency_jitter_us=args.latency_jitter_us,
        seed=args.seed,
        flatten_minutes=args.flatten_minutes,
        max_spread_ticks=args.max_spread_ticks,
        min_top_depth=args.min_top_depth,
    )
    signal_files = ensure_signal_files(
        config.symbol, config.imbalance_threshold, args.levels,
        args.raw_dir, args.processed_dir, args.signals_dir,
    )
    market = load_market_data(signal_files, args.raw_dir)
    result = run_backtest(market, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = args.output_dir / "trade_log.csv"
    equity_path = args.output_dir / "equity_curve.csv"
    result["trades"].to_csv(trades_path, index=False)
    result["equity_curve"].to_csv(equity_path, index=False)

    print(format_stats(result["stats"], config))
    print(f"Trade log written to {trades_path}")
    print(f"Equity curve written to {equity_path}")


if __name__ == "__main__":
    main()
