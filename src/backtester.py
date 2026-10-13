"""Event-driven execution engine for the OBI micro-momentum strategy.

The engine steps through MBP-10 book events in exchange-receive order and
simulates trading on the signals produced by ``trading_signals.py``:

* Entry: go long on signal +1 and short on signal -1 when flat.
* Exit: take-profit or stop-loss measured in ticks of mid-price movement
  since entry, or the OBI crossing back through 0.
* New entries stop once account equity is at or below zero.
* Execution: a decision made on one book event is filled on the next valid
  event (no fill at the quote that generated the signal). Buys pay
  ``ask_px_00`` and sells receive ``bid_px_00``.
* Only events that complete an exchange packet (``F_LAST`` flag) with an
  uncrossed book are used, so halts and pre-open auctions are not traded.
* No trading during the CME daily maintenance break or the weekend; open
  positions are flattened shortly before the 16:00 CT close.
* Every fill pays a per-contract exchange fee and routing fee.

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
    latency_us: float = 0.0
    flatten_minutes: float = 1.0

    def __post_init__(self) -> None:
        if self.symbol not in CONTRACT_SPECS:
            raise ValueError(f"Unsupported symbol {self.symbol!r}; use one of {sorted(CONTRACT_SPECS)}")
        if not 0 < self.imbalance_threshold <= 1:
            raise ValueError("imbalance_threshold must be between 0 and 1.")
        if self.take_profit_ticks <= 0 or self.stop_loss_ticks <= 0:
            raise ValueError("take_profit_ticks and stop_loss_ticks must be positive.")
        if self.quantity < 1:
            raise ValueError("quantity must be at least 1 contract.")
        if min(self.exchange_fee, self.routing_fee, self.latency_us, self.flatten_minutes) < 0:
            raise ValueError("Fees, latency and flatten_minutes cannot be negative.")

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


def run_backtest(market: pd.DataFrame, config: BacktestConfig) -> dict:
    """Simulate the strategy over book events.

    ``market`` needs ``ts_recv, ts_event, bid_px_00, ask_px_00, bid_sz_00,
    ask_sz_00, flags, OBI, signal`` in receive-time order.
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
    ts_recv = market["ts_recv"]
    recv_ns = pd.DatetimeIndex(ts_recv).as_unit("ns").asi8
    ts_event = market["ts_event"].reset_index(drop=True)

    valid_book = (
        ((market["flags"].to_numpy() & F_LAST) > 0)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & (bid < ask)
        & (market["bid_sz_00"].to_numpy() > 0)
        & (market["ask_sz_00"].to_numpy() > 0)
    )
    in_session, closing = session_masks(ts_recv, config.flatten_minutes)
    tradable = np.flatnonzero(valid_book & in_session)

    mid = (bid + ask) / 2
    portfolio = Portfolio(config.initial_cash, config.multiplier, config.fee_per_contract)
    tp_distance = config.take_profit_ticks * config.tick_size
    sl_distance = config.stop_loss_ticks * config.tick_size
    # Float tolerance so prices exactly on the target tick level trigger.
    eps = config.tick_size * 1e-6
    latency_ns = int(config.latency_us * 1_000)

    pending: dict | None = None
    entry_mid = 0.0
    equity_points: list[tuple] = []
    last_minute = None
    counts = {"entries_cancelled": 0, "entries_blocked_no_equity": 0}

    def record_equity(i: int) -> None:
        equity_points.append(
            (ts_recv.iat[i], portfolio.position, portfolio.cash, portfolio.equity(bid[i], ask[i]))
        )

    for i in tradable:
        if pending is not None and recv_ns[i] >= pending["earliest_ns"]:
            if pending["is_entry"] and closing[i]:
                counts["entries_cancelled"] += 1
            else:
                side = pending["side"]
                portfolio.fill(
                    side,
                    pending["quantity"],
                    ask[i] if side > 0 else bid[i],
                    ts_event=ts_event.iat[i],
                    ts_recv=ts_recv.iat[i],
                    reason=pending["reason"],
                    bid_px_00=bid[i],
                    ask_px_00=ask[i],
                    OBI=obi[i],
                    decision_ts_recv=pending["decision_ts_recv"],
                )
                if pending["is_entry"]:
                    entry_mid = mid[i]
                record_equity(i)
            pending = None

        minute = recv_ns[i] // 60_000_000_000
        if minute != last_minute:
            record_equity(i)
            last_minute = minute

        if pending is not None:
            continue

        position = portfolio.position
        order = None
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
                reason = None
            if reason is not None:
                order = (-direction, abs(position), reason)
        elif signal[i] != 0 and not closing[i]:
            if portfolio.equity(bid[i], ask[i]) <= 0:
                counts["entries_blocked_no_equity"] += 1
            else:
                order = (int(signal[i]), config.quantity, "entry_long" if signal[i] > 0 else "entry_short")

        if order is not None:
            pending = {
                "side": order[0],
                "quantity": order[1],
                "reason": order[2],
                "is_entry": position == 0,
                "earliest_ns": recv_ns[i] + max(latency_ns, 1),
                "decision_ts_recv": ts_recv.iat[i],
            }

    if len(tradable):
        last = tradable[-1]
        if portfolio.position != 0:
            side = -int(np.sign(portfolio.position))
            portfolio.fill(
                side,
                abs(portfolio.position),
                ask[last] if side > 0 else bid[last],
                ts_event=ts_event.iat[last],
                ts_recv=ts_recv.iat[last],
                reason="end_of_data",
                bid_px_00=bid[last],
                ask_px_00=ask[last],
                OBI=obi[last],
                decision_ts_recv=ts_recv.iat[last],
            )
        record_equity(last)

    trades = pd.DataFrame(portfolio.fills, columns=TRADE_LOG_COLUMNS)
    equity_curve = pd.DataFrame(equity_points, columns=["ts_recv", "position", "cash", "equity"])
    equity_curve = equity_curve.drop_duplicates("ts_recv", keep="last").reset_index(drop=True)

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "portfolio": portfolio,
        "stats": summarize(trades, equity_curve, portfolio, config, len(market), len(tradable), counts),
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
    # Fills alternate entry, exit, so each exit's entry fee is on the previous row.
    is_exit = trades["position_after"] == 0
    exits = trades[is_exit]
    entry_fees = trades["fee"].shift(1)[is_exit]
    round_trip_pnl = exits["realized_pnl"] - exits["fee"] - entry_fees
    equity = equity_curve["equity"]
    drawdown = (equity.cummax() - equity).max() if len(equity) else 0.0
    return {
        "events": n_events,
        "tradable_events": n_tradable,
        "fills": len(trades),
        "round_trips": len(exits),
        "wins": int((round_trip_pnl > 0).sum()),
        "exit_reasons": exits["reason"].value_counts().to_dict() if len(exits) else {},
        "entries_cancelled": counts["entries_cancelled"],
        "entries_blocked_no_equity": counts["entries_blocked_no_equity"],
        "gross_pnl": portfolio.realized_pnl,
        "fees": portfolio.fees_paid,
        "net_pnl": portfolio.cash - portfolio.initial_cash,
        "final_cash": portfolio.cash,
        "max_drawdown_mtm": float(drawdown),
    }


def format_stats(stats: dict, config: BacktestConfig) -> str:
    win_rate = stats["wins"] / stats["round_trips"] if stats["round_trips"] else float("nan")
    lines = [
        f"Backtest - {config.symbol}, OBI threshold {config.imbalance_threshold}, "
        f"TP {config.take_profit_ticks:g} / SL {config.stop_loss_ticks:g} ticks, "
        f"{config.quantity} contract(s), fees ${config.fee_per_contract:.2f}/contract/side",
        f"Book events:        {stats['events']:,} ({stats['tradable_events']:,} tradable)",
        f"Fills:              {stats['fills']:,}",
        f"Round trips:        {stats['round_trips']:,} (win rate {win_rate:.2%})",
        f"Exit reasons:       {stats['exit_reasons']}",
        f"Entries cancelled at close: {stats['entries_cancelled']}",
        f"Entry signals skipped (equity <= 0): {stats['entries_blocked_no_equity']:,}",
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
    parser.add_argument("--latency_us", type=float, default=0.0,
                        help="Extra delay before an order can fill; it always fills on a later event.")
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
        latency_us=args.latency_us,
        flatten_minutes=args.flatten_minutes,
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
