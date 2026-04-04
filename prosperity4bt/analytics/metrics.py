"""
Metric extraction from BacktestResult.

Parses activity logs and trade history into per-product time series and
computes summary statistics used by plotting and insights.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from prosperity4bt.models import BacktestResult


@dataclass
class OwnTrade:
    timestamp: int
    price: int
    quantity: int       # always positive
    side: str           # "buy" or "sell"
    edge_vs_mid: float  # positive = favorable (bought below mid, or sold above mid)


@dataclass
class ProductMetrics:
    product: str

    # --- Time series (one entry per timestamp in order) ---
    timestamps: list[int] = field(default_factory=list)
    mid_prices: list[float] = field(default_factory=list)
    pnl_series: list[float] = field(default_factory=list)       # realized + unrealized
    inventory_series: list[int] = field(default_factory=list)   # reconstructed from trades
    best_bid_series: list[float] = field(default_factory=list)  # nan when absent
    best_ask_series: list[float] = field(default_factory=list)  # nan when absent

    # --- Own trades (sorted by timestamp) ---
    own_trades: list[OwnTrade] = field(default_factory=list)

    # --- Computed summary stats (filled by _compute_stats) ---
    final_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    n_buys: int = 0
    n_sells: int = 0
    total_buy_volume: int = 0
    total_sell_volume: int = 0
    avg_buy_price: float = 0.0
    avg_sell_price: float = 0.0
    avg_buy_edge: float = 0.0   # positive = bought below mid
    avg_sell_edge: float = 0.0  # positive = sold above mid
    final_inventory: int = 0
    max_inventory: int = 0
    min_inventory: int = 0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_round_trips: int = 0


@dataclass
class BacktestMetrics:
    run_timestamp: str
    products: dict[str, ProductMetrics] = field(default_factory=dict)

    @property
    def total_pnl(self) -> float:
        return sum(p.final_pnl for p in self.products.values())


def _safe_float(value: object) -> float:
    """Convert a column value to float; return nan for empty strings."""
    if value == "" or value is None:
        return float("nan")
    return float(value)  # type: ignore[arg-type]


def extract_metrics(result: BacktestResult, run_timestamp: str) -> BacktestMetrics:
    """Build BacktestMetrics from a BacktestResult (merged or single-day)."""
    metrics = BacktestMetrics(run_timestamp=run_timestamp)

    # --- Group activity log rows by product, preserving order ---
    product_rows: dict[str, list[list]] = defaultdict(list)
    for row in result.activity_logs:
        product_rows[row.columns[2]].append(row.columns)

    # --- Build mid-price lookup: product -> timestamp -> mid_price ---
    mid_at: dict[str, dict[int, float]] = {}
    for product, rows in product_rows.items():
        mid_at[product] = {int(cols[1]): float(cols[15]) for cols in rows}

    # --- Populate time-series for each product ---
    for product, rows in product_rows.items():
        pm = ProductMetrics(product=product)
        rows_sorted = sorted(rows, key=lambda c: int(c[1]))

        for cols in rows_sorted:
            pm.timestamps.append(int(cols[1]))
            pm.mid_prices.append(float(cols[15]))
            pm.pnl_series.append(float(cols[16]))
            pm.best_bid_series.append(_safe_float(cols[3]))
            pm.best_ask_series.append(_safe_float(cols[9]))

        metrics.products[product] = pm

    # --- Parse own trades from result.trades ---
    own_by_product: dict[str, list[OwnTrade]] = defaultdict(list)
    for trade_row in result.trades:
        t = trade_row.trade
        is_own_buy = t.buyer == "SUBMISSION"
        is_own_sell = t.seller == "SUBMISSION"
        if not is_own_buy and not is_own_sell:
            continue  # Market trade, not ours

        mid = mid_at.get(t.symbol, {}).get(t.timestamp, float("nan"))
        if is_own_buy:
            edge = (mid - t.price) if not math.isnan(mid) else float("nan")
            side = "buy"
        else:
            edge = (t.price - mid) if not math.isnan(mid) else float("nan")
            side = "sell"

        own_by_product[t.symbol].append(
            OwnTrade(
                timestamp=t.timestamp,
                price=t.price,
                quantity=t.quantity,
                side=side,
                edge_vs_mid=edge,
            )
        )

    # --- Attach trades, reconstruct inventory, compute stats ---
    for product, pm in metrics.products.items():
        trades = sorted(own_by_product.get(product, []), key=lambda x: x.timestamp)
        pm.own_trades = trades

        # Inventory delta per timestamp (multiple trades can land at same timestamp)
        inv_delta: dict[int, int] = defaultdict(int)
        for ot in trades:
            inv_delta[ot.timestamp] += ot.quantity if ot.side == "buy" else -ot.quantity

        position = 0
        for ts in pm.timestamps:
            position += inv_delta.get(ts, 0)
            pm.inventory_series.append(position)

        _compute_stats(pm)

    return metrics


def _compute_stats(pm: ProductMetrics) -> None:
    buys = [t for t in pm.own_trades if t.side == "buy"]
    sells = [t for t in pm.own_trades if t.side == "sell"]

    pm.n_buys = len(buys)
    pm.n_sells = len(sells)
    pm.total_buy_volume = sum(t.quantity for t in buys)
    pm.total_sell_volume = sum(t.quantity for t in sells)

    if pm.total_buy_volume > 0:
        pm.avg_buy_price = sum(t.price * t.quantity for t in buys) / pm.total_buy_volume
        valid_edges = [t.edge_vs_mid for t in buys if not math.isnan(t.edge_vs_mid)]
        pm.avg_buy_edge = sum(valid_edges) / len(valid_edges) if valid_edges else 0.0

    if pm.total_sell_volume > 0:
        pm.avg_sell_price = sum(t.price * t.quantity for t in sells) / pm.total_sell_volume
        valid_edges = [t.edge_vs_mid for t in sells if not math.isnan(t.edge_vs_mid)]
        pm.avg_sell_edge = sum(valid_edges) / len(valid_edges) if valid_edges else 0.0

    if pm.pnl_series:
        pm.final_pnl = pm.pnl_series[-1]

    if pm.inventory_series and pm.mid_prices:
        pm.final_inventory = pm.inventory_series[-1]
        pm.unrealized_pnl = pm.final_inventory * pm.mid_prices[-1]
        pm.realized_pnl = pm.final_pnl - pm.unrealized_pnl
    else:
        pm.final_inventory = 0
        pm.unrealized_pnl = 0.0
        pm.realized_pnl = pm.final_pnl

    if pm.inventory_series:
        pm.max_inventory = max(pm.inventory_series)
        pm.min_inventory = min(pm.inventory_series)

    # Max drawdown on PnL series
    if pm.pnl_series:
        peak = pm.pnl_series[0]
        max_dd = 0.0
        for pnl in pm.pnl_series:
            if pnl > peak:
                peak = pnl
            dd = peak - pnl
            if dd > max_dd:
                max_dd = dd
        pm.max_drawdown = max_dd

    # Round-trip win rate via FIFO queue (counted per unit volume)
    buy_queue: list[tuple[int, int]] = []  # (price, remaining_qty)
    wins_qty = 0
    losses_qty = 0
    for t in pm.own_trades:
        if t.side == "buy":
            buy_queue.append((t.price, t.quantity))
        else:
            remaining = t.quantity
            while remaining > 0 and buy_queue:
                buy_price, buy_qty = buy_queue[0]
                matched = min(remaining, buy_qty)
                if t.price > buy_price:
                    wins_qty += matched
                else:
                    losses_qty += matched
                remaining_buy = buy_qty - matched
                if remaining_buy == 0:
                    buy_queue.pop(0)
                else:
                    buy_queue[0] = (buy_price, remaining_buy)
                remaining -= matched

    pm.n_round_trips = wins_qty + losses_qty
    pm.win_rate = wins_qty / pm.n_round_trips if pm.n_round_trips > 0 else 0.0
