"""
Matplotlib-based charts for backtest results.

Each product gets a 3-panel figure: price+trades, inventory, PnL.
A combined PnL chart is also generated when multiple products are traded.

Gracefully no-ops if matplotlib is not installed.
"""
from __future__ import annotations

import math
from pathlib import Path

from prosperity4bt.analytics.metrics import BacktestMetrics, ProductMetrics

_MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend; safe in all environments
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def _check_matplotlib() -> bool:
    if not _MATPLOTLIB_AVAILABLE:
        print("  [plots] matplotlib not installed — skipping plots. Install with: pip install matplotlib")
    return _MATPLOTLIB_AVAILABLE


def plot_product(pm: ProductMetrics, output_dir: Path) -> Path | None:
    """
    Generate a 3-panel chart for a single product:
      Panel 1: mid price + best bid/ask + buy/sell trade markers
      Panel 2: inventory over time
      Panel 3: cumulative PnL over time

    Returns the saved file path, or None if matplotlib is unavailable.
    """
    if not _check_matplotlib():
        return None

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.45)

    ts = pm.timestamps

    # ── Panel 1: Price ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    # Best bid / ask (faint lines — may have nan gaps, matplotlib handles them)
    ax1.plot(ts, pm.best_bid_series, color="#4682B4", alpha=0.25, linewidth=0.8, label="Best bid")
    ax1.plot(ts, pm.best_ask_series, color="#FF8C00", alpha=0.25, linewidth=0.8, label="Best ask")

    # Mid price
    ax1.plot(ts, pm.mid_prices, color="#555555", linewidth=1.0, label="Mid price")

    # Buy markers
    buy_ts = [t.timestamp for t in pm.own_trades if t.side == "buy"]
    buy_px = [t.price for t in pm.own_trades if t.side == "buy"]
    if buy_ts:
        ax1.scatter(buy_ts, buy_px, marker="^", color="#22AA44", s=55, zorder=5,
                    label=f"Buy  ({pm.n_buys} trades, avg edge {pm.avg_buy_edge:+.2f})")

    # Sell markers
    sell_ts = [t.timestamp for t in pm.own_trades if t.side == "sell"]
    sell_px = [t.price for t in pm.own_trades if t.side == "sell"]
    if sell_ts:
        ax1.scatter(sell_ts, sell_px, marker="v", color="#CC3333", s=55, zorder=5,
                    label=f"Sell ({pm.n_sells} trades, avg edge {pm.avg_sell_edge:+.2f})")

    pnl_str = f"{pm.final_pnl:+,.0f}"
    inv_str = f"{pm.final_inventory:+d}"
    ax1.set_title(
        f"{pm.product}   PnL: {pnl_str}   "
        f"Trades: {pm.n_buys}B / {pm.n_sells}S   "
        f"Final inv: {inv_str}   "
        f"Max DD: {pm.max_drawdown:,.0f}",
        fontsize=11,
    )
    ax1.set_ylabel("Price")
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.7)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Inventory ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    if pm.inventory_series:
        ax2.plot(ts, pm.inventory_series, color="#8B008B", linewidth=1.0)
        ax2.axhline(y=0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax2.fill_between(
            ts, pm.inventory_series, 0,
            where=[v > 0 for v in pm.inventory_series],
            alpha=0.18, color="#22AA44",
        )
        ax2.fill_between(
            ts, pm.inventory_series, 0,
            where=[v < 0 for v in pm.inventory_series],
            alpha=0.18, color="#CC3333",
        )
    ax2.set_ylabel("Inventory")
    ax2.grid(True, alpha=0.25)

    # ── Panel 3: PnL ─────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    if pm.pnl_series:
        ax3.plot(ts, pm.pnl_series, color="#4169E1", linewidth=1.0)
        ax3.axhline(y=0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
        ax3.fill_between(
            ts, pm.pnl_series, 0,
            where=[v > 0 for v in pm.pnl_series],
            alpha=0.18, color="#22AA44",
        )
        ax3.fill_between(
            ts, pm.pnl_series, 0,
            where=[v < 0 for v in pm.pnl_series],
            alpha=0.18, color="#CC3333",
        )
    ax3.set_ylabel("PnL")
    ax3.set_xlabel("Timestamp")
    ax3.grid(True, alpha=0.25)

    output_path = output_dir / f"{pm.product.lower()}.png"
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_combined(metrics: BacktestMetrics, output_dir: Path) -> Path | None:
    """
    Single figure with one PnL subplot per product + a total PnL line.
    Only meaningful when multiple products are traded.
    """
    if not _check_matplotlib():
        return None

    products = list(metrics.products.values())
    n = len(products)
    if n == 0:
        return None

    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 3 * (n + 1)), sharex=False)
    if n == 0:
        plt.close(fig)
        return None

    colors = ["#4169E1", "#22AA44", "#CC3333", "#FF8C00", "#8B008B", "#00CED1"]

    for i, pm in enumerate(products):
        ax = axes[i]
        ax.plot(pm.timestamps, pm.pnl_series, color=colors[i % len(colors)], linewidth=1.0)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_ylabel(f"{pm.product}\nPnL")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"{pm.product}  final PnL: {pm.final_pnl:+,.0f}", fontsize=10)

    # Total PnL panel (sum across products at each shared timestamp index)
    # Use the product with most timestamps as reference
    ref = max(products, key=lambda p: len(p.timestamps))
    ts_ref = ref.timestamps

    # Build per-product pnl indexed by timestamp for fast lookup
    pnl_by_ts: dict[str, dict[int, float]] = {}
    for pm in products:
        pnl_by_ts[pm.product] = dict(zip(pm.timestamps, pm.pnl_series))

    total_pnl = []
    for ts in ts_ref:
        s = sum(pnl_by_ts[pm.product].get(ts, 0.0) for pm in products)
        total_pnl.append(s)

    ax_total = axes[n]
    ax_total.plot(ts_ref, total_pnl, color="#333333", linewidth=1.2, label="Total PnL")
    ax_total.axhline(y=0, color="black", linewidth=0.6, linestyle="--", alpha=0.6)
    ax_total.fill_between(ts_ref, total_pnl, 0,
                          where=[v > 0 for v in total_pnl], alpha=0.18, color="#22AA44")
    ax_total.fill_between(ts_ref, total_pnl, 0,
                          where=[v < 0 for v in total_pnl], alpha=0.18, color="#CC3333")
    ax_total.set_ylabel("Total PnL")
    ax_total.set_xlabel("Timestamp")
    ax_total.set_title(f"Total  final PnL: {metrics.total_pnl:+,.0f}", fontsize=10)
    ax_total.grid(True, alpha=0.25)

    fig.suptitle("Combined PnL Summary", fontsize=13, y=1.01)
    fig.tight_layout()

    output_path = output_dir / "_combined.png"
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return output_path
