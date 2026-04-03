"""
Insights engine for backtest analytics.

Two layers:
  1. Deterministic rule-based insights (always runs).
  2. Optional LLM commentary via Claude API (requires ANTHROPIC_API_KEY
     and ENABLE_LLM_INSIGHTS=true; gracefully skipped otherwise).
"""
from __future__ import annotations

import os

from prosperity4bt.analytics.metrics import BacktestMetrics, ProductMetrics

# Products where local backtest results are less reliable.
# Reasons: conversion mechanics, bot interaction, correlated-asset pricing.
UNRELIABLE_LOCAL_PRODUCTS: dict[str, str] = {
    "MACARONS": (
        "Macarons rely on conversion mechanics and bot interactions that the local "
        "backtester cannot model accurately. Validate on the official website."
    ),
    "RAINFOREST_RESIN": (
        "Rainforest Resin has complex mean-reversion dynamics driven by market-maker "
        "bots. Local matching may overstate fill rates. Validate on the official website."
    ),
    "KELP": (
        "Kelp pricing is tied to correlated assets and conversion rates not captured "
        "by the local backtester. Validate on the official website."
    ),
}

# Thresholds (tune as needed)
_DIRECTIONAL_IMBALANCE_RATIO = 0.30   # net_vol / max_side_vol to flag directional bias
_OPEN_INVENTORY_THRESHOLD = 10        # final |inventory| to flag unclosed position
_HIGH_TRADE_FREQ_RATIO = 0.50         # trades per timestamp to flag overtrading
_LOW_TRADE_FREQ_RATIO = 0.01          # trades per timestamp to flag undertrading
_DRAWDOWN_MULTIPLE_WARNING = 3.0      # max_dd / final_pnl multiple to flag
_LOW_WIN_RATE = 0.40
_HIGH_WIN_RATE = 0.70
_MIN_ROUND_TRIPS_FOR_WIN_RATE = 5
_EXTREME_INVENTORY = 20               # abs(inventory) to flag extreme sizing


def generate_deterministic_insights(metrics: BacktestMetrics) -> list[str]:
    """
    Return a list of concise insight strings derived purely from metrics.
    Prefixed with [PRODUCT] for easy scanning.
    """
    insights: list[str] = []

    for product, pm in metrics.products.items():
        p = f"[{product}]"
        n_ts = len(pm.timestamps)

        # 1. Directional bias
        max_side = max(pm.total_buy_volume, pm.total_sell_volume, 1)
        net_vol = pm.total_buy_volume - pm.total_sell_volume
        if abs(net_vol) > _DIRECTIONAL_IMBALANCE_RATIO * max_side:
            direction = "long" if net_vol > 0 else "short"
            insights.append(
                f"{p} Strategy is directionally {direction}-biased "
                f"(net volume {net_vol:+d} vs max side {max_side})."
            )

        # 2. Inventory not cleared at end
        if abs(pm.final_inventory) > _OPEN_INVENTORY_THRESHOLD:
            insights.append(
                f"{p} Final inventory {pm.final_inventory:+d} is not flat — "
                f"unrealized exposure {pm.unrealized_pnl:+,.0f}. "
                "Inventory management may need attention."
            )

        # 3. Buy edge vs mid
        if pm.total_buy_volume > 0:
            if pm.avg_buy_edge < -0.5:
                insights.append(
                    f"{p} Buys are chasing price: avg buy edge vs mid = "
                    f"{pm.avg_buy_edge:+.2f} (negative = buying above mid)."
                )
            elif pm.avg_buy_edge > 1.0:
                insights.append(
                    f"{p} Buys are well-placed: avg buy edge vs mid = "
                    f"{pm.avg_buy_edge:+.2f}."
                )

        # 4. Sell edge vs mid
        if pm.total_sell_volume > 0:
            if pm.avg_sell_edge < -0.5:
                insights.append(
                    f"{p} Sells are chasing price: avg sell edge vs mid = "
                    f"{pm.avg_sell_edge:+.2f} (negative = selling below mid)."
                )
            elif pm.avg_sell_edge > 1.0:
                insights.append(
                    f"{p} Sells are well-placed: avg sell edge vs mid = "
                    f"{pm.avg_sell_edge:+.2f}."
                )

        # 5. Trade frequency
        n_trades = pm.n_buys + pm.n_sells
        if n_ts > 0:
            freq = n_trades / n_ts
            if freq > _HIGH_TRADE_FREQ_RATIO:
                insights.append(
                    f"{p} High trade frequency: {n_trades} trades over {n_ts} "
                    f"timestamps ({freq:.2f}/ts). May be overtrading."
                )
            elif freq < _LOW_TRADE_FREQ_RATIO and n_ts > 100:
                insights.append(
                    f"{p} Very low trade frequency: {n_trades} trades over {n_ts} "
                    f"timestamps. Strategy may be too passive or rarely triggering."
                )

        # 6. Drawdown relative to final PnL
        if pm.max_drawdown > 0 and abs(pm.final_pnl) > 0:
            ratio = pm.max_drawdown / abs(pm.final_pnl)
            if ratio > _DRAWDOWN_MULTIPLE_WARNING:
                insights.append(
                    f"{p} Max drawdown ({pm.max_drawdown:,.0f}) is "
                    f"{ratio:.1f}x final PnL ({pm.final_pnl:+,.0f}) — "
                    "significant intra-day risk vs. outcome."
                )

        # 7. Round-trip win rate
        if pm.n_round_trips >= _MIN_ROUND_TRIPS_FOR_WIN_RATE:
            if pm.win_rate < _LOW_WIN_RATE:
                insights.append(
                    f"{p} Low round-trip win rate: {pm.win_rate:.0%} "
                    f"over {pm.n_round_trips} completed pairs."
                )
            elif pm.win_rate > _HIGH_WIN_RATE:
                insights.append(
                    f"{p} Strong round-trip win rate: {pm.win_rate:.0%} "
                    f"over {pm.n_round_trips} completed pairs."
                )

        # 8. Extreme inventory
        if pm.max_inventory > _EXTREME_INVENTORY or pm.min_inventory < -_EXTREME_INVENTORY:
            insights.append(
                f"{p} Inventory reached extremes "
                f"(min={pm.min_inventory}, max={pm.max_inventory}) — "
                "review position sizing and limit controls."
            )

        # 9. Realized vs unrealized split
        if abs(pm.unrealized_pnl) > 0.5 * abs(pm.final_pnl) and abs(pm.final_pnl) > 100:
            insights.append(
                f"{p} {abs(pm.unrealized_pnl / pm.final_pnl):.0%} of PnL "
                f"({pm.unrealized_pnl:+,.0f}) is unrealized — "
                "carry risk if the position is not intentional."
            )

        # 10. No trades at all
        if n_trades == 0:
            insights.append(
                f"{p} No own trades recorded. "
                "Check that orders are being submitted and passing limit enforcement."
            )

        # 11. Product-specific warning
        if product in UNRELIABLE_LOCAL_PRODUCTS:
            insights.append(f"{p} WARNING: {UNRELIABLE_LOCAL_PRODUCTS[product]}")

    if not insights:
        insights.append("No significant concerns detected in this backtest run.")

    return insights


def generate_llm_insights(metrics: BacktestMetrics, deterministic_insights: list[str]) -> str | None:
    """
    Call Claude to produce a human-readable commentary on the backtest.

    Requirements:
      - ANTHROPIC_API_KEY must be set in the environment.
      - ENABLE_LLM_INSIGHTS must be 'true' (case-insensitive).
      - The `anthropic` Python package must be installed.

    Returns the commentary string, or None if any requirement is not met.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    enable_llm = os.environ.get("ENABLE_LLM_INSIGHTS", "").strip().lower() in ("true", "1", "yes")

    if not api_key or not enable_llm:
        return None

    try:
        import anthropic  # type: ignore[import]
    except ImportError:
        print("  [llm] anthropic package not installed — skipping LLM insights. Install with: pip install anthropic")
        return None

    # Build a compact text summary to send (not raw logs)
    lines: list[str] = [
        "=== Backtest Summary ===",
        f"Total PnL: {metrics.total_pnl:+,.0f}",
        "",
    ]
    for product, pm in metrics.products.items():
        lines += [
            f"Product: {product}",
            f"  PnL: {pm.final_pnl:+,.0f}  (realized: {pm.realized_pnl:+,.0f}, unrealized: {pm.unrealized_pnl:+,.0f})",
            f"  Trades: {pm.n_buys} buys ({pm.total_buy_volume} units) / {pm.n_sells} sells ({pm.total_sell_volume} units)",
            f"  Avg buy price: {pm.avg_buy_price:.2f}  |  Avg sell price: {pm.avg_sell_price:.2f}",
            f"  Avg buy edge vs mid: {pm.avg_buy_edge:+.2f}  |  Avg sell edge vs mid: {pm.avg_sell_edge:+.2f}",
            f"  Final inventory: {pm.final_inventory:+d}  (min: {pm.min_inventory}, max: {pm.max_inventory})",
            f"  Max drawdown: {pm.max_drawdown:,.0f}",
            f"  Round-trip win rate: {pm.win_rate:.0%} over {pm.n_round_trips} pairs",
            "",
        ]

    lines += ["=== Deterministic Insights ==="]
    for insight in deterministic_insights:
        lines.append(f"  - {insight}")

    prompt_text = "\n".join(lines)

    system_prompt = (
        "You are an expert quantitative trading analyst reviewing backtest results "
        "for a Prosperity trading competition algorithm.\n\n"
        "Given the backtest metrics and deterministic insights below, provide:\n"
        "1. 5–10 concise bullet-point insights (focus on what the numbers mean, not just restating them)\n"
        "2. Biggest strength of this strategy\n"
        "3. Biggest weakness or risk\n"
        "4. Whether this strategy appears overfit to local backtest assumptions\n"
        "5. Whether inventory/risk control looks healthy\n"
        "6. For each product: whether to trust local backtest results or validate on the official website\n\n"
        "Be specific, actionable, and brief. Do not restate numbers — interpret them."
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return response.content[0].text
    except Exception as exc:
        print(f"  [llm] LLM call failed: {exc}")
        return None
