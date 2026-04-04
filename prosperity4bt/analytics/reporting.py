"""
Reporting orchestrator: creates output artifacts and prints terminal summary.

Called automatically by the backtest CLI after every run.
Output layout per run:
  outputs/backtests/<timestamp>/
    plots/
      <product>.png
      _combined.png   (if >1 product)
    metrics.json
    insights.txt
    llm_insights.txt  (if LLM available)
    trades.csv
    config_snapshot.json
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from prosperity4bt.analytics.insights import (
    UNRELIABLE_LOCAL_PRODUCTS,
    generate_deterministic_insights,
    generate_llm_insights,
)
from prosperity4bt.analytics.metrics import BacktestMetrics, ProductMetrics, extract_metrics
from prosperity4bt.analytics.plotting import plot_combined, plot_product
from prosperity4bt.models import BacktestResult


# ── Artifact writers ──────────────────────────────────────────────────────────

def _write_metrics_json(metrics: BacktestMetrics, output_dir: Path) -> None:
    def pm_to_dict(pm: ProductMetrics) -> dict:
        return {
            "product": pm.product,
            "final_pnl": pm.final_pnl,
            "realized_pnl": pm.realized_pnl,
            "unrealized_pnl": pm.unrealized_pnl,
            "n_buys": pm.n_buys,
            "n_sells": pm.n_sells,
            "total_buy_volume": pm.total_buy_volume,
            "total_sell_volume": pm.total_sell_volume,
            "avg_buy_price": pm.avg_buy_price,
            "avg_sell_price": pm.avg_sell_price,
            "avg_buy_edge_vs_mid": pm.avg_buy_edge,
            "avg_sell_edge_vs_mid": pm.avg_sell_edge,
            "final_inventory": pm.final_inventory,
            "max_inventory": pm.max_inventory,
            "min_inventory": pm.min_inventory,
            "max_drawdown": pm.max_drawdown,
            "win_rate": pm.win_rate,
            "n_round_trips": pm.n_round_trips,
        }

    data = {
        "run_timestamp": metrics.run_timestamp,
        "total_pnl": metrics.total_pnl,
        "products": {p: pm_to_dict(pm) for p, pm in metrics.products.items()},
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_insights_txt(
    deterministic: list[str],
    llm_text: str | None,
    output_dir: Path,
) -> None:
    lines: list[str] = ["=== Deterministic Insights ===", ""]
    for i, insight in enumerate(deterministic, 1):
        lines.append(f"{i:2d}. {insight}")
    lines.append("")

    if llm_text:
        lines += ["=== LLM Insights (Claude) ===", "", llm_text, ""]
    else:
        lines.append("LLM insights skipped; deterministic insights generated.")

    with (output_dir / "insights.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    if llm_text:
        with (output_dir / "llm_insights.txt").open("w", encoding="utf-8") as f:
            f.write(llm_text)


def _write_trades_csv(result: BacktestResult, output_dir: Path) -> None:
    with (output_dir / "trades.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "symbol", "side", "price", "quantity", "buyer", "seller"])
        for trade_row in result.trades:
            t = trade_row.trade
            if t.buyer == "SUBMISSION":
                side = "buy"
            elif t.seller == "SUBMISSION":
                side = "sell"
            else:
                side = "market"
            writer.writerow([t.timestamp, t.symbol, side, t.price, t.quantity, t.buyer, t.seller])


def _write_config(
    algorithm_path: str,
    days: list[tuple[int, int]],
    match_trades_mode: str,
    merge_pnl: bool,
    output_dir: Path,
) -> None:
    config = {
        "algorithm": algorithm_path,
        "days": [f"round{r}_day{d}" for r, d in days],
        "match_trades": match_trades_mode,
        "merge_pnl": merge_pnl,
    }
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ── Terminal summary ──────────────────────────────────────────────────────────

def _print_terminal_summary(
    metrics: BacktestMetrics,
    deterministic_insights: list[str],
    output_dir: Path,
) -> None:
    sep = "─" * 60

    print(f"\n{sep}")
    print("  BACKTEST ANALYTICS")
    print(f"  Results saved to: {output_dir}")
    products_str = ", ".join(metrics.products.keys())
    print(f"  Products traded : {products_str}")
    print(sep)

    for product, pm in metrics.products.items():
        print(f"\n  {product}")
        print(f"    PnL              : {pm.final_pnl:>+12,.0f}  "
              f"(realized {pm.realized_pnl:+,.0f} / unrealized {pm.unrealized_pnl:+,.0f})")
        print(f"    Trades           : {pm.n_buys} buys / {pm.n_sells} sells")
        print(f"    Volume           : {pm.total_buy_volume} buy / {pm.total_sell_volume} sell")
        if pm.total_buy_volume > 0:
            print(f"    Avg buy price    : {pm.avg_buy_price:.2f}  "
                  f"(edge vs mid: {pm.avg_buy_edge:+.2f})")
        if pm.total_sell_volume > 0:
            print(f"    Avg sell price   : {pm.avg_sell_price:.2f}  "
                  f"(edge vs mid: {pm.avg_sell_edge:+.2f})")
        print(f"    Inventory        : final {pm.final_inventory:+d}  "
              f"min {pm.min_inventory}  max {pm.max_inventory}")
        print(f"    Max drawdown     : {pm.max_drawdown:,.0f}")
        if pm.n_round_trips >= 5:
            print(f"    Win rate         : {pm.win_rate:.0%} ({pm.n_round_trips} round-trips)")
        if product in UNRELIABLE_LOCAL_PRODUCTS:
            print(f"    *** WARNING: local backtest may under-model bot interaction / "
                  "conversion effects for this product ***")

    print(f"\n  Total PnL: {metrics.total_pnl:+,.0f}")

    print(f"\n{sep}")
    print("  TOP INSIGHTS")
    print(sep)
    shown = 0
    for insight in deterministic_insights:
        # Skip if just the "no concerns" placeholder
        if "No significant concerns" in insight and len(deterministic_insights) > 1:
            continue
        print(f"  • {insight}")
        shown += 1
        if shown >= 8:
            remaining = len(deterministic_insights) - shown
            if remaining > 0:
                print(f"  ... and {remaining} more in insights.txt")
            break

    print(sep)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_analytics_pipeline(
    result: BacktestResult,
    run_timestamp: str,
    algorithm_path: str,
    days: list[tuple[int, int]],
    match_trades_mode: str,
    merge_pnl: bool,
    no_plots: bool = False,
    no_llm_insights: bool = False,
) -> Path:
    """
    Full analytics pipeline:
      1. Extract metrics from BacktestResult.
      2. Generate plots (unless --no-plots).
      3. Compute deterministic insights.
      4. Optionally call LLM for commentary.
      5. Write all artifacts to disk.
      6. Print terminal summary.

    Returns the output directory path.
    """
    output_dir = Path.cwd() / "outputs" / "backtests" / run_timestamp
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics
    metrics = extract_metrics(result, run_timestamp)

    # 2. Plots
    if not no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        for pm in metrics.products.values():
            plot_product(pm, plots_dir)
        if len(metrics.products) > 1:
            plot_combined(metrics, plots_dir)

    # 3. Deterministic insights
    det_insights = generate_deterministic_insights(metrics)

    # 4. LLM commentary
    llm_text: str | None = None
    if not no_llm_insights:
        llm_text = generate_llm_insights(metrics, det_insights)
        if llm_text is None:
            # Only print skip message if the user seemingly wants LLM (key present but insights off)
            import os
            if not os.environ.get("ANTHROPIC_API_KEY") or \
               os.environ.get("ENABLE_LLM_INSIGHTS", "").lower() not in ("true", "1", "yes"):
                print("  [llm] LLM insights skipped; deterministic insights generated.")
                print("        Set ANTHROPIC_API_KEY and ENABLE_LLM_INSIGHTS=true to enable.")

    # 5. Write artifacts
    _write_metrics_json(metrics, output_dir)
    _write_insights_txt(det_insights, llm_text, output_dir)
    _write_trades_csv(result, output_dir)
    _write_config(algorithm_path, days, match_trades_mode, merge_pnl, output_dir)

    # 6. Terminal summary
    _print_terminal_summary(metrics, det_insights, output_dir)

    return output_dir
