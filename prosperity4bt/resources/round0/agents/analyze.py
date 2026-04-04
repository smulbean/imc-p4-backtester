#!/usr/bin/env python3
"""
analyze.py — Prosperity 4 backtest diagnostic tool (Round 0).

Answers two questions:
  1. Is this strategy market making or directional trading?
  2. Does it look overfit?

Usage:
    python analyze.py <trader.py>           # all round 0 days
    python analyze.py <trader.py> 0--1      # day -1 only
    python analyze.py <trader.py> 0--2      # day -2 only

Output per day per symbol:
  - PnL summary + max drawdown + per-step Sharpe
  - Fill breakdown: taker vs maker (by volume)
  - Inventory profile: mean |pos|, max pos, autocorrelation
  - Directional signals: inventory→return correlation, PnL split flat/positioned
  - Day-over-day consistency table (overfitting indicator)
"""

import math
import statistics
import sys
from collections import defaultdict
from importlib import import_module, reload
from pathlib import Path
from typing import Optional

# Ensure repo root is on path so `from datamodel import ...` works in trader files.
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from prosperity4bt.data import has_day_data
from prosperity4bt.file_reader import PackageResourcesReader
from prosperity4bt.models import BacktestResult, TradeMatchingMode
from prosperity4bt.runner import run_backtest


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_trader_module(path: Path):
    sys.path.append(str(path.parent))
    if str(_ROOT) not in sys.path:
        sys.path.append(str(_ROOT))
    mod = import_module(path.stem)
    if not hasattr(mod, "Trader"):
        print(f"Error: {path} has no Trader class")
        sys.exit(1)
    return mod


def parse_days(spec: str) -> list[tuple[int, int]]:
    """Parse '0', '0--1', '0--2' into (round, day) tuples."""
    reader = PackageResourcesReader()
    if "-" in spec:
        round_num, day_num = map(int, spec.split("-", 1))
        if not has_day_data(reader, round_num, day_num):
            print(f"Error: no data for round {round_num} day {day_num}")
            sys.exit(1)
        return [(round_num, day_num)]
    else:
        round_num = int(spec)
        days = [(round_num, d) for d in range(-5, 100) if has_day_data(reader, round_num, d)]
        if not days:
            print(f"Error: no data for round {round_num}")
            sys.exit(1)
        return days


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

# Activity log column indices (from runner.py create_activity_logs):
#   [0]=day [1]=ts [2]=sym [3]=bid1_p [4]=bid1_v ... [9]=ask1_p [10]=ask1_v
#   ... [15]=mid_price [16]=realized_pnl + position * mid  (pnl_mtm)
_TS  = 1
_SYM = 2
_B1P = 3   # best bid price
_A1P = 9   # best ask price
_MID = 15
_PNL = 16  # realized PnL from prior fills + current position * mid


def build_market_state(result: BacktestResult) -> dict:
    """
    Returns {sym: {ts: {mid, best_bid, best_ask, pnl_mtm}}}.

    pnl_mtm at timestamp t = realized_pnl_from_fills_before_t
                             + position_before_t * mid_t
    (logged before the current step's orders are matched)
    """
    state: dict = defaultdict(dict)
    for row in result.activity_logs:
        c = row.columns
        state[c[_SYM]][c[_TS]] = {
            "mid":      c[_MID],
            "best_bid": c[_B1P] if c[_B1P] != "" else None,
            "best_ask": c[_A1P] if c[_A1P] != "" else None,
            "pnl_mtm":  c[_PNL],
        }
    return dict(state)


def classify_fills(result: BacktestResult, mkt: dict) -> dict:
    """
    Returns {sym: [{"ts", "price", "qty" (signed, +buy/-sell), "type"}]}.

    Fill type heuristic — based on fill price vs market quote at that timestamp:
      taker_buy:   we bought at price >= best_ask  (we lifted the ask)
      maker_buy:   we bought at price <  best_ask  (someone hit our passive bid)
      taker_sell:  we sold   at price <= best_bid  (we hit the bid)
      maker_sell:  we sold   at price >  best_bid  (someone lifted our passive ask)
    """
    fills: dict = defaultdict(list)
    for tr in result.trades:
        t = tr.trade
        is_buy  = t.buyer  == "SUBMISSION"
        is_sell = t.seller == "SUBMISSION"
        if not (is_buy or is_sell):
            continue

        snap = mkt.get(t.symbol, {}).get(t.timestamp, {})
        ba   = snap.get("best_ask")
        bb   = snap.get("best_bid")

        if is_buy:
            ftype = "taker_buy"  if (ba is not None and t.price >= ba) else "maker_buy"
            fills[t.symbol].append({"ts": t.timestamp, "price": t.price, "qty": t.quantity, "type": ftype})
        else:
            ftype = "taker_sell" if (bb is not None and t.price <= bb) else "maker_sell"
            fills[t.symbol].append({"ts": t.timestamp, "price": t.price, "qty": -t.quantity, "type": ftype})

    return dict(fills)


def build_position_series(fills: list[dict], timestamps: list[int]) -> dict[int, int]:
    """
    Reconstructs end-of-step position at each timestamp from fills.
    Position at ts = cumulative signed qty of all our fills up to and including ts.
    """
    delta: dict = defaultdict(int)
    for f in fills:
        delta[f["ts"]] += f["qty"]

    pos, series = 0, {}
    for ts in sorted(timestamps):
        pos += delta.get(ts, 0)
        series[ts] = pos
    return series


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def sharpe(steps: list[float]) -> Optional[float]:
    if len(steps) < 2:
        return None
    mu, sd = statistics.mean(steps), statistics.stdev(steps)
    return (mu / sd * math.sqrt(len(steps))) if sd > 0 else None


def max_drawdown(curve: list[float]) -> float:
    peak, dd = 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        dd   = min(dd, v - peak)
    return dd


def autocorr(series: list[float], lag: int = 1) -> Optional[float]:
    n = len(series)
    if n <= lag + 1:
        return None
    mu  = statistics.mean(series)
    var = statistics.variance(series)
    if var == 0:
        return None
    cov = sum((series[i] - mu) * (series[i - lag] - mu) for i in range(lag, n)) / (n - lag)
    return cov / var


def pearson(xs: list[float], ys: list[float]) -> Optional[float]:
    if len(xs) < 10:
        return None
    try:
        mx, my = statistics.mean(xs), statistics.mean(ys)
        sx, sy = statistics.stdev(xs), statistics.stdev(ys)
        if sx == 0 or sy == 0:
            return None
        cov = statistics.mean((x - mx) * (y - my) for x, y in zip(xs, ys))
        return cov / (sx * sy)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-day analysis
# ---------------------------------------------------------------------------

def analyze_day(result: BacktestResult) -> dict:
    """Run analysis for one day. Returns a summary dict for cross-day comparison."""
    mkt       = build_market_state(result)
    all_fills = classify_fills(result, mkt)
    summaries = {}

    symbols = sorted(mkt.keys())

    print(f"\n{'━' * 64}")
    print(f"  Day {result.day_num:+d}")
    print(f"{'━' * 64}")

    for sym in symbols:
        snap   = mkt[sym]
        ts_seq = sorted(snap.keys())
        fills  = all_fills.get(sym, [])

        # ── Position series ──────────────────────────────────────────────
        pos_series = build_position_series(fills, ts_seq)
        # position at ts = end-of-step position (after ts's fills)
        positions  = [pos_series.get(ts, 0) for ts in ts_seq]

        # ── PnL series ───────────────────────────────────────────────────
        # pnl_mtm[t] = realized_before_t + pos_before_t * mid_t
        # So pnl_mtm already includes unrealized mark-to-market.
        pnl_curve  = [snap[ts]["pnl_mtm"] for ts in ts_seq]
        step_pnls  = [pnl_curve[i] - pnl_curve[i - 1] for i in range(1, len(pnl_curve))]
        total_pnl  = pnl_curve[-1] if pnl_curve else 0.0
        dd         = max_drawdown(pnl_curve)
        sr         = sharpe(step_pnls)

        # ── Fill classification ──────────────────────────────────────────
        tb = [f for f in fills if f["type"] == "taker_buy"]
        mb = [f for f in fills if f["type"] == "maker_buy"]
        ts_ = [f for f in fills if f["type"] == "taker_sell"]
        ms = [f for f in fills if f["type"] == "maker_sell"]

        def vol(lst): return sum(abs(f["qty"]) for f in lst)
        tb_v, mb_v, ts_v, ms_v = vol(tb), vol(mb), vol(ts_), vol(ms)
        total_vol = tb_v + mb_v + ts_v + ms_v

        passive_pct = (mb_v + ms_v) / total_vol * 100 if total_vol else 0.0
        active_pct  = (tb_v + ts_v) / total_vol * 100 if total_vol else 0.0

        # ── Inventory stats ──────────────────────────────────────────────
        abs_pos    = [abs(p) for p in positions]
        mean_ap    = statistics.mean(abs_pos)  if abs_pos  else 0.0
        max_ap     = max(abs_pos)              if abs_pos  else 0
        std_pos    = statistics.stdev(positions) if len(positions) > 1 else 0.0
        ac         = autocorr(positions)
        pct_hi_inv = sum(1 for p in abs_pos if p > 40) / len(abs_pos) * 100 if abs_pos else 0.0

        # ── Inventory → next-return correlation ─────────────────────────
        # Does our end-of-step position predict the next mid-price move?
        # A positive corr means we're positioned in the direction of moves → directional edge.
        mids = [snap[ts]["mid"] for ts in ts_seq]
        inv_xs = [float(positions[i]) for i in range(len(ts_seq) - 1)]
        ret_ys = [mids[i + 1] - mids[i]    for i in range(len(ts_seq) - 1)
                  if mids[i] is not None and mids[i + 1] is not None]
        inv_ret_corr = pearson(inv_xs[:len(ret_ys)], ret_ys)

        # ── PnL split: positioned vs flat ────────────────────────────────
        # step_pnls[i] covers ts_seq[i] → ts_seq[i+1].
        # We use end-of-step position[i] as the position driving that move.
        pnl_pos  = sum(step_pnls[i] for i in range(len(step_pnls)) if abs(positions[i]) > 10)
        pnl_flat = sum(step_pnls[i] for i in range(len(step_pnls)) if abs(positions[i]) <= 10)

        # ── Print ─────────────────────────────────────────────────────────
        def fmt(v, w=10, dp=0):
            return f"{v:>{w},.{dp}f}"

        print(f"\n  {sym}")
        print(f"  {'─' * 60}")

        # PnL
        print(f"\n  PnL Summary")
        print(f"    Total PnL (realized + MTM)  {fmt(total_pnl)}")
        print(f"    Final position              {result.day_num:>+4}  → {pos_series.get(ts_seq[-1], 0):>5}")
        print(f"    Max drawdown                {fmt(dd)}")
        sr_flag = ""
        if sr is not None:
            sr_flag = "  ⚠ low"  if abs(sr) < 0.5 else ("  ✓" if abs(sr) > 1.5 else "")
            print(f"    Per-step Sharpe             {sr:>10.3f}{sr_flag}")

        # Fills
        print(f"\n  Fill Breakdown  (total volume traded: {total_vol})")
        if total_vol:
            def pvol(v): return f"{v:4d} units  ({v / total_vol * 100:4.1f}%)"
            print(f"    Taker buys   {pvol(tb_v)}  — {len(tb)} fills")
            print(f"    Maker buys   {pvol(mb_v)}  — {len(mb)} fills")
            print(f"    Taker sells  {pvol(ts_v)}  — {len(ts_)} fills")
            print(f"    Maker sells  {pvol(ms_v)}  — {len(ms)} fills")
            print(f"    ─")
            verdict = (
                "mostly market making (passive dominant)"         if passive_pct > 60 else
                "mostly directional taking (active dominant)"     if active_pct  > 60 else
                "mixed: both making and taking"
            )
            print(f"    Passive {passive_pct:.1f}%  /  Active {active_pct:.1f}%  → {verdict}")

        # Inventory
        print(f"\n  Inventory Profile")
        print(f"    Mean |position|             {mean_ap:>10.1f}")
        print(f"    Max  |position|             {max_ap:>10}")
        print(f"    Std of position             {std_pos:>10.1f}")
        print(f"    Time at |pos| > 40          {pct_hi_inv:>9.1f}%")
        if ac is not None:
            ac_flag = "  ⚠ inventory persists — directional tendency" if ac > 0.90 else ""
            print(f"    Inventory autocorr (lag-1)  {ac:>10.3f}{ac_flag}")

        # Directional signals
        print(f"\n  Directional vs MM Signals")
        if inv_ret_corr is not None:
            irc_flag = (
                "  ← real alpha"          if abs(inv_ret_corr) > 0.08 else
                "  ← weak predictive edge" if abs(inv_ret_corr) > 0.03 else
                "  ← no predictive edge"
            )
            print(f"    Inventory → next-return corr  {inv_ret_corr:>8.4f}{irc_flag}")
        print(f"    PnL while |pos| > 10      {fmt(pnl_pos)}")
        print(f"    PnL while |pos| ≤ 10      {fmt(pnl_flat)}")
        if total_pnl != 0:
            frac = pnl_pos / total_pnl * 100
            frac_flag = (
                "  ⚠ mostly directional — check if alpha is real"  if frac > 70 else
                "  ✓ spread-dominant, low directional dependency"   if frac < 30 else
                "  mixed — moderate directional component"
            )
            print(f"    % PnL while positioned        {frac:>9.1f}%{frac_flag}")

        # Overfitting checklist
        print(f"\n  Overfitting Checklist  (compare these across days/param perturbations)")
        print(f"    Sharpe          {sr:.3f}"    if sr is not None else "    Sharpe          N/A")
        print(f"    Max DD          {dd:,.0f}")
        print(f"    Passive fill%   {passive_pct:.1f}%"   if total_vol else "    Passive fill%   N/A")
        print(f"    Mean |pos|      {mean_ap:.1f}")
        print(f"    Inv→ret corr    {inv_ret_corr:.4f}" if inv_ret_corr is not None else "    Inv→ret corr    N/A")
        print(f"    → If Sharpe/PnL drops >50% on unseen day or ±20% param change → likely overfit")
        print(f"    → If inv→ret corr flips sign on the other day → alpha is overfit")

        summaries[sym] = {
            "total_pnl":    total_pnl,
            "sharpe":       sr,
            "max_dd":       dd,
            "passive_pct":  passive_pct,
            "mean_abs_pos": mean_ap,
            "inv_ret_corr": inv_ret_corr,
            "day":          result.day_num,
        }

    return summaries


# ---------------------------------------------------------------------------
# Cross-day comparison
# ---------------------------------------------------------------------------

def print_cross_day_table(all_summaries: list[dict]) -> None:
    if len(all_summaries) < 2:
        return

    print(f"\n{'━' * 64}")
    print(f"  Cross-Day Consistency  (key overfitting test)")
    print(f"{'━' * 64}")

    # Gather all symbols across all days
    all_syms: set = set()
    for day_summ in all_summaries:
        all_syms.update(day_summ.keys())

    for sym in sorted(all_syms):
        rows = [d[sym] for d in all_summaries if sym in d]
        if len(rows) < 2:
            continue

        print(f"\n  {sym}")
        header = f"    {'Day':>6}  {'PnL':>10}  {'Sharpe':>8}  {'MaxDD':>8}  {'Pass%':>6}  {'Inv→Ret':>8}"
        print(header)
        print(f"    {'─' * 56}")
        for r in rows:
            sr_str  = f"{r['sharpe']:.3f}"    if r["sharpe"]       is not None else "   N/A"
            irc_str = f"{r['inv_ret_corr']:.4f}" if r["inv_ret_corr"] is not None else "    N/A"
            print(
                f"    {r['day']:>+6}  {r['total_pnl']:>10,.0f}  "
                f"{sr_str:>8}  {r['max_dd']:>8,.0f}  "
                f"{r['passive_pct']:>5.1f}%  {irc_str:>8}"
            )

        # Consistency verdict
        pnls = [r["total_pnl"] for r in rows]
        if max(pnls) > 0 and min(pnls) > 0:
            ratio = min(pnls) / max(pnls)
            if ratio < 0.4:
                print(f"    ⚠ PnL varies {ratio:.0%} between days — inconsistent, possible overfit")
            elif ratio < 0.7:
                print(f"    △ PnL ratio {ratio:.0%} — moderate consistency")
            else:
                print(f"    ✓ PnL ratio {ratio:.0%} — consistent across days")
        elif min(pnls) <= 0:
            print(f"    ⚠ Negative PnL on at least one day — check if strategy generalises")

        sharpes = [r["sharpe"] for r in rows if r["sharpe"] is not None]
        if len(sharpes) == len(rows):
            sr_ratio = min(sharpes) / max(sharpes) if max(sharpes) > 0 else 0
            print(f"    Sharpe consistency: {sr_ratio:.0%}  {'✓' if sr_ratio > 0.6 else '⚠'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    trader_path = Path(sys.argv[1]).resolve()
    day_spec    = sys.argv[2] if len(sys.argv) > 2 else "0"

    days = parse_days(day_spec)
    mod  = load_trader_module(trader_path)

    reader = PackageResourcesReader()

    print(f"\n{'━' * 64}")
    print(f"  BACKTEST ANALYSIS: {trader_path.name}  —  Round {days[0][0]}")
    print(f"  Days: {', '.join(str(d) for _, d in days)}")
    print(f"{'━' * 64}")

    all_summaries: list[dict] = []

    for round_num, day_num in days:
        reload(mod)
        result = run_backtest(
            trader=mod.Trader(),
            file_reader=reader,
            round_num=round_num,
            day_num=day_num,
            print_output=False,
            trade_matching_mode=TradeMatchingMode.all,
            show_progress_bar=True,
        )
        day_summary = analyze_day(result)
        all_summaries.append(day_summary)

    print_cross_day_table(all_summaries)
    print()


if __name__ == "__main__":
    main()
