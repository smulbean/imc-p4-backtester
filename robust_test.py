#!/usr/bin/env python3
"""
robust_test.py — Five overfitting robustness tests for tomatoe_trader.py

Tests run in order:
  1. Parameter perturbation  — ±20% noise on all tunable params, 25 trials
  2. Feature ablation        — zero out each alpha coefficient one at a time
  3. Time split              — first vs second half of each day
  4. Pure MM baseline        — remove all alpha, compare PnL change
  5. Inventory stress        — tighten position limits, see if PnL survives

Usage:
    python robust_test.py                 # all round 0 days
    python robust_test.py 0--1            # day -1 only
    python robust_test.py 0--2            # day -2 only

Requires: tomatoe_trader.py in the repo root (hardcoded — tests are specific
to TomatoesTrader's class attributes).
"""

import math
import random
import statistics
import sys
from collections import defaultdict
from importlib import import_module, reload
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from prosperity4bt.data import has_day_data
from prosperity4bt.file_reader import PackageResourcesReader
from prosperity4bt.models import BacktestResult, TradeMatchingMode
from prosperity4bt.runner import run_backtest

TRADER_FILE = _ROOT / "tomatoe_trader.py"
TRADER_CLASS = "TomatoesTrader"   # class whose attrs we patch

# ── Params eligible for perturbation ────────────────────────────────────────
# (name, default, perturb_pct, absolute_min, absolute_max)
PERTURB_PARAMS: list[tuple[str, float, float, float, float]] = [
    ("TAKE_EDGE",                   1.0,   0.20,  0.1,  5.0),
    ("MAKE_EDGE",                   1.0,   0.20,  0.1,  5.0),
    ("BASE_SIZE",                   8.0,   0.20,  1.0, 40.0),
    ("STRONG_SIZE",                18.0,   0.20,  1.0, 80.0),
    ("INV_SKEW_PER_UNIT",           0.009, 0.20,  0.0,  0.1),
    ("MAX_SKEW",                    5.2,   0.20,  0.5, 15.0),
    ("WMID_L2_EDGE_COEF",           0.35,  0.20, -5.0,  5.0),
    ("OBI_2_COEF",                  1.85,  0.20, -5.0,  5.0),
    ("BOOK_PRESSURE_GRADIENT2_COEF",-0.65, 0.20, -5.0,  5.0),
    ("MR_GAP_3_COEF",               0.08,  0.20, -5.0,  5.0),
    ("SHOCK_REVERSION_10_COEF",     0.01,  0.20, -5.0,  5.0),
    ("ALPHA_CLIP",                  2.0,   0.20,  0.1, 10.0),
]

# ── Alpha coefficients for ablation ─────────────────────────────────────────
ALPHA_COEFS = [
    "WMID_L2_EDGE_COEF",
    "OBI_2_COEF",
    "BOOK_PRESSURE_GRADIENT2_COEF",
    "MR_GAP_3_COEF",
    "SHOCK_REVERSION_10_COEF",
]

# ── Inventory stress scenarios ───────────────────────────────────────────────
INV_STRESS_SCENARIOS = [
    ("Baseline   (SOFT=72, skew=0.009)", {"SOFT_POS": 72, "HARD_POS": 79, "INV_SKEW_PER_UNIT": 0.009}),
    ("Moderate   (SOFT=50, skew=0.018)", {"SOFT_POS": 50, "HARD_POS": 60, "INV_SKEW_PER_UNIT": 0.018}),
    ("Strict     (SOFT=30, skew=0.030)", {"SOFT_POS": 30, "HARD_POS": 40, "INV_SKEW_PER_UNIT": 0.030}),
    ("Very strict(SOFT=15, skew=0.050)", {"SOFT_POS": 15, "HARD_POS": 25, "INV_SKEW_PER_UNIT": 0.050}),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_module():
    sys.path.append(str(TRADER_FILE.parent))
    if str(_ROOT) not in sys.path:
        sys.path.append(str(_ROOT))
    mod = import_module(TRADER_FILE.stem)
    if not hasattr(mod, TRADER_CLASS):
        print(f"Error: {TRADER_FILE} has no class '{TRADER_CLASS}'")
        sys.exit(1)
    return mod


def parse_days(spec: str) -> list[tuple[int, int]]:
    reader = PackageResourcesReader()
    if "-" in spec:
        r, d = map(int, spec.split("-", 1))
        return [(r, d)]
    r = int(spec)
    days = [(r, d) for d in range(-5, 100) if has_day_data(reader, r, d)]
    if not days:
        print(f"Error: no data for round {r}")
        sys.exit(1)
    return days


def patch(mod, overrides: dict[str, Any]) -> None:
    """Apply attribute overrides to the trader subclass."""
    cls = getattr(mod, TRADER_CLASS)
    for k, v in overrides.items():
        # Round size params to int
        if k in ("BASE_SIZE", "STRONG_SIZE", "SOFT_POS", "HARD_POS"):
            v = max(1, int(round(v)))
        setattr(cls, k, v)


def run_days(mod, days: list[tuple[int, int]], show_bar: bool = False,
             overrides: dict[str, Any] | None = None) -> dict:
    """
    Run backtest over all days. Returns per-symbol totals:
      {sym: {total_pnl, step_pnls, pnl_curve, timestamps}}

    overrides: class-attribute patches applied to TomatoesTrader after each reload.
    """
    reader   = PackageResourcesReader()
    combined: dict[str, dict] = defaultdict(lambda: {
        "total_pnl": 0.0, "step_pnls": [], "pnl_curve": [], "ts_seq": []
    })

    for r, d in days:
        reload(mod)
        # Re-apply patches after reload (reload resets class attributes to defaults)
        if overrides:
            patch(mod, overrides)
        result: BacktestResult = run_backtest(
            trader=mod.Trader(),
            file_reader=reader,
            round_num=r,
            day_num=d,
            print_output=False,
            trade_matching_mode=TradeMatchingMode.all,
            show_progress_bar=show_bar,
        )

        # Group activity logs by symbol
        by_sym: dict[str, list] = defaultdict(list)
        for row in result.activity_logs:
            by_sym[row.columns[2]].append(row.columns)

        for sym, rows in by_sym.items():
            rows_s  = sorted(rows, key=lambda c: c[1])
            pnl_col = [c[16] for c in rows_s]          # pnl_mtm
            steps   = [pnl_col[i] - pnl_col[i-1] for i in range(1, len(pnl_col))]

            combined[sym]["total_pnl"]  += pnl_col[-1] if pnl_col else 0.0
            combined[sym]["step_pnls"].extend(steps)
            combined[sym]["pnl_curve"].extend(pnl_col)
            combined[sym]["ts_seq"].extend(c[1] for c in rows_s)

    return dict(combined)


def combined_pnl(result: dict) -> float:
    return sum(v["total_pnl"] for v in result.values())


def sharpe(steps: list[float]) -> Optional[float]:
    if len(steps) < 2:
        return None
    mu, sd = statistics.mean(steps), statistics.stdev(steps)
    return (mu / sd * math.sqrt(len(steps))) if sd > 0 else None


def pct_change(new: float, base: float) -> str:
    if base == 0:
        return "  N/A"
    p = (new - base) / abs(base) * 100
    sign = "+" if p >= 0 else ""
    flag = "  ❌" if p < -40 else ("  ⚠" if p < -20 else ("  ✓" if p > -10 else ""))
    return f"{sign}{p:.1f}%{flag}"


def verdict(positive_frac: float, collapse_frac: float) -> str:
    if positive_frac >= 0.85 and collapse_frac <= 0.10:
        return "✓  ROBUST — performance stable under perturbation"
    elif positive_frac >= 0.65 and collapse_frac <= 0.25:
        return "⚠  BORDERLINE — some sensitivity, watch closely"
    else:
        return "❌  FRAGILE — strategy may be knife-edge tuned"


def section(title: str) -> None:
    print(f"\n{'━' * 68}")
    print(f"  {title}")
    print(f"{'━' * 68}")


# ── Test 1: Parameter perturbation ───────────────────────────────────────────

def test_perturbation(mod, days: list[tuple[int, int]], baseline_pnl: float,
                      n_trials: int = 25, seed: int = 42) -> None:
    section(f"TEST 1  Parameter Perturbation  ({n_trials} trials, ±20% noise)")
    print(f"  Baseline total PnL: {baseline_pnl:,.0f}\n")

    rng = random.Random(seed)
    trial_pnls: list[float] = []

    for i in range(n_trials):
        trial_overrides: dict[str, Any] = {}
        for name, default, pct, lo, hi in PERTURB_PARAMS:
            factor = 1.0 + rng.uniform(-pct, pct)
            val    = max(lo, min(hi, default * factor))
            trial_overrides[name] = val

        res = run_days(mod, days, overrides=trial_overrides)
        pnl = combined_pnl(res)
        trial_pnls.append(pnl)

        flag = "✓" if pnl > 0 else "❌"
        print(f"  Trial {i+1:02d}  PnL={pnl:>10,.0f}  {flag}")

    # Statistics
    positive     = sum(1 for p in trial_pnls if p > 0)
    collapsed    = sum(1 for p in trial_pnls if p < baseline_pnl * 0.5)
    trial_pnls_s = sorted(trial_pnls)
    n            = len(trial_pnls)

    print(f"\n  Distribution of {n} trials:")
    print(f"    Min     {trial_pnls_s[0]:>10,.0f}")
    print(f"    P25     {trial_pnls_s[n//4]:>10,.0f}")
    print(f"    Median  {trial_pnls_s[n//2]:>10,.0f}")
    print(f"    P75     {trial_pnls_s[3*n//4]:>10,.0f}")
    print(f"    Max     {trial_pnls_s[-1]:>10,.0f}")
    print(f"\n  Positive PnL:          {positive}/{n}  ({positive/n*100:.0f}%)")
    print(f"  Collapsed (<50% base): {collapsed}/{n}  ({collapsed/n*100:.0f}%)")
    print(f"\n  VERDICT: {verdict(positive/n, collapsed/n)}")


# ── Test 2: Feature ablation ─────────────────────────────────────────────────

def test_ablation(mod, days: list[tuple[int, int]], baseline_pnl: float) -> None:
    section("TEST 2  Feature Ablation  (zero out each alpha coef)")
    print(f"  Baseline total PnL: {baseline_pnl:,.0f}\n")

    print(f"  {'Feature':<36}  {'PnL':>10}  {'Change':>10}  {'Verdict'}")
    print(f"  {'─'*36}  {'─'*10}  {'─'*10}  {'─'*20}")

    results = []
    for coef in ALPHA_COEFS:
        res = run_days(mod, days, overrides={coef: 0.0})
        pnl = combined_pnl(res)
        chg = pct_change(pnl, baseline_pnl)
        results.append((coef, pnl, chg))
        print(f"  {coef+'=0':<36}  {pnl:>10,.0f}  {chg:>15}")

    # All alpha zeroed (pure structural MM, no signals)
    res_none = run_days(mod, days, overrides={c: 0.0 for c in ALPHA_COEFS})
    pnl_none = combined_pnl(res_none)
    chg_none = pct_change(pnl_none, baseline_pnl)
    print(f"  {'ALL alpha=0 (pure MM)':<36}  {pnl_none:>10,.0f}  {chg_none:>15}")

    print()
    print("  Interpretation:")
    for coef, pnl, chg in results:
        drop_pct = (baseline_pnl - pnl) / abs(baseline_pnl) * 100 if baseline_pnl != 0 else 0
        if drop_pct > 40:
            print(f"    ❌ {coef}: removing it kills {drop_pct:.0f}% of PnL — load-bearing, check for overfit")
        elif drop_pct > 15:
            print(f"    ⚠  {coef}: -${drop_pct:.0f}% — meaningful contributor")
        else:
            print(f"    ✓  {coef}: -${drop_pct:.0f}% — marginal, low overfit risk")

    alpha_add = baseline_pnl - pnl_none
    alpha_pct  = alpha_add / abs(baseline_pnl) * 100 if baseline_pnl != 0 else 0
    print(f"\n  Total alpha contribution vs pure MM: {alpha_add:+,.0f}  ({alpha_pct:+.1f}%)")
    if alpha_pct < 5:
        print("  ⚠  Alpha adds almost nothing — strategy is mostly structural MM")
    elif alpha_pct > 50:
        print("  ⚠  Strategy is heavily alpha-dependent — verify alpha is real")
    else:
        print("  ✓  Alpha adds meaningful edge on top of structural MM")


# ── Test 3: Time split (first 50% vs second 50%) ─────────────────────────────

def test_time_split(mod, days: list[tuple[int, int]]) -> None:
    section("TEST 3  Time Split  (first 50% vs second 50% of each day)")
    print("  PnL measured by pnl_mtm at midpoint vs end-of-day.")
    print("  Unrealized inventory included — split is approximate but directionally correct.\n")

    reader = PackageResourcesReader()

    for r, d in days:
        result: BacktestResult = run_backtest(
            trader=mod.Trader(),
            file_reader=reader,
            round_num=r,
            day_num=d,
            print_output=False,
            trade_matching_mode=TradeMatchingMode.all,
            show_progress_bar=False,
        )

        by_sym: dict[str, list] = defaultdict(list)
        for row in result.activity_logs:
            by_sym[row.columns[2]].append(row.columns)

        print(f"  Day {d:+d}")
        for sym, rows in sorted(by_sym.items()):
            rows_s  = sorted(rows, key=lambda c: c[1])
            pnl_col = [c[16] for c in rows_s]
            n       = len(pnl_col)
            if n < 4:
                continue
            mid_idx  = n // 2
            h1_pnl   = pnl_col[mid_idx]                    # pnl at midpoint (start ≈ 0)
            h2_pnl   = pnl_col[-1] - pnl_col[mid_idx]      # pnl from midpoint to end
            total    = pnl_col[-1]
            ratio    = h2_pnl / h1_pnl if h1_pnl != 0 else float("nan")

            flag = (
                "✓  balanced"              if 0.4 <= ratio <= 2.5 and h2_pnl > 0 else
                "⚠  second half weaker"    if 0 < ratio < 0.4 else
                "⚠  second half stronger"  if ratio > 2.5 else
                "❌ second half negative — possible overfit"
            )
            print(f"    {sym:<12}  Total={total:>8,.0f}  "
                  f"H1={h1_pnl:>8,.0f}  H2={h2_pnl:>8,.0f}  ratio={ratio:>5.2f}  {flag}")
        print()

    print("  Key: if H2 consistently collapses relative to H1 → strategy is front-loaded / overfit")
    print("       if ratio ≈ 1 → performance is uniformly distributed → more robust signal")


# ── Test 4: Pure MM baseline ─────────────────────────────────────────────────

def test_pure_mm(mod, days: list[tuple[int, int]], baseline_pnl: float) -> None:
    section("TEST 4  Pure MM Baseline  (remove all alpha, keep structure)")
    print(f"  Sets all alpha coefs to 0 — only inventory skew + spread logic remains.\n")

    res  = run_days(mod, days, overrides={c: 0.0 for c in ALPHA_COEFS})
    pnl  = combined_pnl(res)
    diff = baseline_pnl - pnl

    print(f"  Current (with alpha):  {baseline_pnl:>12,.0f}")
    print(f"  Pure MM (alpha=0):     {pnl:>12,.0f}")
    print(f"  Alpha contribution:    {diff:>+12,.0f}  ({pct_change(pnl, baseline_pnl).strip()})")
    print()

    if pnl <= 0:
        print("  ❌ Pure MM loses money — structure alone is not viable."
              " All profits depend on alpha being correct.")
    elif diff / abs(baseline_pnl) < 0.05:
        print("  ⚠  Alpha contributes <5% of PnL — your signals are nearly decorative."
              "\n     Either the alpha is weak or the structure is already capturing it.")
    elif diff / abs(baseline_pnl) > 0.60:
        print("  ⚠  Alpha accounts for >60% of PnL — heavy alpha dependence."
              "\n     Validate that these signals generalise to unseen data.")
    else:
        print("  ✓  MM structure provides a solid base; alpha adds meaningful edge on top.")

    # Also test: is inv_skew doing anything?
    res_flat = run_days(mod, days, overrides={c: 0.0 for c in ALPHA_COEFS} | {"INV_SKEW_PER_UNIT": 0.0, "MAX_SKEW": 0.0})
    pnl_flat = combined_pnl(res_flat)
    skew_contrib = pnl - pnl_flat
    print(f"\n  Pure MM with skew=0:   {pnl_flat:>12,.0f}")
    print(f"  Inventory skew value:  {skew_contrib:>+12,.0f}  "
          f"({'helps reduce inventory cost' if skew_contrib > 0 else 'slight drag on flat MM'})")


# ── Test 5: Inventory stress ──────────────────────────────────────────────────

def test_inventory_stress(mod, days: list[tuple[int, int]], baseline_pnl: float) -> None:
    section("TEST 5  Inventory Stress Test  (tighten position constraints)")
    print("  Does PnL survive when the strategy is forced to stay flatter?\n")

    print(f"  {'Scenario':<40}  {'PnL':>10}  {'vs Baseline':>12}")
    print(f"  {'─'*40}  {'─'*10}  {'─'*12}")

    for label, overrides in INV_STRESS_SCENARIOS:
        res = run_days(mod, days, overrides=overrides)
        pnl = combined_pnl(res)
        chg = pct_change(pnl, baseline_pnl)
        print(f"  {label:<40}  {pnl:>10,.0f}  {chg:>15}")

    print()
    print("  Interpretation:")
    print("    ❌ PnL collapses quickly as SOFT_POS decreases")
    print("       → most profits require building large inventory positions")
    print("       → alpha or execution edge is limited; you're essentially")
    print("         betting on holding and price returning to fair value")
    print()
    print("    ✓  PnL degrades slowly / stays positive at strict limits")
    print("       → genuine execution edge; spread capture is real")
    print("       → strategy can trade profitably without leaning on inventory")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    day_spec = sys.argv[1] if len(sys.argv) > 1 else "0"
    days     = parse_days(day_spec)
    mod      = load_module()

    print(f"\n{'━' * 68}")
    print(f"  ROBUSTNESS TESTS: {TRADER_FILE.name}  —  days {[d for _,d in days]}")
    print(f"{'━' * 68}")
    print("  Running baseline...", end=" ", flush=True)

    baseline = run_days(mod, days, show_bar=False)
    base_pnl = combined_pnl(baseline)
    print(f"baseline PnL = {base_pnl:,.0f}")

    test_perturbation(mod, days, base_pnl)
    test_ablation(mod, days, base_pnl)
    test_time_split(mod, days)
    test_pure_mm(mod, days, base_pnl)
    test_inventory_stress(mod, days, base_pnl)

    print(f"\n{'━' * 68}")
    print("  All tests complete.")
    print(f"{'━' * 68}\n")


if __name__ == "__main__":
    main()
