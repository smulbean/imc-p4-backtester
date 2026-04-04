#!/usr/bin/env python3
"""
robust_test_queue_mm.py — Robustness tests for a queue-aware structural Tomatoes market maker.

Designed for the queue-position-aware stable MM version.

Tests:
  1. Parameter perturbation   — noise on queue/MM tuning params
  2. Time split               — first vs second half of each day
  3. Inventory stress         — tighten position limits / skew
  4. Taking stress            — vary TAKE_EDGE
  5. Quote aggressiveness     — vary make-edge settings
  6. Queue logic stress       — vary queue-dominance / skip-thick settings

Usage:
    python robust_test_queue_mm.py
    python robust_test_queue_mm.py 0
    python robust_test_queue_mm.py 0-1
    python robust_test_queue_mm.py 0--1
    python robust_test_queue_mm.py my_trader_file.py
    python robust_test_queue_mm.py my_trader_file.py 0--2
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


DEFAULT_TRADER_FILE = "tomatoe_trader.py"
TRADER_CLASS = "TomatoesTrader"


# Queue-aware MM params from your latest version
PERTURB_PARAMS: list[tuple[str, float, float, float, float]] = [
    ("TAKE_EDGE", 1.5, 0.20, 0.5, 4.0),
    ("BASE_SIZE", 12.0, 0.20, 1.0, 40.0),
    ("STRONG_SIZE", 24.0, 0.20, 1.0, 80.0),
    ("MAX_QUOTE_SIZE", 36.0, 0.20, 1.0, 100.0),
    ("SOFT_POS", 42.0, 0.20, 5.0, 79.0),
    ("HARD_POS", 76.0, 0.10, 10.0, 80.0),
    ("INV_SKEW_PER_UNIT", 0.018, 0.25, 0.0, 0.2),
    ("MAX_SKEW", 4.5, 0.25, 0.0, 15.0),
    ("MIN_MAKE_EDGE", 0.0, 0.25, 0.0, 5.0),
    ("MAX_MAKE_EDGE", 1.0, 0.25, 0.0, 5.0),
    ("SPREAD_EDGE_MULT", 0.08, 0.25, 0.0, 2.0),
    ("WIDE_SPREAD_SIZE_MULT", 1.20, 0.15, 1.0, 3.0),
    ("QUEUE_DOMINANCE_AT_TOUCH", 1.15, 0.15, 0.5, 3.0),
    ("IMPROVE_SIZE_FRAC", 0.85, 0.15, 0.1, 2.0),
    ("THICK_QUEUE_MULT", 2.2, 0.20, 0.5, 8.0),
]

INVENTORY_SCENARIOS = [
    ("Baseline", {
        "SOFT_POS": 42,
        "HARD_POS": 76,
        "INV_SKEW_PER_UNIT": 0.018,
        "MAX_SKEW": 4.5,
    }),
    ("Moderate tighter", {
        "SOFT_POS": 32,
        "HARD_POS": 60,
        "INV_SKEW_PER_UNIT": 0.026,
        "MAX_SKEW": 5.0,
    }),
    ("Strict", {
        "SOFT_POS": 22,
        "HARD_POS": 40,
        "INV_SKEW_PER_UNIT": 0.036,
        "MAX_SKEW": 6.0,
    }),
    ("Very strict", {
        "SOFT_POS": 12,
        "HARD_POS": 25,
        "INV_SKEW_PER_UNIT": 0.050,
        "MAX_SKEW": 7.0,
    }),
]

TAKE_EDGE_SCENARIOS = [
    ("Very aggressive", {"TAKE_EDGE": 1.0}),
    ("Aggressive", {"TAKE_EDGE": 1.3}),
    ("Baseline-ish", {"TAKE_EDGE": 1.5}),
    ("Conservative", {"TAKE_EDGE": 2.0}),
    ("Very conservative", {"TAKE_EDGE": 2.5}),
]

QUOTE_SCENARIOS = [
    ("Very aggressive quotes", {
        "MIN_MAKE_EDGE": 0.0,
        "MAX_MAKE_EDGE": 0.75,
        "SPREAD_EDGE_MULT": 0.05,
    }),
    ("Aggressive quotes", {
        "MIN_MAKE_EDGE": 0.0,
        "MAX_MAKE_EDGE": 1.0,
        "SPREAD_EDGE_MULT": 0.08,
    }),
    ("Medium quotes", {
        "MIN_MAKE_EDGE": 0.10,
        "MAX_MAKE_EDGE": 1.4,
        "SPREAD_EDGE_MULT": 0.14,
    }),
    ("Conservative quotes", {
        "MIN_MAKE_EDGE": 0.25,
        "MAX_MAKE_EDGE": 2.0,
        "SPREAD_EDGE_MULT": 0.22,
    }),
]

QUEUE_SCENARIOS = [
    ("Baseline queue", {
        "QUEUE_DOMINANCE_AT_TOUCH": 1.15,
        "IMPROVE_SIZE_FRAC": 0.85,
        "THICK_QUEUE_MULT": 2.2,
        "SKIP_BAD_QUEUE": True,
    }),
    ("More queue aggressive", {
        "QUEUE_DOMINANCE_AT_TOUCH": 1.30,
        "IMPROVE_SIZE_FRAC": 1.00,
        "THICK_QUEUE_MULT": 3.0,
        "SKIP_BAD_QUEUE": False,
    }),
    ("Cheaper step-ahead", {
        "QUEUE_DOMINANCE_AT_TOUCH": 1.05,
        "IMPROVE_SIZE_FRAC": 0.65,
        "THICK_QUEUE_MULT": 1.8,
        "SKIP_BAD_QUEUE": True,
    }),
    ("Very selective queue", {
        "QUEUE_DOMINANCE_AT_TOUCH": 1.20,
        "IMPROVE_SIZE_FRAC": 0.90,
        "THICK_QUEUE_MULT": 1.4,
        "SKIP_BAD_QUEUE": True,
    }),
    ("Sit in queue more often", {
        "QUEUE_DOMINANCE_AT_TOUCH": 1.05,
        "IMPROVE_SIZE_FRAC": 0.75,
        "THICK_QUEUE_MULT": 4.0,
        "SKIP_BAD_QUEUE": False,
    }),
]


def section(title: str) -> None:
    print(f"\n{'━' * 72}")
    print(f"  {title}")
    print(f"{'━' * 72}")


def parse_cli() -> tuple[Path, str]:
    args = sys.argv[1:]

    trader_file = _ROOT / DEFAULT_TRADER_FILE
    day_spec = "0"

    if len(args) == 1:
        if args[0].endswith(".py"):
            trader_file = (_ROOT / args[0]).resolve()
        else:
            day_spec = args[0]
    elif len(args) >= 2:
        trader_file = (_ROOT / args[0]).resolve()
        day_spec = args[1]

    if not trader_file.exists():
        print(f"Error: trader file not found: {trader_file}")
        sys.exit(1)

    return trader_file, day_spec


def normalize_day_spec(spec: str) -> tuple[int, Optional[int]]:
    if "--" in spec:
        left, right = spec.split("--", 1)
        return int(left), -int(right)
    if "-" in spec:
        left, right = spec.split("-", 1)
        return int(left), int(right)
    return int(spec), None


def parse_days(spec: str) -> list[tuple[int, int]]:
    reader = PackageResourcesReader()
    round_num, maybe_day = normalize_day_spec(spec)

    if maybe_day is not None:
        if not has_day_data(reader, round_num, maybe_day):
            print(f"Error: no data for round {round_num}, day {maybe_day}")
            sys.exit(1)
        return [(round_num, maybe_day)]

    days = [(round_num, d) for d in range(-20, 200) if has_day_data(reader, round_num, d)]
    if not days:
        print(f"Error: no data for round {round_num}")
        sys.exit(1)
    return days


def load_module(trader_file: Path):
    if str(trader_file.parent) not in sys.path:
        sys.path.insert(0, str(trader_file.parent))
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    mod = import_module(trader_file.stem)
    if not hasattr(mod, TRADER_CLASS):
        print(f"Error: {trader_file.name} has no class '{TRADER_CLASS}'")
        sys.exit(1)
    if not hasattr(mod, "Trader"):
        print(f"Error: {trader_file.name} has no class 'Trader'")
        sys.exit(1)
    return mod


def patch(mod, overrides: dict[str, Any]) -> None:
    cls = getattr(mod, TRADER_CLASS)
    for k, v in overrides.items():
        if k in {
            "BASE_SIZE", "STRONG_SIZE", "MAX_QUOTE_SIZE",
            "SOFT_POS", "HARD_POS", "WIDE_SPREAD_THRESHOLD"
        }:
            v = int(round(v))
        setattr(cls, k, v)


def run_days(mod, days: list[tuple[int, int]], overrides: Optional[dict[str, Any]] = None,
             show_bar: bool = False) -> dict:
    reader = PackageResourcesReader()
    combined: dict[str, dict] = defaultdict(lambda: {
        "total_pnl": 0.0,
        "step_pnls": [],
        "pnl_curve": [],
        "ts_seq": [],
    })

    for r, d in days:
        reload(mod)
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

        by_sym: dict[str, list] = defaultdict(list)
        for row in result.activity_logs:
            cols = row.columns
            if len(cols) <= 16:
                continue
            by_sym[cols[2]].append(cols)

        for sym, rows in by_sym.items():
            rows_s = sorted(rows, key=lambda c: c[1])
            pnl_col = [float(c[16]) for c in rows_s]
            steps = [pnl_col[i] - pnl_col[i - 1] for i in range(1, len(pnl_col))]

            combined[sym]["total_pnl"] += pnl_col[-1] if pnl_col else 0.0
            combined[sym]["step_pnls"].extend(steps)
            combined[sym]["pnl_curve"].extend(pnl_col)
            combined[sym]["ts_seq"].extend(c[1] for c in rows_s)

    return dict(combined)


def combined_pnl(result: dict) -> float:
    return sum(v["total_pnl"] for v in result.values())


def sharpe(steps: list[float]) -> Optional[float]:
    if len(steps) < 2:
        return None
    mu = statistics.mean(steps)
    sd = statistics.stdev(steps)
    return (mu / sd * math.sqrt(len(steps))) if sd > 0 else None


def pct_change(new: float, base: float) -> str:
    if base == 0:
        return "N/A"
    p = (new - base) / abs(base) * 100.0
    sign = "+" if p >= 0 else ""
    return f"{sign}{p:.1f}%"


def test_perturbation(mod, days: list[tuple[int, int]], baseline_pnl: float,
                      n_trials: int = 25, seed: int = 42) -> None:
    section(f"TEST 1  Parameter Perturbation  ({n_trials} trials)")
    print(f"  Baseline total PnL: {baseline_pnl:,.0f}\n")

    rng = random.Random(seed)
    trial_pnls: list[float] = []

    for i in range(n_trials):
        overrides: dict[str, Any] = {}
        for name, default, pct, lo, hi in PERTURB_PARAMS:
            factor = 1.0 + rng.uniform(-pct, pct)
            overrides[name] = max(lo, min(hi, default * factor))

        pnl = combined_pnl(run_days(mod, days, overrides=overrides))
        trial_pnls.append(pnl)
        print(f"  Trial {i+1:02d}  PnL={pnl:>10,.0f}  {'✓' if pnl > 0 else '❌'}")

    positive = sum(1 for p in trial_pnls if p > 0)
    under_80 = sum(1 for p in trial_pnls if p < baseline_pnl * 0.8)

    sorted_pnls = sorted(trial_pnls)
    n = len(sorted_pnls)

    print(f"\n  Min     {sorted_pnls[0]:>10,.0f}")
    print(f"  P25     {sorted_pnls[n//4]:>10,.0f}")
    print(f"  Median  {sorted_pnls[n//2]:>10,.0f}")
    print(f"  P75     {sorted_pnls[3*n//4]:>10,.0f}")
    print(f"  Max     {sorted_pnls[-1]:>10,.0f}")
    print(f"\n  Positive PnL trials: {positive}/{n} ({positive/n*100:.0f}%)")
    print(f"  Below 80% of base:   {under_80}/{n} ({under_80/n*100:.0f}%)")


def test_time_split(mod, days: list[tuple[int, int]]) -> None:
    section("TEST 2  Time Split  (first vs second half of each day)")
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
            cols = row.columns
            if len(cols) <= 16:
                continue
            by_sym[cols[2]].append(cols)

        print(f"\n  Day {d:+d}")
        for sym, rows in sorted(by_sym.items()):
            rows_s = sorted(rows, key=lambda c: c[1])
            pnl_col = [float(c[16]) for c in rows_s]
            n = len(pnl_col)
            if n < 4:
                continue

            mid = n // 2
            h1 = pnl_col[mid]
            h2 = pnl_col[-1] - pnl_col[mid]
            total = pnl_col[-1]

            ratio_str = "N/A" if h1 == 0 else f"{h2 / h1:.2f}"
            print(f"    {sym:<12}  Total={total:>8,.0f}  H1={h1:>8,.0f}  H2={h2:>8,.0f}  ratio={ratio_str}")


def test_inventory_stress(mod, days: list[tuple[int, int]], baseline_pnl: float) -> None:
    section("TEST 3  Inventory Stress")
    print(f"  {'Scenario':<18}  {'PnL':>10}  {'vs Baseline':>12}")
    print(f"  {'─'*18}  {'─'*10}  {'─'*12}")

    for label, overrides in INVENTORY_SCENARIOS:
        pnl = combined_pnl(run_days(mod, days, overrides=overrides))
        print(f"  {label:<18}  {pnl:>10,.0f}  {pct_change(pnl, baseline_pnl):>12}")


def test_take_edge_sweep(mod, days: list[tuple[int, int]], baseline_pnl: float) -> None:
    section("TEST 4  Taking Aggressiveness Sweep")
    print(f"  {'Scenario':<20}  {'PnL':>10}  {'vs Baseline':>12}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*12}")

    for label, overrides in TAKE_EDGE_SCENARIOS:
        pnl = combined_pnl(run_days(mod, days, overrides=overrides))
        print(f"  {label:<20}  {pnl:>10,.0f}  {pct_change(pnl, baseline_pnl):>12}")


def test_quote_sweep(mod, days: list[tuple[int, int]], baseline_pnl: float) -> None:
    section("TEST 5  Quote Aggressiveness Sweep")
    print(f"  {'Scenario':<24}  {'PnL':>10}  {'vs Baseline':>12}")
    print(f"  {'─'*24}  {'─'*10}  {'─'*12}")

    for label, overrides in QUOTE_SCENARIOS:
        pnl = combined_pnl(run_days(mod, days, overrides=overrides))
        print(f"  {label:<24}  {pnl:>10,.0f}  {pct_change(pnl, baseline_pnl):>12}")


def test_queue_logic(mod, days: list[tuple[int, int]], baseline_pnl: float) -> None:
    section("TEST 6  Queue Logic Sweep")
    print(f"  {'Scenario':<24}  {'PnL':>10}  {'vs Baseline':>12}")
    print(f"  {'─'*24}  {'─'*10}  {'─'*12}")

    for label, overrides in QUEUE_SCENARIOS:
        pnl = combined_pnl(run_days(mod, days, overrides=overrides))
        print(f"  {label:<24}  {pnl:>10,.0f}  {pct_change(pnl, baseline_pnl):>12}")


def main() -> None:
    trader_file, day_spec = parse_cli()
    days = parse_days(day_spec)
    mod = load_module(trader_file)

    section(f"ROBUSTNESS TESTS: {trader_file.name} — days {[d for _, d in days]}")
    print("  Running baseline...", end=" ", flush=True)
    baseline = run_days(mod, days)
    base_pnl = combined_pnl(baseline)
    print(f"baseline PnL = {base_pnl:,.0f}")

    test_perturbation(mod, days, base_pnl)
    test_time_split(mod, days)
    test_inventory_stress(mod, days, base_pnl)
    test_take_edge_sweep(mod, days, base_pnl)
    test_quote_sweep(mod, days, base_pnl)
    test_queue_logic(mod, days, base_pnl)

    section("All tests complete")


if __name__ == "__main__":
    main()