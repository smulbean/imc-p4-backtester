# Script Reference

Complete guide to every runnable script and trader file in this repo.
All commands assume the venv is active: `source .venv/bin/activate`.

---

## Trader files

These are the files you submit to Prosperity or backtest against.
Each contains a `Trader` class with a `run(state)` method.

### `tomatoe_trader.py` (repo root) — **main submission file**

The primary Tomatoes market maker. Uses `TomatoesTrader(ProductTrader)` with:
- L2 microprice + OBI + book-pressure alpha signal
- Inventory skew to flatten positions
- Two-layer execution: selective taking then passive quoting
- Persists smoothed alpha via `traderData` JSON across timestamps

**Backtest:**
```sh
prosperity4bt tomatoe_trader.py 0
prosperity4bt tomatoe_trader.py 0--1    # day -1 only
prosperity4bt tomatoe_trader.py 0--2    # day -2 only
```

---

### `Tomatoe1.py` (repo root) — alternate Tomatoes variant

Simpler version of the Tomatoes market maker. Fewer alpha components,
slightly different size/edge parameters. Useful as a baseline comparison.

**Backtest:**
```sh
prosperity4bt Tomatoe1.py 0
```

---

### `prosperity4bt/tomatoe2.py` — Tomatoes variant with richer history signals

Fully self-contained `Trader` class (no `ProductTrader` base). Uses:
- Weighted mid / L2 microprice alpha
- OBI, book pressure gradient
- Mean-reversion gap signals (3-bar, 10-bar)
- Internal history lists stored in `traderData`

**Backtest:**
```sh
prosperity4bt prosperity4bt/tomatoe2.py 0
```

---

### `prosperity4bt/mm_tomatoe.py` — alternate Tomatoes market maker

Another Tomatoes trader variant with additional params:
- `WMID_L2_EDGE_COEF`, `SHOCK_REVERSION_10_COEF`
- Alpha-weighted sizing (min/base/max size based on alpha strength)
- Lighter inventory control

**Backtest:**
```sh
prosperity4bt prosperity4bt/mm_tomatoe.py 0
```

---

### `prosperity4bt/emerald.py` — Emeralds-only trader

Market makes EMERALDS with a hard fair value of 10,000 and an
**Olivia signal overlay**: detects when the informed trader "Olivia"
is active, shifts quotes and sizing in her direction.

Signal mechanics:
- Tracks `last_buy_ts` / `last_sell_ts` for Olivia from market and own trades
- Signal expires after `OLIVIA_SIGNAL_TTL = 1200` timestamps
- Skews fair value ±1, adjusts bid/ask and size when signal is live

**Backtest:**
```sh
prosperity4bt prosperity4bt/emerald.py 0
```

---

### `prosperity4bt/resources/round0/agents/current.py` — combined Tomatoes + Emeralds trader

Runs both `TomatoesTrader` and `EmeraldsTrader` (with Olivia overlay) together
in one submission. Intended to be the combined bot for official submission.

**Backtest:**
```sh
prosperity4bt prosperity4bt/resources/round0/agents/current.py 0
```

---

## Analysis and optimization scripts

These scripts run independently — not through the `prosperity4bt` CLI.

---

### `prosperity4bt/robust_test.py` — robustness test suite

Runs 6 structured stress tests against a trader without writing any output files:

| Test | What it checks |
|------|---------------|
| 1. Parameter perturbation | Adds ±20% noise to all tuning params; checks PnL distribution |
| 2. Time split | Compares H1 vs H2 PnL within each day; flags time-of-day dependency |
| 3. Inventory stress | Tests 4 tighter inventory control regimes |
| 4. Taking aggressiveness | Sweeps `TAKE_EDGE` from very aggressive to very conservative |
| 5. Quote aggressiveness | Sweeps `MIN_MAKE_EDGE` / `MAX_MAKE_EDGE` / `SPREAD_EDGE_MULT` |
| 6. Queue logic | Sweeps queue-dominance / skip-bad-queue / thick-queue thresholds |

**Usage:**
```sh
# Default: runs tomatoe_trader.py on all available days in round 0
python prosperity4bt/robust_test.py

# Specify a different trader
python prosperity4bt/robust_test.py prosperity4bt/tomatoe2.py

# Specify a day
python prosperity4bt/robust_test.py 0--1
python prosperity4bt/robust_test.py tomatoe_trader.py 0--2

# Both trader and day
python prosperity4bt/robust_test.py prosperity4bt/mm_tomatoe.py 0
```

Output is printed to terminal. No files written.

> **Note:** The perturbation params in `PERTURB_PARAMS` are tuned for a queue-aware
> MM trader. Edit them to match whichever trader you're testing.

---

### `prosperity4bt/resources/round0/agents/analyze.py` — diagnostic tool

Answers two questions per day: **is this strategy making markets or trading directionally?**
and **does it look overfit?**

Per day, per symbol it computes:
- PnL summary, max drawdown, per-step Sharpe ratio
- Fill breakdown: taker buy/sell vs maker buy/sell (by volume)
- Inventory profile: mean |position|, max position, autocorrelation
- Directional signals: inventory→return correlation, PnL split flat vs positioned
- Cross-day consistency table as an overfitting indicator

Fill classification heuristic:
- `taker_buy`: bought at price ≥ best ask (lifted the ask)
- `maker_buy`: bought below best ask (passive bid was hit)
- `taker_sell`: sold at price ≤ best bid (hit the bid)
- `maker_sell`: sold above best bid (passive ask was lifted)

**Usage:**
```sh
# All round 0 days
python prosperity4bt/resources/round0/agents/analyze.py tomatoe_trader.py

# Specific day
python prosperity4bt/resources/round0/agents/analyze.py tomatoe_trader.py 0--1
python prosperity4bt/resources/round0/agents/analyze.py tomatoe_trader.py 0--2
```

Output is printed to terminal. No files written.

---

### `prosperity4bt/gsTom.py` — grid search optimizer

Patches class constants in `TomatoesTrader` via regex, runs backtests for
each parameter combination, and ranks them by a weighted score:

```
score = PNL_WEIGHT * total_pnl
      + SMOOTHNESS_WEIGHT * (1 - std_dev_across_days / mean_pnl)
      - DOWNSIDE_WEIGHT * worst_day_penalty
```

Saves results to `grid_search_results.json` and `grid_search_results.csv`.

Configure at the top of the file:
- `PARAM_GRID` — dict of param name → list of values to try
- `MAX_TESTS` — cap on number of combinations (default 250)
- `PNL_WEIGHT` / `SMOOTHNESS_WEIGHT` / `DOWNSIDE_WEIGHT` — scoring weights

> **Note:** `gsTom.py` invokes `prosperity4bt` as a subprocess. Make sure the
> venv is active and `prosperity4bt` is on PATH before running.

**Usage:**
```sh
python prosperity4bt/gsTom.py
```

Output files (`grid_search_results.json`, `grid_search_results.csv`) are
gitignored automatically.

---

## Analytics pipeline (auto-runs after every backtest)

Built into the backtester itself. Runs automatically after every
`prosperity4bt ...` command. No separate invocation needed.

Output location: `outputs/backtests/<timestamp>/`

| File | Contents |
|------|----------|
| `plots/<product>.png` | 3-panel chart: price+trades, inventory, PnL |
| `plots/_combined.png` | PnL across all products (when >1 product traded) |
| `metrics.json` | All computed stats: PnL, edges, inventory, drawdown, win rate |
| `insights.txt` | Deterministic rule-based insights + optional LLM commentary |
| `llm_insights.txt` | LLM commentary only (if enabled) |
| `trades.csv` | Every fill: timestamp, symbol, side, price, quantity |
| `config_snapshot.json` | Algorithm path, days, match mode, merge_pnl flag |

**CLI flags to control analytics:**

| Flag | Effect |
|------|--------|
| `--no-plots` | Skip generating PNG charts |
| `--no-llm-insights` | Skip Claude commentary even if API key is set |

**Environment variables for LLM commentary:**

```sh
export ANTHROPIC_API_KEY=sk-ant-...
export ENABLE_LLM_INSIGHTS=true
```

Both must be set. Without them the analytics pipeline still runs fully.

---

## Quick reference

| What you want | Command |
|--------------|---------|
| Backtest main trader, all days | `prosperity4bt tomatoe_trader.py 0` |
| Backtest one day | `prosperity4bt tomatoe_trader.py 0--1` |
| Backtest, skip plots | `prosperity4bt tomatoe_trader.py 0 --no-plots` |
| Backtest, merge PnL across days | `prosperity4bt tomatoe_trader.py 0 --merge-pnl` |
| Run robustness suite | `python prosperity4bt/robust_test.py` |
| Run diagnostic analysis | `python prosperity4bt/resources/round0/agents/analyze.py tomatoe_trader.py` |
| Run grid search | `python prosperity4bt/gsTom.py` |
| Enable LLM insights | `ANTHROPIC_API_KEY=... ENABLE_LLM_INSIGHTS=true prosperity4bt tomatoe_trader.py 0` |
