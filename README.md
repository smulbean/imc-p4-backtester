# IMC Prosperity 4 Backtester

A local backtester for Prosperity 4 algorithms, with automatic post-backtest analytics, plots, and insights.
Built on top of [jmerle's Prosperity 3 backtester](https://github.com/jmerle/imc-prosperity-3-backtester) with P4-specific changes.

## Setup

```sh
# Install uv if needed: https://docs.astral.sh/uv/
uv venv && source .venv/bin/activate
uv sync
```

## Running a backtest

Always activate the venv first (`source .venv/bin/activate`), then:

```sh
# All days in Round 0 — analytics run automatically after
prosperity4bt tomatoe_trader.py 0

# Single day
prosperity4bt tomatoe_trader.py 0--1
prosperity4bt tomatoe_trader.py 0--2

# Both days, cumulative PnL
prosperity4bt tomatoe_trader.py 0 --merge-pnl

# Skip saving the .log file
prosperity4bt tomatoe_trader.py 0 --no-out

# Print trader stdout while running
prosperity4bt tomatoe_trader.py 0 --print

# Skip chart generation (faster iteration)
prosperity4bt tomatoe_trader.py 0 --no-plots

# Skip LLM commentary
prosperity4bt tomatoe_trader.py 0 --no-llm-insights
```

If `prosperity4bt` is not on PATH, use:
```sh
python -m prosperity4bt tomatoe_trader.py 0
```

## Analytics (auto-runs after every backtest)

After each backtest, results are written to `outputs/backtests/<timestamp>/`:

```
outputs/backtests/2026-04-03_13-45-10/
├── plots/
│   ├── tomatoes.png        # price + buy/sell markers + inventory + PnL panels
│   ├── emeralds.png
│   └── _combined.png       # PnL across all products (when >1 product)
├── metrics.json            # all computed stats
├── insights.txt            # deterministic insights + optional LLM commentary
├── trades.csv              # every fill with side/price/qty
└── config_snapshot.json    # params used for this run
```

A summary is also printed to terminal after every run.

### Optional LLM insights (Claude)

Set two environment variables to enable natural-language commentary:

```sh
export ANTHROPIC_API_KEY=sk-ant-...
export ENABLE_LLM_INSIGHTS=true
prosperity4bt tomatoe_trader.py 0
```

Without these, the analytics pipeline still runs fully — only the LLM step is skipped.

## Other scripts

See [SCRIPTS.md](SCRIPTS.md) for full documentation of every script in the repo.

## Data file format

Price files and trade files use semicolon-delimited CSV.

**Prices** (`prices_round_<R>_day_<D>.csv`):
```
day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss
```

**Trades** (`trades_round_<R>_day_<D>.csv`):
```
timestamp;buyer;seller;symbol;currency;price;quantity
```

Up to 3 levels of bid/ask depth are supported. Empty cells are fine.

## Adding new round data

1. Create a directory: `prosperity4bt/resources/round<N>/`
2. Add an empty `__init__.py`.
3. Drop in your CSV files following the naming convention above.
4. Add new products to `LIMITS` in [prosperity4bt/data.py](prosperity4bt/data.py).

## Active products and limits (Round 0)

Defined in [prosperity4bt/data.py](prosperity4bt/data.py):

| Product  | Position limit |
|----------|---------------|
| TOMATOES | 80            |
| EMERALDS | 80            |

## Order matching

Orders from `Trader.run()` are matched against:
1. **Order depths** (priority): fills at the depth price.
2. **Market trades** (fallback): fills at *your* order price.

Control market-trade matching with `--match-trades`:
- `all` (default): match trades at or worse than your quote.
- `worse`: match only trades strictly worse than your quote.
- `none`: ignore market trades entirely.

Limits are enforced before matching. If orders for a product would breach the limit, **all** orders for that product are cancelled.

## Environment variables

Set automatically during a backtest (do not use in submitted code):
- `PROSPERITY4BT_ROUND`
- `PROSPERITY4BT_DAY`

Set manually to configure analytics:
- `ANTHROPIC_API_KEY` — enables LLM insights
- `ENABLE_LLM_INSIGHTS=true` — must also be set to actually call the API
