# IMC Prosperity 4 Backtester

A local backtester for Prosperity 4 algorithms, built on top of [jmerle's Prosperity 3 backtester](https://github.com/jmerle/imc-prosperity-3-backtester) with P4-specific changes.

## Setup

```sh
pip install -U prosperity4bt
```

Or, for development (changes take effect immediately):

```sh
# Install uv if needed: https://docs.astral.sh/uv/
uv venv && source .venv/bin/activate
uv sync
```

## Running a backtest (Prosperity 4)

```sh
# All days in Round 0
prosperity4bt example/trader.py 0

# Single day (Round 0, Day -1)
prosperity4bt example/trader.py 0--1

# Single day (Round 0, Day -2)
prosperity4bt example/trader.py 0--2

# Both days, merged PnL
prosperity4bt example/trader.py 0 --merge-pnl

# Use a custom data directory instead of the bundled data
prosperity4bt example/trader.py 0 --data prosperity4bt/resources

# Skip saving the output log
prosperity4bt example/trader.py 0 --no-out

# Print trader stdout while running (useful for debugging)
prosperity4bt example/trader.py 0 --print
```

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
2. Add a `__init__.py` (empty) to make it a package.
3. Drop in your price and trade CSV files following the naming convention above.
4. If the new round introduces new products, add them to `LIMITS` in [prosperity4bt/data.py](prosperity4bt/data.py).

## Active products and limits (Round 0)

Defined in [prosperity4bt/data.py](prosperity4bt/data.py):

| Product  | Position limit |
|----------|---------------|
| TOMATOES | 80            |
| EMERALDS | 80            |

## Order matching

Orders placed by `Trader.run()` at a given timestamp are matched against:
1. **Order depths** (priority): fills at the depth price.
2. **Market trades** (fallback): fills at *your* order price (not the trade price).

Configure market-trade matching with `--match-trades`:
- `all` (default): match trades at or worse than your quote.
- `worse`: match only trades strictly worse than your quote.
- `none`: ignore market trades entirely.

Limits are enforced before matching. If all your orders for a product would breach the limit, **all** orders for that product are cancelled.

## Environment variables

During a backtest the following variables are set (do not rely on these in submitted code):
- `PROSPERITY4BT_ROUND`
- `PROSPERITY4BT_DAY`
