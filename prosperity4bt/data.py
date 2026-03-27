from collections import defaultdict
from dataclasses import dataclass

from prosperity4bt.datamodel import Symbol, Trade
from prosperity4bt.file_reader import FileReader

# P4 Round 0 products and position limits.
# Add new products here as rounds progress.
LIMITS: dict[str, int] = {
    "TOMATOES": 80,
    "EMERALDS": 80,
}


@dataclass
class PriceRow:
    day: int
    timestamp: int
    product: Symbol
    bid_prices: list[int]
    bid_volumes: list[int]
    ask_prices: list[int]
    ask_volumes: list[int]
    mid_price: float
    profit_loss: float


def get_column_values(columns: list[str], indices: list[int]) -> list[int]:
    values = []

    for index in indices:
        value = columns[index]
        if value == "":
            break

        values.append(int(value))

    return values


@dataclass
class BacktestData:
    round_num: int
    day_num: int

    prices: dict[int, dict[Symbol, PriceRow]]
    trades: dict[int, dict[Symbol, list[Trade]]]
    products: list[Symbol]
    profit_loss: dict[Symbol, float]


def create_backtest_data(
    round_num: int, day_num: int, prices: list[PriceRow], trades: list[Trade]
) -> "BacktestData":
    prices_by_timestamp: dict[int, dict[Symbol, PriceRow]] = defaultdict(dict)
    for row in prices:
        prices_by_timestamp[row.timestamp][row.product] = row

    trades_by_timestamp: dict[int, dict[Symbol, list[Trade]]] = defaultdict(lambda: defaultdict(list))
    for trade in trades:
        trades_by_timestamp[trade.timestamp][trade.symbol].append(trade)

    products = sorted(set(row.product for row in prices))
    profit_loss = {product: 0.0 for product in products}

    return BacktestData(
        round_num=round_num,
        day_num=day_num,
        prices=prices_by_timestamp,
        trades=trades_by_timestamp,
        products=products,
        profit_loss=profit_loss,
    )


def has_day_data(file_reader: FileReader, round_num: int, day_num: int) -> bool:
    with file_reader.file([f"round{round_num}", f"prices_round_{round_num}_day_{day_num}.csv"]) as file:
        return file is not None


def read_day_data(file_reader: FileReader, round_num: int, day_num: int) -> "BacktestData":
    # --- Prices file ---
    # Expected columns (semicolon-delimited, P4 format):
    #   day;timestamp;product;
    #   bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;
    #   ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;
    #   mid_price;profit_and_loss
    prices = []
    with file_reader.file([f"round{round_num}", f"prices_round_{round_num}_day_{day_num}.csv"]) as file:
        if file is None:
            raise ValueError(f"Prices data is not available for round {round_num} day {day_num}")

        raw = file.read_text(encoding="utf-8").splitlines()
        header = raw[0]
        required_cols = {"day", "timestamp", "product", "mid_price"}
        actual_cols = set(header.split(";"))
        missing = required_cols - actual_cols
        if missing:
            raise ValueError(f"Prices file is missing required columns: {missing}\nDetected: {header}")

        for line in raw[1:]:
            columns = line.split(";")

            prices.append(
                PriceRow(
                    day=int(columns[0]),
                    timestamp=int(columns[1]),
                    product=columns[2],
                    bid_prices=get_column_values(columns, [3, 5, 7]),
                    bid_volumes=get_column_values(columns, [4, 6, 8]),
                    ask_prices=get_column_values(columns, [9, 11, 13]),
                    ask_volumes=get_column_values(columns, [10, 12, 14]),
                    mid_price=float(columns[15]),
                    profit_loss=float(columns[16]),
                )
            )

    # --- Trades file ---
    # Expected columns (semicolon-delimited, P4 format):
    #   timestamp;buyer;seller;symbol;currency;price;quantity
    trades: list[Trade] = []
    with file_reader.file([f"round{round_num}", f"trades_round_{round_num}_day_{day_num}.csv"]) as file:
        if file is not None:
            raw_trades = file.read_text(encoding="utf-8").splitlines()
            header_t = raw_trades[0]
            required_trade_cols = {"timestamp", "symbol", "price", "quantity"}
            actual_trade_cols = set(header_t.split(";"))
            missing_t = required_trade_cols - actual_trade_cols
            if missing_t:
                raise ValueError(f"Trades file is missing required columns: {missing_t}\nDetected: {header_t}")

            for line in raw_trades[1:]:
                columns = line.split(";")

                trades.append(
                    Trade(
                        symbol=columns[3],
                        price=int(float(columns[5])),
                        quantity=int(float(columns[6])),
                        buyer=columns[1],
                        seller=columns[2],
                        timestamp=int(columns[0]),
                    )
                )

    return create_backtest_data(round_num, day_num, prices, trades)
