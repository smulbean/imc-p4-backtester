from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any, Tuple
import json
import math

EMERALDS_SYMBOL = "EMERALDS"

POS_LIMITS = {
    EMERALDS_SYMBOL: 80,
}


class ProductTrader:
    def __init__(self, name: str, state: TradingState, prints: Dict[str, Any], new_trader_data: Dict[str, Any]):
        self.orders: List[Order] = []
        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data

        self.last_trader_data = self.get_last_trader_data()
        self.position_limit = POS_LIMITS.get(self.name, 0)
        self.initial_position = self.state.position.get(self.name, 0)

        self.mkt_buy_orders, self.mkt_sell_orders = self.get_order_depth()
        self.best_bid, self.best_ask = self.get_best_bid_ask()

        self.max_allowed_buy_volume = self.position_limit - self.initial_position
        self.max_allowed_sell_volume = self.position_limit + self.initial_position

    def get_last_trader_data(self) -> Dict[str, Any]:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception:
            pass
        return {}

    def get_order_depth(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        buy_orders: Dict[int, int] = {}
        sell_orders: Dict[int, int] = {}
        try:
            depth: OrderDepth = self.state.order_depths[self.name]
            buy_orders = {p: abs(v) for p, v in sorted(depth.buy_orders.items(), key=lambda x: x[0], reverse=True)}
            sell_orders = {p: abs(v) for p, v in sorted(depth.sell_orders.items(), key=lambda x: x[0])}
        except Exception:
            pass
        return buy_orders, sell_orders

    def get_best_bid_ask(self) -> Tuple[int | None, int | None]:
        best_bid = max(self.mkt_buy_orders.keys()) if self.mkt_buy_orders else None
        best_ask = min(self.mkt_sell_orders.keys()) if self.mkt_sell_orders else None
        return best_bid, best_ask

    def log(self, key: str, value: Any) -> None:
        self.prints[key] = value

    def bid(self, price: int, volume: int) -> None:
        size = min(abs(int(volume)), self.max_allowed_buy_volume)
        if size <= 0:
            return
        self.orders.append(Order(self.name, int(price), size))
        self.max_allowed_buy_volume -= size

    def ask(self, price: int, volume: int) -> None:
        size = min(abs(int(volume)), self.max_allowed_sell_volume)
        if size <= 0:
            return
        self.orders.append(Order(self.name, int(price), -size))
        self.max_allowed_sell_volume -= size

    def top_levels(self, depth: int = 2) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = list(self.mkt_buy_orders.items())[:depth]
        asks = list(self.mkt_sell_orders.items())[:depth]
        return bids, asks


class EmeraldsTrader(ProductTrader):
    FAIR_VALUE = 10000
    MARKET_MAKE_SIZE = 8

    # Olivia overlay
    INFORMED_TRADER_ID = "Olivia"
    OLIVIA_BID_LEVEL = 9992
    OLIVIA_ASK_LEVEL = 10008
    OLIVIA_SIGNAL_TTL = 1200
    OLIVIA_QUOTE_STEP = 1
    OLIVIA_SIZE_BONUS = 4
    OLIVIA_FAIR_SHIFT = 1

    def __init__(self, state: TradingState, prints: Dict, new_trader_data: Dict):
        super().__init__(EMERALDS_SYMBOL, state, prints, new_trader_data)

    def _get_olivia_signal(self):
        """
        Track the most recent Olivia buy/sell timestamps and infer direction.
        Returns: (direction, last_buy_ts, last_sell_ts)
        direction in {"LONG", "SHORT", "NEUTRAL"}
        """
        # fixed: last_trader_data, not last_traderData
        last_buy_ts, last_sell_ts = self.last_trader_data.get(
            f"{self.name}_olivia", [None, None]
        )

        trades = []
        trades += self.state.market_trades.get(self.name, [])
        trades += self.state.own_trades.get(self.name, [])

        for trade in trades:
            if getattr(trade, "buyer", None) == self.INFORMED_TRADER_ID:
                last_buy_ts = trade.timestamp
            if getattr(trade, "seller", None) == self.INFORMED_TRADER_ID:
                last_sell_ts = trade.timestamp

        now = self.state.timestamp

        # Expire stale signal before saving / using it
        if last_buy_ts is not None and now - last_buy_ts > self.OLIVIA_SIGNAL_TTL:
            last_buy_ts = None
        if last_sell_ts is not None and now - last_sell_ts > self.OLIVIA_SIGNAL_TTL:
            last_sell_ts = None

        self.new_trader_data[f"{self.name}_olivia"] = [last_buy_ts, last_sell_ts]

        if last_buy_ts is None and last_sell_ts is None:
            return "NEUTRAL", last_buy_ts, last_sell_ts
        if last_buy_ts is not None and last_sell_ts is None:
            return "LONG", last_buy_ts, last_sell_ts
        if last_buy_ts is None and last_sell_ts is not None:
            return "SHORT", last_buy_ts, last_sell_ts

        if last_buy_ts > last_sell_ts:
            return "LONG", last_buy_ts, last_sell_ts
        if last_sell_ts > last_buy_ts:
            return "SHORT", last_buy_ts, last_sell_ts
        return "NEUTRAL", last_buy_ts, last_sell_ts

    def get_orders(self):
        fair = self.FAIR_VALUE

        # 0. Olivia overlay
        olivia_direction, olivia_buy_ts, olivia_sell_ts = self._get_olivia_signal()

        adjusted_fair = fair
        if olivia_direction == "LONG":
            adjusted_fair += self.OLIVIA_FAIR_SHIFT
        elif olivia_direction == "SHORT":
            adjusted_fair -= self.OLIVIA_FAIR_SHIFT

        # 1. Take obvious edge
        for ask_price, ask_volume in self.mkt_sell_orders.items():
            if ask_price < adjusted_fair:
                self.bid(int(ask_price), ask_volume)
            elif ask_price == adjusted_fair and self.initial_position < 0:
                self.bid(int(ask_price), min(ask_volume, -self.initial_position))

        for bid_price, bid_volume in self.mkt_buy_orders.items():
            if bid_price > adjusted_fair:
                self.ask(int(bid_price), bid_volume)
            elif bid_price == adjusted_fair and self.initial_position > 0:
                self.ask(int(bid_price), min(bid_volume, self.initial_position))

        # 2. Passive market making
        if self.best_bid is not None:
            buy_quote = min(adjusted_fair - 1, self.best_bid + 1)
        else:
            buy_quote = adjusted_fair - 1

        if self.best_ask is not None:
            sell_quote = max(adjusted_fair + 1, self.best_ask - 1)
        else:
            sell_quote = adjusted_fair + 1

        if buy_quote >= sell_quote:
            buy_quote = adjusted_fair - 1
            sell_quote = adjusted_fair + 1

        # 3. Olivia-specific passive skew
        if olivia_direction == "LONG":
            olivia_buy_quote = self.OLIVIA_BID_LEVEL + self.OLIVIA_QUOTE_STEP
            buy_quote = max(buy_quote, olivia_buy_quote)
            sell_quote = max(sell_quote, adjusted_fair + 1)

        elif olivia_direction == "SHORT":
            olivia_sell_quote = self.OLIVIA_ASK_LEVEL - self.OLIVIA_QUOTE_STEP
            sell_quote = min(sell_quote, olivia_sell_quote)
            buy_quote = min(buy_quote, adjusted_fair - 1)

        if buy_quote >= sell_quote:
            buy_quote = adjusted_fair - 1
            sell_quote = adjusted_fair + 1

        # 4. Size logic
        buy_size = min(self.MARKET_MAKE_SIZE, self.max_allowed_buy_volume)
        sell_size = min(self.MARKET_MAKE_SIZE, self.max_allowed_sell_volume)

        if self.initial_position < 0:
            buy_size = min(
                self.max_allowed_buy_volume,
                self.MARKET_MAKE_SIZE + abs(self.initial_position) // 4
            )
        elif self.initial_position > 0:
            sell_size = min(
                self.max_allowed_sell_volume,
                self.MARKET_MAKE_SIZE + abs(self.initial_position) // 4
            )

        if olivia_direction == "LONG":
            buy_size = min(self.max_allowed_buy_volume, buy_size + self.OLIVIA_SIZE_BONUS)
        elif olivia_direction == "SHORT":
            sell_size = min(self.max_allowed_sell_volume, sell_size + self.OLIVIA_SIZE_BONUS)

        # 5. Place passive orders
        if buy_size > 0:
            self.bid(int(buy_quote), buy_size)

        if sell_size > 0:
            self.ask(int(sell_quote), sell_size)

        # 6. Logging
        self.log("FAIR", fair)
        self.log("ADJ_FAIR", adjusted_fair)
        self.log("best_bid", self.best_bid)
        self.log("best_ask", self.best_ask)
        self.log("OLIVIA_DIR", olivia_direction)
        self.log("OLIVIA_BUY_TS", olivia_buy_ts)
        self.log("OLIVIA_SELL_TS", olivia_sell_ts)
        self.log("BUY_QUOTE", buy_quote)
        self.log("SELL_QUOTE", sell_quote)

        return {self.name: self.orders}


class Trader:
    def run(self, state: TradingState):
        prints: Dict[str, Any] = {}
        new_trader_data: Dict[str, Any] = {}

        result: Dict[str, List[Order]] = {}

        emeralds = EmeraldsTrader(state, prints, new_trader_data)
        result.update(emeralds.get_orders())

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data