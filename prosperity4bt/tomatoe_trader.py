from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any, Tuple, Optional
import json
import math

TOMATOES_SYMBOL = "TOMATOES"
EMERALDS_SYMBOL = "EMERALDS"

POS_LIMITS = {
    TOMATOES_SYMBOL: 80,
    EMERALDS_SYMBOL: 80,
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class ProductTrader:
    def __init__(
        self,
        name: str,
        state: TradingState,
        prints: Dict[str, Any],
        new_trader_data: Dict[str, Any],
    ):
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
            buy_orders = {
                p: abs(v)
                for p, v in sorted(depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
            }
            sell_orders = {
                p: abs(v)
                for p, v in sorted(depth.sell_orders.items(), key=lambda x: x[0])
            }
        except Exception:
            pass
        return buy_orders, sell_orders

    def get_best_bid_ask(self) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(self.mkt_buy_orders.keys()) if self.mkt_buy_orders else None
        best_ask = min(self.mkt_sell_orders.keys()) if self.mkt_sell_orders else None
        return best_bid, best_ask

    def log(self, key: str, value: Any) -> None:
        self.prints[f"{self.name}_{key}"] = value

    def bid(self, price: int, volume: int, logging: bool = True) -> None:
        size = min(abs(int(volume)), self.max_allowed_buy_volume)
        if size <= 0:
            return
        self.orders.append(Order(self.name, int(price), size))
        self.max_allowed_buy_volume -= size

    def ask(self, price: int, volume: int, logging: bool = True) -> None:
        size = min(abs(int(volume)), self.max_allowed_sell_volume)
        if size <= 0:
            return
        self.orders.append(Order(self.name, int(price), -size))
        self.max_allowed_sell_volume -= size

    def top_levels(self, depth: int = 2) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = list(self.mkt_buy_orders.items())[:depth]
        asks = list(self.mkt_sell_orders.items())[:depth]
        return bids, asks


class TomatoesTrader(ProductTrader):
    # Core taking / making
    BASE_TAKE_EDGE = 0.9
    STRONG_TAKE_EDGE = 0.35

    BASE_SIZE = 12
    STRONG_SIZE = 24
    MAX_TAKE_SIZE = 28

    # Inventory
    SOFT_POS = 42
    HARD_POS = 75
    INV_SKEW_PER_UNIT = 0.022
    MAX_SKEW = 5.0

    # Spread-aware making
    MIN_MAKE_EDGE = 0.0
    MAX_MAKE_EDGE = 1.2
    SPREAD_EDGE_MULT = 0.12

    # Alpha model
    IMB_ALPHA_COEF = 1.4
    MOM_ALPHA_COEF = 0.75
    ALPHA_SMOOTH = 0.82
    ALPHA_CLIP = 2.5

    # Strong-alpha thresholds
    STRONG_ALPHA = 0.9
    VERY_STRONG_ALPHA = 1.6

    def __init__(self, state: TradingState, prints: Dict[str, Any], new_trader_data: Dict[str, Any]):
        super().__init__(TOMATOES_SYMBOL, state, prints, new_trader_data)

    def get_mid(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    def compute_micro_l2(self) -> Optional[float]:
        bids, asks = self.top_levels(2)
        mid = self.get_mid()
        if mid is None:
            return None
        if not bids or not asks:
            return mid

        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(v for _, v in asks)
        if bid_vol <= 0 or ask_vol <= 0:
            return mid

        bid_wavg = sum(p * v for p, v in bids) / bid_vol
        ask_wavg = sum(p * v for p, v in asks) / ask_vol

        # Standard microprice-style weighted cross
        return (ask_wavg * bid_vol + bid_wavg * ask_vol) / (bid_vol + ask_vol)

    def compute_imbalance(self) -> float:
        bids, asks = self.top_levels(2)
        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(v for _, v in asks)
        total = bid_vol + ask_vol
        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def compute_alpha(self) -> float:
        mid = self.get_mid()
        if mid is None:
            prev_alpha = self.last_trader_data.get("tomatoes_alpha", 0.0)
            return float(prev_alpha)

        imbalance = self.compute_imbalance()
        prev_mid = self.last_trader_data.get("tomatoes_mid")
        prev_alpha = float(self.last_trader_data.get("tomatoes_alpha", 0.0))

        momentum = 0.0
        if prev_mid is not None:
            momentum = mid - float(prev_mid)

        alpha_raw = self.IMB_ALPHA_COEF * imbalance + self.MOM_ALPHA_COEF * momentum
        alpha_smooth = self.ALPHA_SMOOTH * prev_alpha + (1.0 - self.ALPHA_SMOOTH) * alpha_raw
        alpha = clamp(alpha_smooth, -self.ALPHA_CLIP, self.ALPHA_CLIP)

        self.new_trader_data["tomatoes_mid"] = mid
        self.new_trader_data["tomatoes_alpha"] = alpha

        self.log("mid", round(mid, 4))
        self.log("imbalance", round(imbalance, 4))
        self.log("momentum", round(momentum, 4))
        self.log("alpha_raw", round(alpha_raw, 4))
        self.log("alpha", round(alpha, 4))

        return alpha

    def compute_fair(self) -> Optional[float]:
        micro = self.compute_micro_l2()
        if micro is None:
            return None

        alpha = self.compute_alpha()
        fair = micro + alpha

        self.log("micro_l2", round(micro, 4))
        self.log("fair", round(fair, 4))
        self.log("position", self.initial_position)
        return fair

    def get_inventory_skew(self, pos: int) -> float:
        return clamp(-pos * self.INV_SKEW_PER_UNIT, -self.MAX_SKEW, self.MAX_SKEW)

    def quote_sizes(self, pos: int, alpha: float) -> Tuple[int, int]:
        buy_size = self.BASE_SIZE
        sell_size = self.BASE_SIZE

        # inventory shaping
        if pos > self.SOFT_POS:
            buy_size = max(3, self.BASE_SIZE // 2)
            sell_size = self.STRONG_SIZE
        elif pos < -self.SOFT_POS:
            sell_size = max(3, self.BASE_SIZE // 2)
            buy_size = self.STRONG_SIZE

        if pos >= self.HARD_POS:
            buy_size = 0
            sell_size = self.STRONG_SIZE
        elif pos <= -self.HARD_POS:
            sell_size = 0
            buy_size = self.STRONG_SIZE

        # alpha-weighted asymmetry while staying MM-ish
        if alpha > self.STRONG_ALPHA:
            buy_size = min(self.STRONG_SIZE, max(buy_size, self.BASE_SIZE + 6))
            sell_size = max(3, sell_size - 4)
        elif alpha < -self.STRONG_ALPHA:
            sell_size = min(self.STRONG_SIZE, max(sell_size, self.BASE_SIZE + 6))
            buy_size = max(3, buy_size - 4)

        return buy_size, sell_size

    def compute_make_edge(self) -> float:
        if self.best_bid is None or self.best_ask is None:
            return self.MIN_MAKE_EDGE
        spread = self.best_ask - self.best_bid
        edge = clamp(spread * self.SPREAD_EDGE_MULT, self.MIN_MAKE_EDGE, self.MAX_MAKE_EDGE)
        self.log("spread", spread)
        self.log("make_edge", round(edge, 4))
        return edge

    def compute_take_edge(self, alpha: float) -> float:
        if abs(alpha) >= self.VERY_STRONG_ALPHA:
            return self.STRONG_TAKE_EDGE
        if abs(alpha) >= self.STRONG_ALPHA:
            return 0.55
        return self.BASE_TAKE_EDGE

    def take_sizes(self, alpha: float, passive_size: int) -> int:
        if abs(alpha) >= self.VERY_STRONG_ALPHA:
            return min(self.MAX_TAKE_SIZE, max(passive_size, self.STRONG_SIZE))
        if abs(alpha) >= self.STRONG_ALPHA:
            return min(self.MAX_TAKE_SIZE, max(passive_size, self.BASE_SIZE + 8))
        return min(self.MAX_TAKE_SIZE, max(passive_size, self.BASE_SIZE))

    def get_orders(self) -> Dict[str, List[Order]]:
        fair = self.compute_fair()
        if fair is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        alpha = float(self.new_trader_data.get("tomatoes_alpha", self.last_trader_data.get("tomatoes_alpha", 0.0)))
        pos = self.initial_position

        inv_skew = self.get_inventory_skew(pos)
        fair_adj = fair + inv_skew

        buy_size, sell_size = self.quote_sizes(pos, alpha)
        make_edge = self.compute_make_edge()
        take_edge = self.compute_take_edge(alpha)
        spread = self.best_ask - self.best_bid

        # More aggressive taking when alpha agrees
        max_take_buy = self.take_sizes(alpha, buy_size)
        max_take_sell = self.take_sizes(alpha, sell_size)

        for ask_price, ask_vol in self.mkt_sell_orders.items():
            edge_to_buy = fair_adj - ask_price
            bullish_boost = max(0.0, alpha)
            threshold = take_edge - 0.35 * bullish_boost
            if edge_to_buy >= threshold and self.max_allowed_buy_volume > 0:
                take_size = min(ask_vol, max_take_buy, self.max_allowed_buy_volume)
                if take_size > 0:
                    self.bid(ask_price, take_size)
            else:
                break

        for bid_price, bid_vol in self.mkt_buy_orders.items():
            edge_to_sell = bid_price - fair_adj
            bearish_boost = max(0.0, -alpha)
            threshold = take_edge - 0.35 * bearish_boost
            if edge_to_sell >= threshold and self.max_allowed_sell_volume > 0:
                take_size = min(bid_vol, max_take_sell, self.max_allowed_sell_volume)
                if take_size > 0:
                    self.ask(bid_price, take_size)
            else:
                break

        # Recompute fair limits after takes
        bid_fair_limit = math.floor(fair_adj - make_edge)
        ask_fair_limit = math.ceil(fair_adj + make_edge)

        # Always try to quote competitively first
        bid_quote = self.best_bid + 1
        ask_quote = self.best_ask - 1

        # Clamp to fair limits
        bid_quote = min(bid_quote, bid_fair_limit)
        ask_quote = max(ask_quote, ask_fair_limit)

        # If spread is tight, still keep quoting near fair
        if spread <= 1:
            bid_quote = min(self.best_bid, bid_fair_limit)
            ask_quote = max(self.best_ask, ask_fair_limit)

        # Alpha tilt: push stronger side slightly, but keep MM structure
        if alpha > self.STRONG_ALPHA:
            bid_quote += 1
        elif alpha < -self.STRONG_ALPHA:
            ask_quote -= 1

        # Inventory tilt
        if pos > self.SOFT_POS:
            bid_quote -= 1
            ask_quote -= 1
        elif pos < -self.SOFT_POS:
            bid_quote += 1
            ask_quote += 1

        # Hard position defense
        if pos >= self.HARD_POS:
            bid_quote = None
            ask_quote = max(self.best_bid + 1, self.best_ask - 1)
        elif pos <= -self.HARD_POS:
            ask_quote = None
            bid_quote = min(self.best_ask - 1, self.best_bid + 1)

        if bid_quote is not None:
            bid_quote = min(bid_quote, self.best_ask - 1)
        if ask_quote is not None:
            ask_quote = max(ask_quote, self.best_bid + 1)

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            # Fall back safely to inside/outside touch
            bid_quote = min(self.best_bid, bid_fair_limit)
            ask_quote = max(self.best_ask, ask_fair_limit)
            if bid_quote >= ask_quote:
                bid_quote = self.best_bid
                ask_quote = self.best_ask

        self.log("fair_adj", round(fair_adj, 4))
        self.log("inv_skew", round(inv_skew, 4))
        self.log("take_edge", round(take_edge, 4))
        self.log("bid_fair_limit", bid_fair_limit)
        self.log("ask_fair_limit", ask_fair_limit)
        self.log("bid_quote", bid_quote)
        self.log("ask_quote", ask_quote)
        self.log("buy_size", buy_size)
        self.log("sell_size", sell_size)

        if bid_quote is not None and self.max_allowed_buy_volume > 0 and buy_size > 0:
            self.bid(int(bid_quote), min(buy_size, self.max_allowed_buy_volume))

        if ask_quote is not None and self.max_allowed_sell_volume > 0 and sell_size > 0:
            self.ask(int(ask_quote), min(sell_size, self.max_allowed_sell_volume))

        return {self.name: self.orders}


class EmeraldsTrader(ProductTrader):
    FAIR_VALUE = 10000
    MARKET_MAKE_SIZE = 8

    def __init__(self, state: TradingState, prints: Dict[str, Any], new_trader_data: Dict[str, Any]):
        super().__init__(EMERALDS_SYMBOL, state, prints, new_trader_data)

    def get_orders(self) -> Dict[str, List[Order]]:
        fair = self.FAIR_VALUE

        for ask_price, ask_volume in self.mkt_sell_orders.items():
            if ask_price < fair:
                self.bid(ask_price, ask_volume, logging=False)
            elif ask_price == fair and self.initial_position < 0:
                self.bid(ask_price, min(ask_volume, -self.initial_position), logging=False)

        for bid_price, bid_volume in self.mkt_buy_orders.items():
            if bid_price > fair:
                self.ask(bid_price, bid_volume, logging=False)
            elif bid_price == fair and self.initial_position > 0:
                self.ask(bid_price, min(bid_volume, self.initial_position), logging=False)

        if self.best_bid is not None:
            buy_quote = min(fair - 1, self.best_bid + 1)
        else:
            buy_quote = fair - 1

        if self.best_ask is not None:
            sell_quote = max(fair + 1, self.best_ask - 1)
        else:
            sell_quote = fair + 1

        if buy_quote >= sell_quote:
            buy_quote = fair - 1
            sell_quote = fair + 1

        buy_size = min(self.MARKET_MAKE_SIZE, self.max_allowed_buy_volume)
        sell_size = min(self.MARKET_MAKE_SIZE, self.max_allowed_sell_volume)

        if self.initial_position < 0:
            buy_size = min(self.max_allowed_buy_volume, self.MARKET_MAKE_SIZE + abs(self.initial_position) // 4)
        elif self.initial_position > 0:
            sell_size = min(self.max_allowed_sell_volume, self.MARKET_MAKE_SIZE + abs(self.initial_position) // 4)

        if buy_size > 0:
            self.bid(buy_quote, buy_size)

        if sell_size > 0:
            self.ask(sell_quote, sell_size)

        self.log("FAIR", fair)
        self.log("best_bid", self.best_bid)
        self.log("best_ask", self.best_ask)

        return {self.name: self.orders}


class Trader:
    def run(self, state: TradingState):
        prints: Dict[str, Any] = {}
        new_trader_data: Dict[str, Any] = {}

        result: Dict[str, List[Order]] = {}

        tomatoes = TomatoesTrader(state, prints, new_trader_data)
        result.update(tomatoes.get_orders())

        emeralds = EmeraldsTrader(state, prints, new_trader_data)
        result.update(emeralds.get_orders())

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data