from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any, Tuple
import json
import math

TOMATOES_SYMBOL = "TOMATOES"

POS_LIMITS = {
    TOMATOES_SYMBOL: 80
}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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


class TomatoesTrader(ProductTrader):
    # execution-first MM params
    TAKE_EDGE = 2.0
    MAKE_EDGE = 1.0

    BASE_SIZE = 8
    STRONG_SIZE = 16

    # stricter inventory control
    SOFT_POS = 30
    HARD_POS = 65
    INV_SKEW_PER_UNIT = 0.03
    MAX_SKEW = 6.0

    # keep only the alpha pieces that might matter
    OBI_2_COEF = 1.5
    BOOK_PRESSURE_GRADIENT2_COEF = -0.5
    ALPHA_CLIP = 1.2

    # lighter smoothing so alpha can move, but not dominate
    ALPHA_SMOOTH_A = 0.65
    ALPHA_SMOOTH_B = 0.35

    def get_mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    def compute_micro_l2(self) -> float | None:
        bids, asks = self.top_levels(2)
        mid = self.get_mid()
        if mid is None:
            return None
        if len(bids) == 0 or len(asks) == 0:
            return mid

        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(v for _, v in asks)
        if bid_vol <= 0 or ask_vol <= 0:
            return mid

        bid_wavg = sum(p * v for p, v in bids) / bid_vol
        ask_wavg = sum(p * v for p, v in asks) / ask_vol

        return (ask_wavg * bid_vol + bid_wavg * ask_vol) / (bid_vol + ask_vol)

    def compute_obi_2(self) -> float:
        bids, asks = self.top_levels(2)
        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(v for _, v in asks)
        den = bid_vol + ask_vol
        if den <= 0:
            return 0.0
        return (bid_vol - ask_vol) / den

    def compute_book_pressure_gradient2(self) -> float:
        bids = list(self.mkt_buy_orders.items())[:3]
        asks = list(self.mkt_sell_orders.items())[:3]

        bid_pressure = sum((i + 1) * v for i, (_, v) in enumerate(bids))
        ask_pressure = sum((i + 1) * v for i, (_, v) in enumerate(asks))
        den = bid_pressure + ask_pressure
        if den <= 0:
            return 0.0
        return (bid_pressure - ask_pressure) / den

    def smooth_alpha(self, raw_alpha: float) -> float:
        key = f"{self.name}_smoothed_alpha"
        prev_alpha = self.last_trader_data.get(key)
        if prev_alpha is None:
            smoothed = raw_alpha
        else:
            smoothed = self.ALPHA_SMOOTH_A * float(prev_alpha) + self.ALPHA_SMOOTH_B * float(raw_alpha)
        self.new_trader_data[key] = smoothed
        return smoothed

    def compute_alpha(self) -> float:
        alpha_raw = (
            self.OBI_2_COEF * self.compute_obi_2()
            + self.BOOK_PRESSURE_GRADIENT2_COEF * self.compute_book_pressure_gradient2()
        )
        alpha_clipped = math.tanh(alpha_raw) * self.ALPHA_CLIP
        return self.smooth_alpha(alpha_clipped)

    def compute_fair(self) -> float | None:
        mid = self.get_mid()
        if mid is None:
            return None

        micro_l2 = self.compute_micro_l2()
        if micro_l2 is None:
            micro_l2 = mid

        alpha = self.compute_alpha()
        fair = micro_l2 + alpha

        self.log("mid", mid)
        self.log("micro_l2", round(micro_l2, 4))
        self.log("alpha", round(alpha, 4))
        self.log("fair", round(fair, 4))

        return fair

    def get_inventory_skew(self, pos: int) -> float:
        return clamp(-pos * self.INV_SKEW_PER_UNIT, -self.MAX_SKEW, self.MAX_SKEW)

    def quote_sizes(self, pos: int, alpha_abs: float) -> Tuple[int, int]:
        # start with small base size
        buy_size = self.BASE_SIZE
        sell_size = self.BASE_SIZE

        # allow larger size only if alpha actually has some strength
        if alpha_abs >= 0.75:
            buy_size = self.STRONG_SIZE
            sell_size = self.STRONG_SIZE

        # flatten inventory aggressively
        if pos > self.SOFT_POS:
            buy_size = max(2, self.BASE_SIZE // 2)
            sell_size = self.STRONG_SIZE
        elif pos < -self.SOFT_POS:
            sell_size = max(2, self.BASE_SIZE // 2)
            buy_size = self.STRONG_SIZE

        # hard inventory behavior
        if pos >= self.HARD_POS:
            buy_size = 0
            sell_size = self.STRONG_SIZE
        elif pos <= -self.HARD_POS:
            sell_size = 0
            buy_size = self.STRONG_SIZE

        return buy_size, sell_size

    def get_orders(self) -> Dict[str, List[Order]]:
        fair = self.compute_fair()
        if fair is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        pos = self.initial_position
        alpha = self.last_trader_data.get(f"{self.name}_smoothed_alpha", 0.0)
        alpha_abs = abs(float(alpha))

        inv_skew = self.get_inventory_skew(pos)
        fair_adj = fair + inv_skew

        buy_size, sell_size = self.quote_sizes(pos, alpha_abs)

        # 1) selective taking — only take clearly stale liquidity
        for ask_price, ask_vol in self.mkt_sell_orders.items():
            if ask_price <= fair_adj - self.TAKE_EDGE and self.max_allowed_buy_volume > 0:
                take_size = min(ask_vol, buy_size, self.max_allowed_buy_volume)
                if take_size > 0:
                    self.bid(ask_price, take_size)
            else:
                break

        for bid_price, bid_vol in self.mkt_buy_orders.items():
            if bid_price >= fair_adj + self.TAKE_EDGE and self.max_allowed_sell_volume > 0:
                take_size = min(bid_vol, sell_size, self.max_allowed_sell_volume)
                if take_size > 0:
                    self.ask(bid_price, take_size)
            else:
                break

        # 2) passive quotes with stronger flattening bias
        bid_quote = min(self.best_bid + 1, math.floor(fair_adj - self.MAKE_EDGE))
        ask_quote = max(self.best_ask - 1, math.ceil(fair_adj + self.MAKE_EDGE))

        # if long, make bid less aggressive and ask more aggressive
        if pos > self.SOFT_POS:
            bid_quote -= 1
            ask_quote -= 1
        elif pos < -self.SOFT_POS:
            bid_quote += 1
            ask_quote += 1

        # avoid crossing
        if bid_quote >= self.best_ask:
            bid_quote = self.best_bid
        if ask_quote <= self.best_bid:
            ask_quote = self.best_ask

        if self.max_allowed_buy_volume > 0 and buy_size > 0:
            self.bid(int(bid_quote), min(buy_size, self.max_allowed_buy_volume))

        if self.max_allowed_sell_volume > 0 and sell_size > 0:
            self.ask(int(ask_quote), min(sell_size, self.max_allowed_sell_volume))

        return {self.name: self.orders}


class Trader:
    def run(self, state: TradingState):
        prints: Dict[str, Any] = {}
        new_trader_data: Dict[str, Any] = {}

        result: Dict[str, List[Order]] = {}

        tomatoes = TomatoesTrader(TOMATOES_SYMBOL, state, prints, new_trader_data)
        result.update(tomatoes.get_orders())

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data