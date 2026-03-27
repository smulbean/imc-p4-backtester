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


class TomatoesTrader(ProductTrader):
    # execution / risk knobs
    TAKE_EDGE = 1.0
    MAKE_EDGE = 1.0
    BASE_SIZE = 8
    STRONG_SIZE = 18

    SOFT_POS = 72
    HARD_POS = 79
    INV_SKEW_PER_UNIT = 0.009
    MAX_SKEW = 5.2

    # predictive alpha coefficients
    WMID_L2_EDGE_COEF = 0.35
    OBI_2_COEF = 1.85
    BOOK_PRESSURE_GRADIENT2_COEF = -0.65
    MR_GAP_3_COEF = 0.08
    SHOCK_REVERSION_10_COEF = 0.01

    ALPHA_CLIP = 2.0  # tanh scaled cap

    def get_mid(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    def _get_hist(self, key: str) -> List[float]:
        hist = self.last_trader_data.get(key, [])
        if isinstance(hist, list):
            return hist
        return []

    def _store_hist(self, key: str, value: float, keep: int) -> List[float]:
        hist = self._get_hist(key)
        hist.append(float(value))
        if len(hist) > keep:
            hist = hist[-keep:]
        self.new_trader_data[key] = hist
        return hist

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

        # cross-weighted microprice-style L2 anchor
        return (ask_wavg * bid_vol + bid_wavg * ask_vol) / (bid_vol + ask_vol)

    def compute_wmid_l2_edge(self) -> float:
        bids, asks = self.top_levels(2)
        mid = self.get_mid()
        if mid is None or len(bids) == 0 or len(asks) == 0:
            return 0.0

        den = sum(v for _, v in bids) + sum(v for _, v in asks)
        if den <= 0:
            return 0.0

        wmid = (sum(p * v for p, v in bids) + sum(p * v for p, v in asks)) / den
        return wmid - mid

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

    def compute_mr_gap_3(self, mid: float) -> float:
        hist = self._store_hist(f"{self.name}_mid_hist_3", mid, 3)
        if len(hist) < 3:
            return 0.0
        return mid - (sum(hist) / len(hist))

    def compute_shock_reversion_10(self, mid: float) -> float:
        hist = self._store_hist(f"{self.name}_mid_hist_10", mid, 10)
        if len(hist) < 10:
            return 0.0
        return mid - (sum(hist) / len(hist))

    def compute_fair(self) -> float | None:
        mid = self.get_mid()
        if mid is None:
            return None

        micro_l2 = self.compute_micro_l2()
        if micro_l2 is None:
            micro_l2 = mid

        alpha_raw = (
            self.WMID_L2_EDGE_COEF * self.compute_wmid_l2_edge()
            + self.OBI_2_COEF * self.compute_obi_2()
            + self.BOOK_PRESSURE_GRADIENT2_COEF * self.compute_book_pressure_gradient2()
            + self.MR_GAP_3_COEF * self.compute_mr_gap_3(mid)
            + self.SHOCK_REVERSION_10_COEF * self.compute_shock_reversion_10(mid)
        )

        alpha = math.tanh(alpha_raw) * self.ALPHA_CLIP
        fair = micro_l2 + alpha

        self.log("mid", mid)
        self.log("micro_l2", micro_l2)
        self.log("alpha_raw", round(alpha_raw, 4))
        self.log("alpha", round(alpha, 4))
        self.log("fair", round(fair, 4))

        return fair

    def quote_size(self) -> Tuple[int, int]:
        pos = self.initial_position

        buy_size = self.STRONG_SIZE if pos < -self.SOFT_POS else self.BASE_SIZE
        sell_size = self.STRONG_SIZE if pos > self.SOFT_POS else self.BASE_SIZE

        if pos >= self.HARD_POS:
            buy_size = 0
            sell_size = max(sell_size, self.STRONG_SIZE)
        if pos <= -self.HARD_POS:
            sell_size = 0
            buy_size = max(buy_size, self.STRONG_SIZE)

        return buy_size, sell_size

    def get_orders(self) -> Dict[str, List[Order]]:
        fair = self.compute_fair()
        if fair is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        pos = self.initial_position
        inv_skew = clamp(-pos * self.INV_SKEW_PER_UNIT, -self.MAX_SKEW, self.MAX_SKEW)
        fair_adj = fair + inv_skew

        buy_size, sell_size = self.quote_size()

        # 1) Take favorable displayed liquidity
        for ask_price, ask_vol in self.mkt_sell_orders.items():
            if ask_price <= fair_adj - self.TAKE_EDGE:
                self.bid(ask_price, min(ask_vol, max(buy_size, self.max_allowed_buy_volume)))
            else:
                break

        for bid_price, bid_vol in self.mkt_buy_orders.items():
            if bid_price >= fair_adj + self.TAKE_EDGE:
                self.ask(bid_price, min(bid_vol, max(sell_size, self.max_allowed_sell_volume)))
            else:
                break

        # 2) Make around predictive fair
        bid_quote = min(self.best_bid + 1, math.floor(fair_adj - self.MAKE_EDGE))
        ask_quote = max(self.best_ask - 1, math.ceil(fair_adj + self.MAKE_EDGE))

        # avoid crossing on passive quotes
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
