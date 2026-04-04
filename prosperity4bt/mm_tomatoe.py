from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import math
import json

TOMATOES_SYMBOL = "TOMATOES"
POSITION_LIMIT = 80


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class Trader:
    def __init__(self):
        self.orders: Dict[str, List[Order]] = {}
        self.traderData = ""

        # fair / alpha params
        self.WMID_L2_EDGE_COEF = 0.35
        self.OBI_2_COEF = 1.85
        self.BOOK_PRESSURE_GRADIENT2_COEF = -0.65
        self.MR_GAP_3_COEF = 0.08
        self.SHOCK_REVERSION_10_COEF = 0.01
        self.ALPHA_CLIP = 1.5

        # smooth alpha only
        self.ALPHA_SMOOTH_A = 0.94
        self.ALPHA_SMOOTH_B = 0.06
        self.ALPHA_SMOOTH_C = 0.0

        # disciplined execution
        self.TAKE_EDGE = 1.4
        self.QUOTE_EDGE = 2.0

        # alpha-weighted size
        self.MIN_SIZE = 3
        self.BASE_SIZE = 8
        self.MAX_SIZE = 18
        self.STRONG_ALPHA_THRESHOLD = 0.70  # normalized alpha strength

        # light inventory control
        self.POS_SLOWDOWN_START = 50
        self.POS_HARD_STOP = 75

    def load_data(self, state: TradingState) -> Dict:
        try:
            if state.traderData:
                return json.loads(state.traderData)
        except Exception:
            pass
        return {}

    def save_data(self, data: Dict) -> str:
        try:
            return json.dumps(data, separators=(",", ":"))
        except Exception:
            return ""

    def get_book(self, state: TradingState) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        depth = state.order_depths[TOMATOES_SYMBOL]
        bids = sorted(depth.buy_orders.items(), key=lambda x: -x[0])
        asks = sorted(depth.sell_orders.items(), key=lambda x: x[0])
        return bids, asks

    def get_mid(self, bids: List[Tuple[int, int]], asks: List[Tuple[int, int]]) -> float | None:
        if not bids or not asks:
            return None
        return (bids[0][0] + asks[0][0]) / 2.0

    def store_hist(self, data: Dict, key: str, value: float, keep: int) -> List[float]:
        hist = data.get(key, [])
        if not isinstance(hist, list):
            hist = []
        hist.append(float(value))
        if len(hist) > keep:
            hist = hist[-keep:]
        data[key] = hist
        return hist

    def compute_micro_l2(self, bids: List[Tuple[int, int]], asks: List[Tuple[int, int]], mid: float) -> float:
        bids2 = bids[:2]
        asks2 = asks[:2]

        if not bids2 or not asks2:
            return mid

        bid_vol = sum(v for _, v in bids2)
        ask_vol = sum(abs(v) for _, v in asks2)

        if bid_vol <= 0 or ask_vol <= 0:
            return mid

        bid_wavg = sum(p * v for p, v in bids2) / bid_vol
        ask_wavg = sum(p * abs(v) for p, v in asks2) / ask_vol

        return (ask_wavg * bid_vol + bid_wavg * ask_vol) / (bid_vol + ask_vol)

    def compute_wmid_l2_edge(self, bids: List[Tuple[int, int]], asks: List[Tuple[int, int]], mid: float) -> float:
        bids2 = bids[:2]
        asks2 = asks[:2]

        if not bids2 or not asks2:
            return 0.0

        den = sum(v for _, v in bids2) + sum(abs(v) for _, v in asks2)
        if den <= 0:
            return 0.0

        wmid = (
            sum(p * v for p, v in bids2) + sum(p * abs(v) for p, v in asks2)
        ) / den

        return wmid - mid

    def compute_obi_2(self, bids: List[Tuple[int, int]], asks: List[Tuple[int, int]]) -> float:
        bids2 = bids[:2]
        asks2 = asks[:2]

        bid_vol = sum(v for _, v in bids2)
        ask_vol = sum(abs(v) for _, v in asks2)
        den = bid_vol + ask_vol

        if den <= 0:
            return 0.0

        return (bid_vol - ask_vol) / den

    def compute_book_pressure_gradient2(self, bids: List[Tuple[int, int]], asks: List[Tuple[int, int]]) -> float:
        bids3 = bids[:3]
        asks3 = asks[:3]

        bid_pressure = sum((i + 1) * v for i, (_, v) in enumerate(bids3))
        ask_pressure = sum((i + 1) * abs(v) for i, (_, v) in enumerate(asks3))
        den = bid_pressure + ask_pressure

        if den <= 0:
            return 0.0

        return (bid_pressure - ask_pressure) / den

    def compute_mr_gap_3(self, data: Dict, mid: float) -> float:
        hist = self.store_hist(data, "mid_hist_3", mid, 3)
        if len(hist) < 3:
            return 0.0
        return mid - (sum(hist) / len(hist))

    def compute_shock_reversion_10(self, data: Dict, mid: float) -> float:
        hist = self.store_hist(data, "mid_hist_10", mid, 10)
        if len(hist) < 10:
            return 0.0
        return mid - (sum(hist) / len(hist))

    def smooth_alpha(self, data: Dict, raw_alpha: float) -> float:
        prev = data.get("smoothed_alpha")
        if prev is None:
            smoothed = raw_alpha
        else:
            smoothed = (
                self.ALPHA_SMOOTH_A * float(prev)
                + self.ALPHA_SMOOTH_B * float(raw_alpha)
                + self.ALPHA_SMOOTH_C
            )
        data["smoothed_alpha"] = smoothed
        return smoothed

    def compute_fair_and_alpha(self, state: TradingState, data: Dict) -> Tuple[float | None, float]:
        bids, asks = self.get_book(state)
        mid = self.get_mid(bids, asks)
        if mid is None:
            return None, 0.0

        micro_l2 = self.compute_micro_l2(bids, asks, mid)

        alpha_raw = (
            self.WMID_L2_EDGE_COEF * self.compute_wmid_l2_edge(bids, asks, mid)
            + self.OBI_2_COEF * self.compute_obi_2(bids, asks)
            + self.BOOK_PRESSURE_GRADIENT2_COEF * self.compute_book_pressure_gradient2(bids, asks)
            + self.MR_GAP_3_COEF * self.compute_mr_gap_3(data, mid)
            + self.SHOCK_REVERSION_10_COEF * self.compute_shock_reversion_10(data, mid)
        )

        alpha_clipped = math.tanh(alpha_raw) * self.ALPHA_CLIP
        alpha = self.smooth_alpha(data, alpha_clipped)

        fair = micro_l2 + alpha
        return fair, alpha

    def alpha_weighted_size(self, alpha: float, position: int) -> int:
        alpha_strength = min(1.0, abs(alpha) / max(self.ALPHA_CLIP, 1e-9))

        # continuous size scaling
        raw_size = self.MIN_SIZE + (self.MAX_SIZE - self.MIN_SIZE) * alpha_strength

        # light penalty when already carrying inventory
        pos_frac = abs(position) / POSITION_LIMIT
        if abs(position) >= self.POS_SLOWDOWN_START:
            raw_size *= max(0.35, 1.0 - 0.9 * pos_frac)

        size = int(round(raw_size))
        size = max(1, min(self.MAX_SIZE, size))
        return size

    def get_quote_prices(
        self,
        fair: float,
        bids: List[Tuple[int, int]],
        asks: List[Tuple[int, int]],
    ) -> Tuple[int, int]:
        best_bid = bids[0][0]
        best_ask = asks[0][0]

        buy_price = math.floor(fair - self.QUOTE_EDGE)
        sell_price = math.ceil(fair + self.QUOTE_EDGE)

        # step in front only if still clearly on the right side of fair
        if best_bid + 1 < fair:
            buy_price = max(buy_price, best_bid + 1)
        if best_ask - 1 > fair:
            sell_price = min(sell_price, best_ask - 1)

        # avoid crossing
        if buy_price >= best_ask:
            buy_price = best_bid
        if sell_price <= best_bid:
            sell_price = best_ask

        return int(buy_price), int(sell_price)

    def take_clear_edges(
        self,
        state: TradingState,
        fair: float,
        alpha: float,
        bids: List[Tuple[int, int]],
        asks: List[Tuple[int, int]],
    ) -> int:
        position = state.position.get(TOMATOES_SYMBOL, 0)
        remaining_buy = POSITION_LIMIT - position
        remaining_sell = POSITION_LIMIT + position

        size_cap = self.alpha_weighted_size(alpha, position)

        # buy clearly cheap asks
        for ask, vol in asks:
            ask_vol = abs(vol)
            if ask <= fair - self.TAKE_EDGE and remaining_buy > 0:
                size = min(ask_vol, remaining_buy, size_cap)
                if size > 0:
                    self.orders[TOMATOES_SYMBOL].append(Order(TOMATOES_SYMBOL, int(ask), int(size)))
                    position += size
                    remaining_buy -= size
                    remaining_sell += size
            else:
                break

        # sell clearly rich bids
        for bid, vol in bids:
            bid_vol = abs(vol)
            if bid >= fair + self.TAKE_EDGE and remaining_sell > 0:
                size = min(bid_vol, remaining_sell, size_cap)
                if size > 0:
                    self.orders[TOMATOES_SYMBOL].append(Order(TOMATOES_SYMBOL, int(bid), int(-size)))
                    position -= size
                    remaining_sell -= size
                    remaining_buy += size
            else:
                break

        return position

    def make_market(
        self,
        state: TradingState,
        fair: float,
        alpha: float,
        position: int,
        bids: List[Tuple[int, int]],
        asks: List[Tuple[int, int]],
    ) -> None:
        remaining_buy = POSITION_LIMIT - position
        remaining_sell = POSITION_LIMIT + position

        buy_price, sell_price = self.get_quote_prices(fair, bids, asks)
        size = self.alpha_weighted_size(alpha, position)

        # directional bias only through quote presence, not skew math
        # if alpha is positive, prioritize buy quote size a bit; if negative, sell quote size a bit
        alpha_strength = min(1.0, abs(alpha) / max(self.ALPHA_CLIP, 1e-9))

        buy_size = size
        sell_size = size

        if alpha > 0:
            buy_size = int(round(size * (1.0 + 0.35 * alpha_strength)))
            sell_size = int(round(size * (1.0 - 0.25 * alpha_strength)))
        elif alpha < 0:
            sell_size = int(round(size * (1.0 + 0.35 * alpha_strength)))
            buy_size = int(round(size * (1.0 - 0.25 * alpha_strength)))

        buy_size = max(1, min(self.MAX_SIZE, buy_size))
        sell_size = max(1, min(self.MAX_SIZE, sell_size))

        # hard inventory guardrails
        if position >= self.POS_HARD_STOP:
            buy_size = 0
            sell_size = max(sell_size, self.BASE_SIZE)
        elif position <= -self.POS_HARD_STOP:
            sell_size = 0
            buy_size = max(buy_size, self.BASE_SIZE)

        if remaining_buy > 0 and buy_size > 0:
            self.orders[TOMATOES_SYMBOL].append(
                Order(TOMATOES_SYMBOL, buy_price, min(buy_size, remaining_buy))
            )

        if remaining_sell > 0 and sell_size > 0:
            self.orders[TOMATOES_SYMBOL].append(
                Order(TOMATOES_SYMBOL, sell_price, -min(sell_size, remaining_sell))
            )

    def run(self, state: TradingState):
        self.orders = {TOMATOES_SYMBOL: []}
        data = self.load_data(state)

        if TOMATOES_SYMBOL not in state.order_depths:
            return self.orders, 0, self.save_data(data)

        bids, asks = self.get_book(state)
        if not bids or not asks:
            return self.orders, 0, self.save_data(data)

        fair, alpha = self.compute_fair_and_alpha(state, data)
        if fair is None:
            return self.orders, 0, self.save_data(data)

        position_after_taking = self.take_clear_edges(state, fair, alpha, bids, asks)
        self.make_market(state, fair, alpha, position_after_taking, bids, asks)

        return self.orders, 0, self.save_data(data)