from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Any, Tuple
import json
import math

TOMATOES_SYMBOL = "TOMATOES"

POS_LIMITS = {
    TOMATOES_SYMBOL: 80,
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

    def push_debug(self, msg: str) -> None:
        dbg = self.prints.setdefault("debug", [])
        if len(dbg) < 30:
            dbg.append(msg)

    def bid(self, price: int, volume: int, reason: str = "") -> None:
        size = min(abs(int(volume)), self.max_allowed_buy_volume)
        if size <= 0:
            return
        self.orders.append(Order(self.name, int(price), size))
        self.max_allowed_buy_volume -= size
        if reason:
            self.push_debug(f"BUY  {size:>2} @ {price} | {reason}")

    def ask(self, price: int, volume: int, reason: str = "") -> None:
        size = min(abs(int(volume)), self.max_allowed_sell_volume)
        if size <= 0:
            return
        self.orders.append(Order(self.name, int(price), -size))
        self.max_allowed_sell_volume -= size
        if reason:
            self.push_debug(f"SELL {size:>2} @ {price} | {reason}")

    def top_levels(self, depth: int = 2) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        bids = list(self.mkt_buy_orders.items())[:depth]
        asks = list(self.mkt_sell_orders.items())[:depth]
        return bids, asks


class TomatoesTrader(ProductTrader):
    # Based on robustness: taking matters more than quote tuning
    TAKE_EDGE = 1.25

    BASE_SIZE = 12
    STRONG_SIZE = 22

    # Loosen back a bit: tighter inventory did not help this version
    SOFT_POS = 40
    HARD_POS = 76
    INV_SKEW_PER_UNIT = 0.022
    MAX_SKEW = 5.0

    # Quote logic: keep simple since sweep shows weak sensitivity
    MIN_MAKE_EDGE = 0.0
    MAX_MAKE_EDGE = 1.2
    SPREAD_EDGE_MULT = 0.18

    # Taking controls
    TAKE_SIZE_CAP_MULT = 1.60
    EXTRA_TAKE_IF_SPREAD_WIDE = 0.20
    WIDE_SPREAD = 4

    # Quoting controls
    IMPROVE_IF_SPREAD_AT_LEAST = 3
    IMPROVE_ONLY_WITH_EDGE_ROOM = 0.10


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

    def compute_fair(self) -> float | None:
        fair = self.compute_micro_l2()
        if fair is None:
            return None
        self.log("fair", round(fair, 4))
        self.log("position", self.initial_position)
        return fair

    def get_inventory_skew(self, pos: int) -> float:
        return clamp(-pos * self.INV_SKEW_PER_UNIT, -self.MAX_SKEW, self.MAX_SKEW)

    def quote_sizes(self, pos: int) -> Tuple[int, int]:
        buy_size = self.BASE_SIZE
        sell_size = self.BASE_SIZE

        if pos > self.SOFT_POS:
            buy_size = max(4, self.BASE_SIZE // 2)
            sell_size = self.STRONG_SIZE
        elif pos < -self.SOFT_POS:
            sell_size = max(4, self.BASE_SIZE // 2)
            buy_size = self.STRONG_SIZE

        if pos >= self.HARD_POS:
            buy_size = 0
            sell_size = self.STRONG_SIZE
        elif pos <= -self.HARD_POS:
            sell_size = 0
            buy_size = self.STRONG_SIZE

        return buy_size, sell_size

    def compute_make_edge(self) -> float:
        if self.best_bid is None or self.best_ask is None:
            return self.MIN_MAKE_EDGE
        spread = self.best_ask - self.best_bid
        edge = clamp(spread * self.SPREAD_EDGE_MULT, self.MIN_MAKE_EDGE, self.MAX_MAKE_EDGE)
        self.log("spread", spread)
        self.log("make_edge", round(edge, 4))
        return edge

    def get_orders(self) -> Dict[str, List[Order]]:
        fair = self.compute_fair()
        if fair is None or self.best_bid is None or self.best_ask is None:
            return {self.name: self.orders}

        pos = self.initial_position
        mid = self.get_mid()
        inv_skew = self.get_inventory_skew(pos)
        fair_adj = fair + inv_skew
        spread = self.best_ask - self.best_bid

        buy_size, sell_size = self.quote_sizes(pos)
        make_edge = self.compute_make_edge()

        self.log("timestamp", getattr(self.state, "timestamp", None))
        self.log("best_bid", self.best_bid)
        self.log("best_ask", self.best_ask)
        self.log("mid", mid)
        self.log("inv_skew", round(inv_skew, 4))
        self.log("fair_adj", round(fair_adj, 4))
        self.log("buy_size", buy_size)
        self.log("sell_size", sell_size)
        self.log("max_buy_left", self.max_allowed_buy_volume)
        self.log("max_sell_left", self.max_allowed_sell_volume)

        # ----------------------------------------------------------
        # 1) Selective taking: stronger because robustness says it helps
        # ----------------------------------------------------------
        take_buy_cap = max(self.BASE_SIZE, int(buy_size * self.TAKE_SIZE_CAP_MULT))
        take_sell_cap = max(self.BASE_SIZE, int(sell_size * self.TAKE_SIZE_CAP_MULT))

        dynamic_take_edge = self.TAKE_EDGE
        if spread >= self.WIDE_SPREAD:
            dynamic_take_edge -= self.EXTRA_TAKE_IF_SPREAD_WIDE

        took_buy = 0
        took_sell = 0

        for ask_price, ask_vol in self.mkt_sell_orders.items():
            edge = fair_adj - ask_price
            if edge >= dynamic_take_edge and self.max_allowed_buy_volume > 0:
                take_size = min(ask_vol, take_buy_cap - took_buy, self.max_allowed_buy_volume)
                if take_size > 0:
                    self.bid(
                        ask_price,
                        take_size,
                        reason=f"TAKE_ASK edge={edge:.2f} fair_adj={fair_adj:.2f}"
                    )
                    took_buy += take_size
            else:
                break

        for bid_price, bid_vol in self.mkt_buy_orders.items():
            edge = bid_price - fair_adj
            if edge >= dynamic_take_edge and self.max_allowed_sell_volume > 0:
                take_size = min(bid_vol, take_sell_cap - took_sell, self.max_allowed_sell_volume)
                if take_size > 0:
                    self.ask(
                        bid_price,
                        take_size,
                        reason=f"TAKE_BID edge={edge:.2f} fair_adj={fair_adj:.2f}"
                    )
                    took_sell += take_size
            else:
                break

        self.log("dynamic_take_edge", round(dynamic_take_edge, 4))
        self.log("took_buy", took_buy)
        self.log("took_sell", took_sell)

        # ----------------------------------------------------------
        # 2) Base quote limits around adjusted fair
        # ----------------------------------------------------------
        bid_fair_limit = math.floor(fair_adj - make_edge)
        ask_fair_limit = math.ceil(fair_adj + make_edge)

        bid_quote = self.best_bid if self.best_bid <= bid_fair_limit else bid_fair_limit
        ask_quote = self.best_ask if self.best_ask >= ask_fair_limit else ask_fair_limit

        bid_reason = "JOIN" if self.best_bid <= bid_fair_limit else "FAIR_LIMIT"
        ask_reason = "JOIN" if self.best_ask >= ask_fair_limit else "FAIR_LIMIT"

        # ----------------------------------------------------------
        # 3) Improve by one tick only when wide spread and enough room
        # ----------------------------------------------------------
        improved_bid = self.best_bid + 1
        improved_ask = self.best_ask - 1

        bid_room = bid_fair_limit - improved_bid
        ask_room = improved_ask - ask_fair_limit

        if spread >= self.IMPROVE_IF_SPREAD_AT_LEAST and bid_room >= self.IMPROVE_ONLY_WITH_EDGE_ROOM and pos <= self.SOFT_POS:
            bid_quote = improved_bid
            bid_reason = "IMPROVE_QUEUE"

        if spread >= self.IMPROVE_IF_SPREAD_AT_LEAST and ask_room >= self.IMPROVE_ONLY_WITH_EDGE_ROOM and pos >= -self.SOFT_POS:
            ask_quote = improved_ask
            ask_reason = "IMPROVE_QUEUE"

        # ----------------------------------------------------------
        # 4) Inventory flattening
        # ----------------------------------------------------------
        if pos > self.SOFT_POS:
            bid_quote -= 1
            ask_quote -= 1
            bid_reason += "|FLAT_LONG"
            ask_reason += "|FLAT_LONG"
        elif pos < -self.SOFT_POS:
            bid_quote += 1
            ask_quote += 1
            bid_reason += "|FLAT_SHORT"
            ask_reason += "|FLAT_SHORT"

        # ----------------------------------------------------------
        # 5) Hard-cap handling
        # ----------------------------------------------------------
        if pos >= self.HARD_POS:
            bid_quote = None
            ask_quote = max(self.best_bid + 1, self.best_ask - 1)
            bid_reason = "BLOCK_HARD_LONG"
            ask_reason = "HARD_FLAT_LONG"
        elif pos <= -self.HARD_POS:
            ask_quote = None
            bid_quote = min(self.best_ask - 1, self.best_bid + 1)
            ask_reason = "BLOCK_HARD_SHORT"
            bid_reason = "HARD_FLAT_SHORT"

        # ----------------------------------------------------------
        # 6) Non-crossing guards
        # ----------------------------------------------------------
        if bid_quote is not None and bid_quote >= self.best_ask:
            bid_quote = self.best_ask - 1
            bid_reason += "|NON_CROSS"
        if ask_quote is not None and ask_quote <= self.best_bid:
            ask_quote = self.best_bid + 1
            ask_reason += "|NON_CROSS"

        if bid_quote is not None and ask_quote is not None and bid_quote >= ask_quote:
            bid_quote = min(bid_quote, self.best_bid)
            ask_quote = max(ask_quote, self.best_ask)
            bid_reason += "|FINAL_GUARD"
            ask_reason += "|FINAL_GUARD"

        self.log("bid_fair_limit", bid_fair_limit)
        self.log("ask_fair_limit", ask_fair_limit)
        self.log("bid_quote", bid_quote)
        self.log("ask_quote", ask_quote)
        self.log("bid_reason", bid_reason)
        self.log("ask_reason", ask_reason)

        # ----------------------------------------------------------
        # 7) Post passive quotes
        # ----------------------------------------------------------
        if bid_quote is not None and self.max_allowed_buy_volume > 0 and buy_size > 0:
            self.bid(
                int(bid_quote),
                min(buy_size, self.max_allowed_buy_volume),
                reason=f"QUOTE_BID {bid_reason}"
            )

        if ask_quote is not None and self.max_allowed_sell_volume > 0 and sell_size > 0:
            self.ask(
                int(ask_quote),
                min(sell_size, self.max_allowed_sell_volume),
                reason=f"QUOTE_ASK {ask_reason}"
            )

        # summary/debug
        self.log("orders_sent", [(o.symbol, o.price, o.quantity) for o in self.orders])
        self.log("summary", {
            "pos": pos,
            "spread": spread,
            "fair": round(fair, 3),
            "fair_adj": round(fair_adj, 3),
            "took_buy": took_buy,
            "took_sell": took_sell,
            "bid_q": bid_quote,
            "ask_q": ask_quote,
        })

        return {self.name: self.orders}


class Trader:
    def run(self, state: TradingState):
        prints: Dict[str, Any] = {}
        new_trader_data: Dict[str, Any] = {}

        result: Dict[str, List[Order]] = {}
        tomatoes = TomatoesTrader(TOMATOES_SYMBOL, state, prints, new_trader_data)
        result.update(tomatoes.get_orders())

        # Uncomment for raw stdout if your harness shows it cleanly.
        # print(json.dumps(prints, separators=(",", ":"), sort_keys=True))

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data