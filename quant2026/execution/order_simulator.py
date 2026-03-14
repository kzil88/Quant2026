"""限价单 / 市价单 / VWAP 模拟器。"""

from dataclasses import dataclass
from datetime import date

from loguru import logger


@dataclass
class Order:
    """委托订单。"""
    stock_code: str
    direction: str      # "buy" | "sell"
    price: float        # 委托价格
    shares: int         # 委托股数
    order_type: str     # "market" | "limit" | "vwap"
    created_at: date


@dataclass
class Fill:
    """成交回报。"""
    order: Order
    fill_price: float
    fill_shares: int
    fill_date: date
    slippage: float     # 实际滑点


class OrderSimulator:
    """订单模拟器：模拟限价单成交。"""

    def __init__(
        self,
        default_slippage: float = 0.001,
        partial_fill: bool = True,
    ) -> None:
        self.default_slippage = default_slippage
        self.partial_fill = partial_fill

    def simulate_market_order(
        self,
        order: Order,
        ohlcv: dict,
    ) -> Fill:
        """市价单：以开盘价 + 滑点成交。"""
        open_price = ohlcv["open"]
        if order.direction == "buy":
            fill_price = open_price * (1 + self.default_slippage)
        else:
            fill_price = open_price * (1 - self.default_slippage)

        slippage = (fill_price - open_price) / open_price if open_price > 0 else 0.0

        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_shares=order.shares,
            fill_date=order.created_at,
            slippage=slippage,
        )
        logger.debug(f"Market fill: {order.stock_code} {order.direction} {fill.fill_shares}@{fill.fill_price:.4f}")
        return fill

    def simulate_limit_order(
        self,
        order: Order,
        ohlcv: dict,
    ) -> Fill | None:
        """限价单模拟。

        买入：如果 low <= order.price，以 min(order.price, open) 成交
        卖出：如果 high >= order.price，以 max(order.price, open) 成交
        否则返回 None（未成交）
        部分成交：根据成交量比例
        """
        open_p, high, low, volume = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["volume"]

        if order.direction == "buy":
            if low > order.price:
                logger.debug(f"Limit buy {order.stock_code} not filled: low {low} > limit {order.price}")
                return None
            fill_price = min(order.price, open_p)
        else:  # sell
            if high < order.price:
                logger.debug(f"Limit sell {order.stock_code} not filled: high {high} < limit {order.price}")
                return None
            fill_price = max(order.price, open_p)

        # 部分成交
        if self.partial_fill and volume > 0:
            max_fill = int(volume * 0.1)  # 假设最多成交当日10%成交量
            fill_shares = min(order.shares, max_fill) if max_fill > 0 else order.shares
        else:
            fill_shares = order.shares

        slippage = (fill_price - order.price) / order.price if order.price > 0 else 0.0

        fill = Fill(
            order=order,
            fill_price=fill_price,
            fill_shares=fill_shares,
            fill_date=order.created_at,
            slippage=slippage,
        )
        logger.debug(f"Limit fill: {order.stock_code} {order.direction} {fill.fill_shares}@{fill.fill_price:.4f}")
        return fill

    def simulate_vwap_order(
        self,
        order: Order,
        ohlcv: dict,
    ) -> Fill:
        """VWAP 模拟：以 (open+high+low+close)/4 近似 VWAP 成交。"""
        vwap = (ohlcv["open"] + ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 4
        slippage = (vwap - ohlcv["open"]) / ohlcv["open"] if ohlcv["open"] > 0 else 0.0

        fill = Fill(
            order=order,
            fill_price=vwap,
            fill_shares=order.shares,
            fill_date=order.created_at,
            slippage=slippage,
        )
        logger.debug(f"VWAP fill: {order.stock_code} {order.direction} {fill.fill_shares}@{fill.fill_price:.4f}")
        return fill
