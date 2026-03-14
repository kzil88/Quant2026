"""A股 T+1 交易约束管理器。"""

from datetime import date

import pandas as pd
from loguru import logger


class TPlusOneManager:
    """A股 T+1 交易约束管理器。

    当日买入的股票次日才能卖出。
    """

    def __init__(self) -> None:
        self._buy_dates: dict[str, date] = {}  # stock_code -> 最近买入日期

    def record_buy(self, stock_code: str, buy_date: date) -> None:
        """记录买入日期。"""
        self._buy_dates[stock_code] = buy_date
        logger.debug(f"T+1: recorded buy {stock_code} on {buy_date}")

    def can_sell(self, stock_code: str, current_date: date) -> bool:
        """是否可以卖出（当日买入的不能卖）。"""
        buy_date = self._buy_dates.get(stock_code)
        if buy_date is None:
            return True  # 没有买入记录，可以卖出（持仓来自更早之前）
        return current_date > buy_date

    def filter_sells(
        self,
        target_weights: pd.Series,
        current_holdings: pd.Series,
        current_date: date,
    ) -> pd.Series:
        """过滤不能卖出的股票。

        如果 target_weight < current_weight 但 can_sell=False，保持 current_weight。
        最后重新归一化权重使总和为 1。
        """
        adjusted = target_weights.copy()

        for stock in current_holdings.index:
            current_w = current_holdings.get(stock, 0.0)
            target_w = adjusted.get(stock, 0.0)

            if target_w < current_w and not self.can_sell(stock, current_date):
                logger.info(
                    f"T+1: cannot sell {stock} on {current_date}, "
                    f"keeping weight {current_w:.4f} (target was {target_w:.4f})"
                )
                adjusted[stock] = current_w

        # 重新归一化
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total

        return adjusted

    def reset(self) -> None:
        """清空记录。"""
        self._buy_dates.clear()
        logger.debug("T+1: reset all buy records")
