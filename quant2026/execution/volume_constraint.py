"""成交量约束：单笔成交不超过当日成交量的 N%。"""

import pandas as pd
from loguru import logger


class VolumeConstraint:
    """成交量约束：单笔成交不超过当日成交量的 N%。"""

    def __init__(
        self,
        max_participation_rate: float = 0.10,
        min_volume_threshold: int = 100_000,
    ) -> None:
        self.max_participation_rate = max_participation_rate
        self.min_volume_threshold = min_volume_threshold

    def check_executable(
        self,
        stock_code: str,
        order_shares: int,
        daily_volume: int,
    ) -> dict:
        """检查是否可执行。

        Returns:
            dict with keys: executable, max_shares, actual_shares,
            participation_rate, reason
        """
        if daily_volume < self.min_volume_threshold:
            return {
                "executable": False,
                "max_shares": 0,
                "actual_shares": 0,
                "participation_rate": 0.0,
                "reason": f"{stock_code} daily volume {daily_volume} below threshold {self.min_volume_threshold}",
            }

        max_shares = int(daily_volume * self.max_participation_rate)
        actual_shares = min(order_shares, max_shares)
        rate = actual_shares / daily_volume if daily_volume > 0 else 0.0

        executable = order_shares <= max_shares
        reason = "OK" if executable else (
            f"{stock_code} order {order_shares} exceeds max {max_shares} "
            f"({self.max_participation_rate:.0%} of {daily_volume})"
        )

        return {
            "executable": executable,
            "max_shares": max_shares,
            "actual_shares": actual_shares,
            "participation_rate": rate,
            "reason": reason,
        }

    def adjust_portfolio(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        total_capital: float,
        daily_volumes: pd.Series,
        stock_prices: pd.Series,
    ) -> tuple[pd.Series, dict]:
        """根据成交量约束调整目标权重。

        超过成交量限制的股票权重被截断，剩余权重重新分配。

        Returns:
            (adjusted_weights, adjustment_details)
        """
        adjusted = target_weights.copy()
        details: dict = {}

        for stock in target_weights.index:
            weight_diff = abs(target_weights.get(stock, 0.0) - current_weights.get(stock, 0.0))
            if weight_diff < 1e-8:
                continue

            price = stock_prices.get(stock, 0.0)
            volume = daily_volumes.get(stock, 0)

            if price <= 0 or volume < self.min_volume_threshold:
                # 无法交易，保持当前权重
                adjusted[stock] = current_weights.get(stock, 0.0)
                details[stock] = {"capped": True, "reason": "low volume or no price"}
                logger.info(f"Volume: {stock} cannot trade, volume={volume}")
                continue

            trade_value = weight_diff * total_capital
            order_shares = int(trade_value / price)
            max_shares = int(volume * self.max_participation_rate)

            if order_shares > max_shares:
                # 截断到最大可成交量
                max_value = max_shares * price
                max_weight_diff = max_value / total_capital
                current_w = current_weights.get(stock, 0.0)
                target_w = target_weights.get(stock, 0.0)

                if target_w > current_w:
                    adjusted[stock] = current_w + max_weight_diff
                else:
                    adjusted[stock] = current_w - max_weight_diff

                details[stock] = {
                    "capped": True,
                    "original_weight": target_w,
                    "adjusted_weight": adjusted[stock],
                    "max_shares": max_shares,
                    "order_shares": order_shares,
                }
                logger.info(
                    f"Volume: {stock} capped {order_shares} -> {max_shares} shares"
                )

        # 归一化
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total

        return adjusted, details
