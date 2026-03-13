"""Trade execution simulator with A-share constraints."""

import pandas as pd
from loguru import logger

from quant2026.types import PortfolioTarget, TradeOrder, Signal


class ExecutionSimulator:
    """Simulate trade execution with A-share constraints."""

    def __init__(
        self,
        check_limit_up_down: bool = True,
        check_suspended: bool = True,
        check_t_plus_1: bool = True,
    ):
        self._check_limit = check_limit_up_down
        self._check_suspended = check_suspended
        self._check_t1 = check_t_plus_1

    def generate_orders(
        self,
        current_weights: dict[str, float],
        target: PortfolioTarget,
        market_data: pd.DataFrame | None = None,
    ) -> list[TradeOrder]:
        """Generate trade orders from current to target portfolio.

        Respects A-share constraints:
        - T+1: can't sell stocks bought today
        - Limit up: can't buy
        - Limit down: can't sell
        - Suspended: can't trade
        """
        orders = []
        all_codes = set(list(current_weights.keys()) + list(target.weights.index))

        for code in all_codes:
            current_w = current_weights.get(code, 0.0)
            target_w = target.weights.get(code, 0.0)
            diff = target_w - current_w

            if abs(diff) < 0.001:
                continue

            # Check A-share constraints
            if market_data is not None and code in market_data.get("stock_code", pd.Series()).values:
                stock_data = market_data[market_data["stock_code"] == code].iloc[-1]
                if self._check_limit and stock_data.get("is_limit_up", False) and diff > 0:
                    logger.info(f"Skip BUY {code}: limit up")
                    continue
                if self._check_limit and stock_data.get("is_limit_down", False) and diff < 0:
                    logger.info(f"Skip SELL {code}: limit down")
                    continue
                if self._check_suspended and stock_data.get("is_suspended", False):
                    logger.info(f"Skip {code}: suspended")
                    continue

            signal = Signal.BUY if diff > 0 else Signal.SELL
            orders.append(TradeOrder(
                stock_code=code,
                signal=signal,
                target_weight=target_w,
                current_weight=current_w,
            ))

        return orders
