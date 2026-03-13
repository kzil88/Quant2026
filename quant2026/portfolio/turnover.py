"""Turnover constraint: limit rebalance turnover to control transaction costs."""

from __future__ import annotations

import pandas as pd
from loguru import logger


class TurnoverConstraint:
    """换手率约束：限制每次调仓的换手量。

    可通过两种方式使用：
    1. apply() — 硬约束，超限时按比例缩减调整幅度
    2. turnover_penalty() — 软约束，返回惩罚项加到优化目标函数
    """

    def __init__(
        self,
        max_turnover: float = 0.3,
        penalty_weight: float = 0.01,
    ):
        """初始化换手率约束。

        Args:
            max_turnover: 单次最大换手率 (0~1)，默认 30%。
            penalty_weight: 换手惩罚系数，用于优化目标函数中的软约束。
        """
        if not 0 < max_turnover <= 1:
            raise ValueError(f"max_turnover must be in (0, 1], got {max_turnover}")
        self.max_turnover = max_turnover
        self.penalty_weight = penalty_weight

    def apply(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
    ) -> pd.Series:
        """如果换手率超过 max_turnover，按比例缩减调整幅度。

        Args:
            target_weights: 新目标权重。
            current_weights: 当前持仓权重。

        Returns:
            调整后的权重 (归一化，和为 1)。
        """
        # Align indexes
        all_stocks = target_weights.index.union(current_weights.index)
        tw = target_weights.reindex(all_stocks, fill_value=0.0)
        cw = current_weights.reindex(all_stocks, fill_value=0.0)

        delta = tw - cw
        turnover = delta.abs().sum() / 2

        if turnover <= self.max_turnover:
            logger.debug(f"Turnover {turnover:.2%} <= {self.max_turnover:.2%}, no adjustment")
            return target_weights

        scale = self.max_turnover / turnover
        new_weights = cw + delta * scale

        # Remove near-zero weights and normalize
        new_weights = new_weights[new_weights > 1e-8]
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()

        logger.info(
            f"Turnover constrained: {turnover:.2%} → {self.max_turnover:.2%} "
            f"(scale={scale:.4f}, {len(new_weights)} stocks)"
        )
        return new_weights

    def turnover_penalty(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
    ) -> float:
        """计算换手惩罚项，可加到优化目标函数中。

        penalty = penalty_weight * sum(|target - current|)

        Args:
            target_weights: 新目标权重。
            current_weights: 当前持仓权重。

        Returns:
            惩罚值 (float)。
        """
        all_stocks = target_weights.index.union(current_weights.index)
        tw = target_weights.reindex(all_stocks, fill_value=0.0)
        cw = current_weights.reindex(all_stocks, fill_value=0.0)
        return self.penalty_weight * (tw - cw).abs().sum()

    def estimate_cost(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
        commission: float = 0.0003,
        stamp_tax: float = 0.0005,
        slippage: float = 0.001,
    ) -> dict:
        """估算调仓成本。

        Args:
            target_weights: 新目标权重。
            current_weights: 当前持仓权重。
            commission: 佣金率 (双边)。
            stamp_tax: 印花税率 (卖出)。
            slippage: 滑点。

        Returns:
            dict with keys: turnover, buy_turnover, sell_turnover,
            total_cost_pct, cost_breakdown.
        """
        all_stocks = target_weights.index.union(current_weights.index)
        tw = target_weights.reindex(all_stocks, fill_value=0.0)
        cw = current_weights.reindex(all_stocks, fill_value=0.0)

        delta = tw - cw
        buy_turnover = float(delta[delta > 0].sum())
        sell_turnover = float((-delta[delta < 0]).sum())
        turnover = (buy_turnover + sell_turnover) / 2

        # Costs: commission on both sides, stamp_tax on sell only, slippage on both
        buy_cost = buy_turnover * (commission + slippage)
        sell_cost = sell_turnover * (commission + stamp_tax + slippage)
        total_cost = buy_cost + sell_cost

        return {
            "turnover": turnover,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "total_cost_pct": total_cost,
            "cost_breakdown": {
                "commission": (buy_turnover + sell_turnover) * commission,
                "stamp_tax": sell_turnover * stamp_tax,
                "slippage": (buy_turnover + sell_turnover) * slippage,
            },
        }
