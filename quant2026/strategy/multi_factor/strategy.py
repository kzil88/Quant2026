"""Multi-factor stock selection strategy (多因子选股)."""

from datetime import date

import pandas as pd

from quant2026.types import StrategyResult
from quant2026.strategy.base import Strategy


class MultiFactorStrategy(Strategy):
    """Equal/custom weighted multi-factor scoring."""

    def __init__(self, factor_weights: dict[str, float] | None = None):
        self._factor_weights = factor_weights

    @property
    def name(self) -> str:
        return "multi_factor"

    def generate(
        self,
        data: pd.DataFrame,
        factor_matrix: pd.DataFrame | None,
        target_date: date,
    ) -> StrategyResult:
        if factor_matrix is None or factor_matrix.empty:
            raise ValueError("MultiFactorStrategy requires factor_matrix")

        weights = self._factor_weights
        if weights is None:
            # Equal weight all factors
            weights = {col: 1.0 / len(factor_matrix.columns) for col in factor_matrix.columns}

        # Weighted score
        scores = pd.Series(0.0, index=factor_matrix.index)
        for factor_name, weight in weights.items():
            if factor_name in factor_matrix.columns:
                scores += factor_matrix[factor_name].fillna(0) * weight

        return StrategyResult(
            name=self.name,
            date=target_date,
            scores=scores.sort_values(ascending=False),
        )
