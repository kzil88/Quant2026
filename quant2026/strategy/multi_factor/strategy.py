"""Multi-factor stock selection strategy (多因子选股)."""

from datetime import date

import pandas as pd

from quant2026.types import StrategyResult
from quant2026.strategy.base import Strategy


class MultiFactorStrategy(Strategy):
    """Equal/custom weighted multi-factor scoring."""

    def __init__(self, factor_weights: dict[str, float] | None = None, top_n: int = 30):
        self._factor_weights = factor_weights
        self._top_n = top_n

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

        sorted_scores = scores.sort_values(ascending=False)

        # Generate BUY/SELL/HOLD signals based on ranking
        from quant2026.types import Signal
        signals = pd.Series(Signal.HOLD, index=sorted_scores.index)
        top_stocks = sorted_scores.head(self._top_n).index
        bottom_stocks = sorted_scores.tail(self._top_n).index
        signals[top_stocks] = Signal.BUY
        signals[bottom_stocks] = Signal.SELL

        return StrategyResult(
            name=self.name,
            date=target_date,
            scores=sorted_scores,
            signals=signals,
        )
