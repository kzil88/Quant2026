"""Portfolio optimizer: combine strategy signals into target portfolio."""

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from quant2026.types import StrategyResult, PortfolioTarget


class PortfolioOptimizer:
    """Combine multiple strategy outputs into one portfolio."""

    def __init__(
        self,
        strategy_weights: dict[str, float] | None = None,
        max_stocks: int = 30,
        max_single_weight: float = 0.1,
    ):
        self._strategy_weights = strategy_weights or {}
        self._max_stocks = max_stocks
        self._max_single_weight = max_single_weight

    def combine(
        self,
        results: list[StrategyResult],
        target_date: date,
    ) -> PortfolioTarget:
        """Merge strategy scores into final stock weights."""
        if not results:
            raise ValueError("No strategy results to combine")

        # Normalize strategy weights
        weights = {}
        for r in results:
            weights[r.name] = self._strategy_weights.get(
                r.name, 1.0 / len(results)
            )
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        # Weighted score aggregation
        all_scores = pd.DataFrame({r.name: r.scores for r in results})

        # Rank-based normalization (more robust than raw scores)
        ranked = all_scores.rank(pct=True)

        combined = pd.Series(0.0, index=ranked.index)
        for name, w in weights.items():
            if name in ranked.columns:
                combined += ranked[name].fillna(0.5) * w

        # Select top N stocks
        top = combined.nlargest(self._max_stocks)

        # Equal weight within selected, capped by max_single_weight
        n = len(top)
        target_w = min(1.0 / n, self._max_single_weight)
        portfolio_weights = pd.Series(target_w, index=top.index)

        # Renormalize to sum to 1
        portfolio_weights = portfolio_weights / portfolio_weights.sum()

        logger.info(f"Portfolio: {n} stocks selected, top score: {top.iloc[0]:.4f}")

        return PortfolioTarget(
            date=target_date,
            weights=portfolio_weights,
            strategy_weights=weights,
        )
