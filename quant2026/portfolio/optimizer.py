"""Portfolio optimizer: combine strategy signals into target portfolio."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from quant2026.types import StrategyResult, PortfolioTarget
from quant2026.portfolio.markowitz import MarkowitzOptimizer
from quant2026.portfolio.risk_parity import RiskParityOptimizer
from quant2026.portfolio.turnover import TurnoverConstraint


def _compute_cov_matrix(price_data: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Compute shrunk covariance matrix from daily close prices.

    Args:
        price_data: DataFrame with columns [stock_code, date, close].
        lookback: Number of trailing days for estimation.

    Returns:
        Covariance matrix as DataFrame (stock_code x stock_code).
    """
    pivot = price_data.pivot_table(index="date", columns="stock_code", values="close")
    pivot = pivot.sort_index().tail(lookback)
    returns = pivot.pct_change().dropna()

    if len(returns) < 10:
        logger.warning("Too few data points for cov estimation, using sample cov")
        return returns.cov()

    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(returns.dropna(axis=1).values)
        clean_cols = returns.dropna(axis=1).columns
        return pd.DataFrame(lw.covariance_, index=clean_cols, columns=clean_cols)
    except Exception as e:
        logger.warning(f"LedoitWolf failed, using sample cov: {e}")
        return returns.cov()


class PortfolioOptimizer:
    """Combine multiple strategy outputs into one portfolio.

    Supports three optimization methods:
    - "equal": rank-based equal-weight (original behavior)
    - "markowitz": mean-variance optimization (max_sharpe by default)
    - "risk_parity": risk parity allocation
    """

    def __init__(
        self,
        strategy_weights: dict[str, float] | None = None,
        max_stocks: int = 30,
        max_single_weight: float = 0.1,
        method: str = "equal",
        markowitz_method: str = "max_sharpe",
        risk_free_rate: float = 0.025,
        turnover_constraint: TurnoverConstraint | None = None,
    ):
        self._strategy_weights = strategy_weights or {}
        self._max_stocks = max_stocks
        self._max_single_weight = max_single_weight
        self._method = method
        self._markowitz_method = markowitz_method
        self._risk_free_rate = risk_free_rate
        self._turnover_constraint = turnover_constraint

    def combine(
        self,
        results: list[StrategyResult],
        target_date: date,
        price_data: pd.DataFrame | None = None,
        current_weights: pd.Series | None = None,
    ) -> PortfolioTarget:
        """Merge strategy scores into final stock weights.

        Args:
            results: Strategy outputs with scores.
            target_date: The rebalance date.
            price_data: Daily OHLCV data (required for markowitz/risk_parity).

        Returns:
            PortfolioTarget with optimized weights.
        """
        if not results:
            raise ValueError("No strategy results to combine")

        # Normalize strategy weights
        weights = {}
        for r in results:
            weights[r.name] = self._strategy_weights.get(r.name, 1.0 / len(results))
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

        # Weighted score aggregation (rank-based)
        all_scores = pd.DataFrame({r.name: r.scores for r in results})
        ranked = all_scores.rank(pct=True)
        combined = pd.Series(0.0, index=ranked.index)
        for name, w in weights.items():
            if name in ranked.columns:
                combined += ranked[name].fillna(0.5) * w

        if self._method == "equal" or price_data is None:
            if self._method != "equal" and price_data is None:
                logger.warning(f"price_data required for method='{self._method}', falling back to equal")
            target = self._equal_weight(combined, target_date, weights)
        else:
            # Compute covariance from price_data
            cov_matrix = _compute_cov_matrix(price_data, lookback=60)

            if self._method == "markowitz":
                target = self._markowitz_optimize(combined, cov_matrix, target_date, weights)
            elif self._method == "risk_parity":
                target = self._risk_parity_optimize(combined, cov_matrix, target_date, weights)
            else:
                logger.warning(f"Unknown method '{self._method}', falling back to equal")
                target = self._equal_weight(combined, target_date, weights)

        # Apply turnover constraint if configured
        if self._turnover_constraint is not None and current_weights is not None:
            target = PortfolioTarget(
                date=target.date,
                weights=self._turnover_constraint.apply(target.weights, current_weights),
                strategy_weights=target.strategy_weights,
                metadata=target.metadata,
            )

        return target

    def _equal_weight(
        self, combined: pd.Series, target_date: date, strategy_weights: dict[str, float],
    ) -> PortfolioTarget:
        """Original rank-based equal weight."""
        top = combined.nlargest(self._max_stocks)
        n = len(top)
        target_w = min(1.0 / n, self._max_single_weight)
        portfolio_weights = pd.Series(target_w, index=top.index)
        portfolio_weights = portfolio_weights / portfolio_weights.sum()
        logger.info(f"Portfolio(equal): {n} stocks selected")
        return PortfolioTarget(date=target_date, weights=portfolio_weights, strategy_weights=strategy_weights)

    def _markowitz_optimize(
        self, combined: pd.Series, cov_matrix: pd.DataFrame,
        target_date: date, strategy_weights: dict[str, float],
    ) -> PortfolioTarget:
        """Markowitz mean-variance optimization using scores as expected returns proxy."""
        opt = MarkowitzOptimizer(
            risk_free_rate=self._risk_free_rate,
            max_single_weight=self._max_single_weight,
            min_weight=0.01,
            max_stocks=self._max_stocks,
        )
        # Use combined scores as expected returns proxy
        common = combined.index.intersection(cov_matrix.index)
        if len(common) == 0:
            logger.warning("No common stocks for Markowitz, falling back to equal")
            return self._equal_weight(combined, target_date, strategy_weights)

        try:
            w = opt.optimize(combined.loc[common], cov_matrix.loc[common, common], method=self._markowitz_method)
            return PortfolioTarget(date=target_date, weights=w, strategy_weights=strategy_weights)
        except Exception as e:
            logger.warning(f"Markowitz failed ({e}), falling back to equal")
            return self._equal_weight(combined, target_date, strategy_weights)

    def _risk_parity_optimize(
        self, combined: pd.Series, cov_matrix: pd.DataFrame,
        target_date: date, strategy_weights: dict[str, float],
    ) -> PortfolioTarget:
        """Risk parity optimization on top-ranked stocks."""
        top = combined.nlargest(self._max_stocks)
        common = top.index.intersection(cov_matrix.index)
        if len(common) == 0:
            logger.warning("No common stocks for RiskParity, falling back to equal")
            return self._equal_weight(combined, target_date, strategy_weights)

        opt = RiskParityOptimizer(max_single_weight=self._max_single_weight)
        try:
            w = opt.optimize(cov_matrix.loc[common, common])
            return PortfolioTarget(date=target_date, weights=w, strategy_weights=strategy_weights)
        except Exception as e:
            logger.warning(f"RiskParity failed ({e}), falling back to equal")
            return self._equal_weight(combined, target_date, strategy_weights)
