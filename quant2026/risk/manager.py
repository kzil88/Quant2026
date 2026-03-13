"""Risk manager: pre-trade and post-trade risk checks."""

import pandas as pd
from loguru import logger

from quant2026.types import PortfolioTarget, RiskMetrics


class RiskManager:
    """Risk control for A-share specific constraints."""

    def __init__(
        self,
        max_single_weight: float = 0.10,      # 单只最大仓位
        max_industry_weight: float = 0.30,     # 单行业最大仓位
        max_drawdown_limit: float = 0.15,      # 最大回撤止损线
        min_stocks: int = 10,                  # 最少持仓数
    ):
        self.max_single_weight = max_single_weight
        self.max_industry_weight = max_industry_weight
        self.max_drawdown_limit = max_drawdown_limit
        self.min_stocks = min_stocks

    def check_pre_trade(
        self,
        target: PortfolioTarget,
        industry_map: pd.Series | None = None,
    ) -> PortfolioTarget:
        """Pre-trade risk check: enforce limits before execution."""
        weights = target.weights.copy()
        warnings = []

        # 1. Single stock cap
        capped = weights.clip(upper=self.max_single_weight)
        if (capped != weights).any():
            warnings.append(f"Capped {(weights > self.max_single_weight).sum()} stocks to {self.max_single_weight:.0%}")
            weights = capped / capped.sum()

        # 2. Industry concentration check
        if industry_map is not None:
            ind_weights = weights.groupby(industry_map).sum()
            over = ind_weights[ind_weights > self.max_industry_weight]
            if len(over) > 0:
                warnings.append(f"Industry overweight: {dict(over.items())}")
                # TODO: implement industry rebalancing

        # 3. Minimum diversification
        if len(weights) < self.min_stocks:
            warnings.append(f"Only {len(weights)} stocks, below min {self.min_stocks}")

        if warnings:
            for w in warnings:
                logger.warning(f"[Risk] {w}")

        target.weights = weights
        target.metadata["risk_warnings"] = warnings
        return target

    def check_post_trade(
        self,
        equity_curve: pd.Series,
    ) -> RiskMetrics:
        """Post-trade risk assessment on portfolio equity curve."""
        returns = equity_curve.pct_change().dropna()

        # Max drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min()

        # Volatility (annualized)
        vol = returns.std() * (252 ** 0.5)

        # Sharpe ratio (assuming 2.5% risk-free for China)
        rf = 0.025 / 252
        sharpe = (returns.mean() - rf) / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0

        warnings = []
        if abs(max_dd) > self.max_drawdown_limit:
            warnings.append(f"Max drawdown {max_dd:.2%} exceeds limit {self.max_drawdown_limit:.2%}")

        return RiskMetrics(
            date=equity_curve.index[-1] if hasattr(equity_curve.index[-1], 'date') else None,
            max_drawdown=max_dd,
            volatility=vol,
            sharpe_ratio=sharpe,
            warnings=warnings,
        )
