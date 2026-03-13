"""Risk manager: pre-trade and post-trade risk checks."""

import pandas as pd
from loguru import logger

from quant2026.types import PortfolioTarget, RiskMetrics
from quant2026.risk.var import VaRCalculator
from quant2026.risk.stop_loss import StopLossManager


class RiskManager:
    """Risk control for A-share specific constraints."""

    def __init__(
        self,
        max_single_weight: float = 0.10,
        max_industry_weight: float = 0.30,
        max_drawdown_limit: float = 0.15,
        min_stocks: int = 10,
        var_confidence: float = 0.95,
        stock_stop_loss: float = -0.10,
        portfolio_stop_loss: float = -0.15,
        trailing_stop: float = -0.08,
        cooldown_days: int = 5,
    ):
        self.max_single_weight = max_single_weight
        self.max_industry_weight = max_industry_weight
        self.max_drawdown_limit = max_drawdown_limit
        self.min_stocks = min_stocks
        self.var_calc = VaRCalculator(confidence=var_confidence)
        self.stop_loss = StopLossManager(
            stock_stop_loss=stock_stop_loss,
            portfolio_stop_loss=portfolio_stop_loss,
            trailing_stop=trailing_stop,
            cooldown_days=cooldown_days,
        )

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
        """Post-trade risk assessment with VaR/CVaR."""
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

        # VaR / CVaR
        var_95 = self.var_calc.historical_var(returns)
        cvar_95 = self.var_calc.cvar(returns)

        warnings = []
        if abs(max_dd) > self.max_drawdown_limit:
            warnings.append(f"Max drawdown {max_dd:.2%} exceeds limit {self.max_drawdown_limit:.2%}")

        return RiskMetrics(
            date=equity_curve.index[-1] if hasattr(equity_curve.index[-1], 'date') else None,
            max_drawdown=max_dd,
            volatility=vol,
            sharpe_ratio=sharpe,
            warnings=warnings,
            var_95=var_95,
            cvar_95=cvar_95,
        )

    def check_stop_loss(
        self,
        equity_curve: pd.Series,
        holdings: dict[str, float] | None = None,
        stock_prices: dict[str, pd.Series] | None = None,
        entry_prices: dict[str, float] | None = None,
    ) -> dict:
        """综合止损检查，供 BacktestEngine 调用

        Returns:
            {portfolio_stop: bool, stock_stops: list[str], trailing_stops: list[str]}
        """
        result: dict = {
            "portfolio_stop": False,
            "stock_stops": [],
            "trailing_stops": [],
        }

        result["portfolio_stop"] = self.stop_loss.check_portfolio_stop_loss(equity_curve)

        if holdings:
            result["stock_stops"] = self.stop_loss.check_stock_stop_loss(holdings)

        if stock_prices and entry_prices:
            result["trailing_stops"] = self.stop_loss.check_trailing_stop(
                stock_prices, entry_prices
            )

        return result
