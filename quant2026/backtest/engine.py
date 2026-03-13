"""Backtesting engine with A-share specific logic."""

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from quant2026.types import PortfolioTarget


@dataclass
class BacktestConfig:
    """Backtest parameters."""
    start_date: date
    end_date: date
    initial_capital: float = 1_000_000.0
    commission_rate: float = 0.0003      # 万三佣金
    stamp_tax_rate: float = 0.0005       # 印花税 (卖出时收取, 2024年减半)
    slippage_pct: float = 0.001          # 滑点
    rebalance_days: int = 5              # 每N个交易日调仓
    t_plus_1: bool = True                # T+1 限制


@dataclass
class BacktestResult:
    """Backtest output."""
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    trades: list[dict] = field(default_factory=list)
    turnover: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Compute performance summary."""
        if self.equity_curve.empty:
            return {}
        total_ret = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        years = max((self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25, 0.01)
        annual_ret = (1 + total_ret) ** (1 / years) - 1
        vol = self.daily_returns.std() * np.sqrt(252)
        sharpe = (self.daily_returns.mean() - 0.025/252) / self.daily_returns.std() * np.sqrt(252) if self.daily_returns.std() > 0 else 0

        cummax = self.equity_curve.cummax()
        max_dd = ((self.equity_curve - cummax) / cummax).min()

        return {
            "total_return": f"{total_ret:.2%}",
            "annual_return": f"{annual_ret:.2%}",
            "volatility": f"{vol:.2%}",
            "sharpe_ratio": f"{sharpe:.2f}",
            "max_drawdown": f"{max_dd:.2%}",
            "trade_count": len(self.trades),
            "avg_turnover": f"{np.mean(self.turnover):.2%}" if self.turnover else "N/A",
        }


class BacktestEngine:
    """Event-driven backtester with A-share constraints."""

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        data: pd.DataFrame,
        targets: dict[date, PortfolioTarget],
        optimizer: object | None = None,
    ) -> BacktestResult:
        """Run backtest given data and rebalance targets.

        Args:
            data: daily OHLCV with columns [stock_code, date, open, high, low, close, volume]
            targets: {rebalance_date: PortfolioTarget}
            optimizer: Optional PortfolioOptimizer; if provided with turnover_constraint,
                       current_weights will be passed to combine() during rebalance.

        Returns:
            BacktestResult
        """
        cfg = self.config
        pivot = data.pivot_table(index="date", columns="stock_code", values="close")
        pivot = pivot.sort_index()

        dates = [d for d in pivot.index if cfg.start_date <= pd.Timestamp(d).date() <= cfg.end_date]
        if not dates:
            logger.error("No trading dates in range")
            return BacktestResult()

        capital = cfg.initial_capital
        holdings: dict[str, float] = {}  # stock_code -> weight
        equity = []
        returns = []
        trades = []
        turnover = []

        rebalance_dates = sorted(targets.keys())
        rebal_idx = 0

        for i, dt in enumerate(dates):
            dt_date = pd.Timestamp(dt).date()

            # Check rebalance
            if rebal_idx < len(rebalance_dates) and dt_date >= rebalance_dates[rebal_idx]:
                target = targets[rebalance_dates[rebal_idx]]
                new_weights = target.weights.to_dict()

                # Calculate turnover
                turn = sum(
                    abs(new_weights.get(s, 0) - holdings.get(s, 0))
                    for s in set(list(new_weights.keys()) + list(holdings.keys()))
                ) / 2
                turnover.append(turn)

                # Transaction cost
                cost = turn * (cfg.commission_rate + cfg.stamp_tax_rate + cfg.slippage_pct)
                capital *= (1 - cost)

                holdings = new_weights
                rebal_idx += 1

            # Daily P&L
            if i > 0 and holdings:
                prev_prices = pivot.loc[dates[i-1]]
                curr_prices = pivot.loc[dt]
                daily_ret = sum(
                    holdings.get(s, 0) * (curr_prices.get(s, 0) / prev_prices.get(s, 1) - 1)
                    for s in holdings
                    if prev_prices.get(s, 0) > 0
                )
                capital *= (1 + daily_ret)
                returns.append(daily_ret)
            else:
                returns.append(0.0)

            equity.append(capital)

        result = BacktestResult(
            equity_curve=pd.Series(equity, index=dates),
            daily_returns=pd.Series(returns, index=dates),
            trades=trades,
            turnover=turnover,
        )
        result.metrics = result.summary()
        logger.info(f"Backtest done: {result.metrics}")
        return result
