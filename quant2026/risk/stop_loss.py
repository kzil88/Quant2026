"""Stop-loss strategies and blacklist management."""

import pandas as pd
from loguru import logger

from quant2026.types import PortfolioTarget


class StopLossManager:
    """止损策略管理器"""

    def __init__(
        self,
        stock_stop_loss: float = -0.10,
        portfolio_stop_loss: float = -0.15,
        trailing_stop: float = -0.08,
        cooldown_days: int = 5,
    ):
        self.stock_stop_loss = stock_stop_loss
        self.portfolio_stop_loss = portfolio_stop_loss
        self.trailing_stop = trailing_stop
        self.cooldown_days = cooldown_days
        self._cooldown: dict[str, int] = {}  # stock -> remaining cooldown days

    def check_stock_stop_loss(
        self,
        holdings: dict[str, float],
    ) -> list[str]:
        """返回需要止损卖出的股票列表

        Args:
            holdings: stock_code -> 持仓收益率 (e.g. -0.12 means -12%)
        """
        triggered = [
            stock for stock, ret in holdings.items()
            if ret <= self.stock_stop_loss
        ]
        if triggered:
            logger.warning(f"[StopLoss] 个股止损触发: {triggered}")
        return triggered

    def check_portfolio_stop_loss(
        self,
        equity_curve: pd.Series,
    ) -> bool:
        """是否触发组合止损（清仓或大幅减仓）"""
        if len(equity_curve) < 2:
            return False
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        current_dd = drawdown.iloc[-1]
        if current_dd <= self.portfolio_stop_loss:
            logger.warning(f"[StopLoss] 组合止损触发: 当前回撤 {current_dd:.2%}")
            return True
        return False

    def check_trailing_stop(
        self,
        stock_prices: dict[str, pd.Series],
        entry_prices: dict[str, float],
    ) -> list[str]:
        """移动止损：从持仓期最高点回撤超过阈值就卖"""
        triggered = []
        for stock, prices in stock_prices.items():
            if stock not in entry_prices or prices.empty:
                continue
            entry = entry_prices[stock]
            # 只看买入后的价格
            holding_prices = prices[prices.index >= prices.index[0]]
            peak = holding_prices.max()
            current = holding_prices.iloc[-1]
            if peak > 0:
                drawdown_from_peak = (current - peak) / peak
                if drawdown_from_peak <= self.trailing_stop:
                    triggered.append(stock)
        if triggered:
            logger.warning(f"[StopLoss] 移动止损触发: {triggered}")
        return triggered

    def apply_blacklist(
        self,
        target: PortfolioTarget,
        blacklist: set[str],
        industry_blacklist: set[str] | None = None,
        industry_map: pd.Series | None = None,
    ) -> PortfolioTarget:
        """从目标组合中剔除黑名单股票/行业，重新归一化权重"""
        weights = target.weights.copy()
        removed = []

        # 股票黑名单
        to_remove = weights.index.intersection(list(blacklist))
        if len(to_remove) > 0:
            removed.extend(list(to_remove))
            weights = weights.drop(to_remove)

        # 行业黑名单
        if industry_blacklist and industry_map is not None:
            for stock in list(weights.index):
                if stock in industry_map.index:
                    if industry_map[stock] in industry_blacklist:
                        removed.append(stock)
                        weights = weights.drop(stock)

        # 重新归一化
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        if removed:
            logger.info(f"[Blacklist] 剔除 {len(removed)} 只: {removed}")

        target.weights = weights
        return target
