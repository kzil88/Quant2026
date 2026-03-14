"""组件工厂：根据 YAML 配置创建策略、优化器、风控等组件。"""

from __future__ import annotations

from datetime import date

from loguru import logger

from quant2026.backtest.engine import BacktestConfig
from quant2026.config import Quant2026Config, StrategyConfig
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.portfolio.turnover import TurnoverConstraint
from quant2026.risk.manager import RiskManager
from quant2026.strategy.base import Strategy
from quant2026.strategy.mean_reversion.strategy import MeanReversionStrategy
from quant2026.strategy.multi_factor.strategy import MultiFactorStrategy
from quant2026.strategy.stat_arb.strategy import StatArbStrategy
from quant2026.strategy.event_driven.strategy import EventDrivenStrategy
from quant2026.strategy.ml_model.strategy import MLStrategy


_STRATEGY_MAP: dict[str, type[Strategy]] = {
    "MultiFactorStrategy": MultiFactorStrategy,
    "MeanReversionStrategy": MeanReversionStrategy,
    "StatArbStrategy": StatArbStrategy,
    "EventDrivenStrategy": EventDrivenStrategy,
    "MLStrategy": MLStrategy,
}

# 每种策略的 __init__ 接受的参数映射
_REBALANCE_MAP = {
    "daily": 1,
    "weekly": 5,
    "monthly": 20,
}


class ComponentFactory:
    """根据 Quant2026Config 创建各类组件。"""

    @staticmethod
    def create_strategies(config: Quant2026Config) -> list[tuple[Strategy, float]]:
        """根据配置创建策略实例列表。

        Returns:
            [(strategy_instance, weight), ...]
        """
        results: list[tuple[Strategy, float]] = []
        for sc in config.strategies:
            cls = _STRATEGY_MAP.get(sc.type)
            if cls is None:
                logger.warning(f"未知策略类型 {sc.type!r}，跳过")
                continue
            try:
                strategy = cls(**sc.params)
            except TypeError as e:
                logger.warning(f"策略 {sc.name} 参数错误: {e}，使用默认参数")
                strategy = cls()
            results.append((strategy, sc.weight))
            logger.info(f"创建策略: {sc.name} ({sc.type}) weight={sc.weight}")
        return results

    @staticmethod
    def create_optimizer(config: Quant2026Config) -> PortfolioOptimizer:
        """创建组合优化器（含换手率约束）。"""
        turnover = TurnoverConstraint(
            max_turnover=config.portfolio.max_turnover,
            penalty_weight=config.portfolio.turnover_penalty_weight,
        )
        # 构造 strategy_weights dict
        strategy_weights = {s.name: s.weight for s in config.strategies}

        optimizer = PortfolioOptimizer(
            strategy_weights=strategy_weights,
            max_single_weight=config.portfolio.markowitz_max_single_weight,
            method=config.portfolio.method,
            risk_free_rate=config.portfolio.markowitz_risk_free_rate,
            turnover_constraint=turnover,
        )
        logger.info(f"创建优化器: method={config.portfolio.method}")
        return optimizer

    @staticmethod
    def create_risk_manager(config: Quant2026Config) -> RiskManager:
        """创建风控管理器。"""
        rm = RiskManager(
            max_single_weight=config.risk.max_position_size,
            max_industry_weight=config.risk.max_sector_exposure,
            stock_stop_loss=config.risk.stock_stop_loss,
            portfolio_stop_loss=config.risk.portfolio_stop_loss,
            trailing_stop=config.risk.trailing_stop,
        )
        logger.info("创建风控管理器")
        return rm

    @staticmethod
    def create_backtest_config(config: Quant2026Config) -> BacktestConfig:
        """将 YAML 配置转为 BacktestConfig。"""
        rebalance_days = _REBALANCE_MAP.get(config.backtest.rebalance_frequency, 20)
        return BacktestConfig(
            start_date=date.fromisoformat(config.data.start_date),
            end_date=date.fromisoformat(config.data.end_date),
            initial_capital=config.backtest.initial_capital,
            commission_rate=config.backtest.commission,
            stamp_tax_rate=config.backtest.stamp_tax,
            slippage_pct=config.backtest.slippage,
            rebalance_days=rebalance_days,
        )
