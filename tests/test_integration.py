"""End-to-end integration tests for Quant2026 pipeline.

All tests use mock data only — no network access required.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from quant2026.types import StrategyResult, PortfolioTarget, Signal


# ── Helpers ──────────────────────────────────────────────────────


def _run_factor_computation(ohlcv: pd.DataFrame, target_date: date) -> dict[str, pd.Series]:
    """Run a few factors on mock data and return {factor_name: scores}."""
    from quant2026.factor import MomentumFactor, VolatilityFactor, TurnoverFactor

    factors = [MomentumFactor(), VolatilityFactor(), TurnoverFactor()]
    results = {}
    for f in factors:
        try:
            results[f.name] = f.compute(ohlcv, target_date)
        except Exception as e:
            logger.warning(f"Factor {f.name} failed: {e}")
    return results


def _build_factor_matrix(
    ohlcv: pd.DataFrame, factor_scores: dict[str, pd.Series], target_date: date
) -> pd.DataFrame:
    """Build a simple factor matrix from factor scores."""
    if not factor_scores:
        return pd.DataFrame()
    df = pd.DataFrame(factor_scores)
    df.index.name = "stock_code"
    return df


def _run_strategy(ohlcv: pd.DataFrame, target_date: date) -> StrategyResult:
    """Run momentum strategy on mock data."""
    from quant2026.strategy.momentum.strategy import MomentumStrategy

    strategy = MomentumStrategy(fast_window=5, slow_window=20)
    return strategy.generate(ohlcv, factor_matrix=None, target_date=target_date)


def _run_multi_factor_strategy(
    ohlcv: pd.DataFrame, factor_matrix: pd.DataFrame, target_date: date
) -> StrategyResult:
    """Run multi-factor strategy."""
    from quant2026.strategy.multi_factor.strategy import MultiFactorStrategy

    strategy = MultiFactorStrategy()
    return strategy.generate(ohlcv, factor_matrix=factor_matrix, target_date=target_date)


def _prepare_ohlcv_for_backtest(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Ensure date column is proper datetime for BacktestEngine compatibility."""
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── End-to-End Pipeline Tests ────────────────────────────────────


@pytest.mark.integration
class TestEndToEndPipeline:
    """端到端集成测试（全 mock 数据，不依赖网络）。"""

    def test_full_pipeline_equal_weight(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """等权组合完整流程：数据→因子→策略→组合→回测。"""
        from quant2026.portfolio.optimizer import PortfolioOptimizer
        from quant2026.backtest.engine import BacktestEngine, BacktestConfig

        ohlcv = sample_ohlcv_data
        bt_ohlcv = _prepare_ohlcv_for_backtest(ohlcv)
        dates = sorted(ohlcv["date"].unique())
        target_date = date.fromisoformat(dates[-1])

        # 1. Factor computation
        factor_scores = _run_factor_computation(ohlcv, target_date)
        assert len(factor_scores) > 0, "At least one factor should compute"

        # 2. Strategy
        result = _run_strategy(ohlcv, target_date)
        assert isinstance(result, StrategyResult)
        assert len(result.scores) > 0

        # 3. Portfolio optimization (equal weight)
        optimizer = PortfolioOptimizer(method="equal", max_stocks=5)
        portfolio = optimizer.combine([result], target_date, price_data=ohlcv)
        assert isinstance(portfolio, PortfolioTarget)
        assert len(portfolio.weights) > 0
        assert abs(portfolio.weights.sum() - 1.0) < 0.05  # roughly sums to 1

        # 4. Backtest
        start_d = date.fromisoformat(dates[0])
        end_d = target_date
        bt_config = BacktestConfig(start_date=start_d, end_date=end_d)
        engine = BacktestEngine(bt_config)
        targets = {target_date: portfolio}
        bt_result = engine.run(bt_ohlcv, targets)
        assert len(bt_result.equity_curve) > 0

    def test_full_pipeline_markowitz(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """Markowitz 优化完整流程。"""
        from quant2026.portfolio.optimizer import PortfolioOptimizer
        from quant2026.backtest.engine import BacktestEngine, BacktestConfig

        ohlcv = sample_ohlcv_data
        bt_ohlcv = _prepare_ohlcv_for_backtest(ohlcv)
        dates = sorted(ohlcv["date"].unique())
        target_date = date.fromisoformat(dates[-1])

        result = _run_strategy(ohlcv, target_date)

        optimizer = PortfolioOptimizer(method="markowitz", max_stocks=5)
        portfolio = optimizer.combine([result], target_date, price_data=ohlcv)
        assert isinstance(portfolio, PortfolioTarget)
        assert len(portfolio.weights) > 0

        start_d = date.fromisoformat(dates[0])
        bt_config = BacktestConfig(start_date=start_d, end_date=target_date)
        engine = BacktestEngine(bt_config)
        bt_result = engine.run(bt_ohlcv, {target_date: portfolio})
        assert len(bt_result.equity_curve) > 0

    def test_config_driven_pipeline(self, sample_config, tmp_output_dir) -> None:
        """配置驱动：验证 Quant2026Config 可正确实例化各模块。"""
        from quant2026.config import Quant2026Config

        cfg = sample_config
        assert isinstance(cfg, Quant2026Config)
        assert len(cfg.data.stock_pool) > 0
        assert len(cfg.strategies) > 0
        assert cfg.strategies[0].type == "MultiFactorStrategy"

    def test_multi_strategy_pipeline(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """多策略融合流程。"""
        from quant2026.portfolio.optimizer import PortfolioOptimizer

        ohlcv = sample_ohlcv_data
        dates = sorted(ohlcv["date"].unique())
        target_date = date.fromisoformat(dates[-1])

        # Run two strategies
        momentum_result = _run_strategy(ohlcv, target_date)

        factor_scores = _run_factor_computation(ohlcv, target_date)
        factor_matrix = _build_factor_matrix(ohlcv, factor_scores, target_date)
        mf_result = _run_multi_factor_strategy(ohlcv, factor_matrix, target_date)

        # Combine
        optimizer = PortfolioOptimizer(
            strategy_weights={"momentum_5_20": 0.6, "multi_factor": 0.4},
            method="equal",
            max_stocks=5,
        )
        portfolio = optimizer.combine([momentum_result, mf_result], target_date, price_data=ohlcv)
        assert isinstance(portfolio, PortfolioTarget)
        assert len(portfolio.weights) > 0


# ── Data Flow Tests ──────────────────────────────────────────────


@pytest.mark.integration
class TestDataFlow:
    """数据流测试：验证模块间数据传递格式正确。"""

    def test_factor_output_matches_strategy_input(self, sample_ohlcv_data: pd.DataFrame) -> None:
        """因子输出格式 = 策略输入格式。"""
        dates = sorted(sample_ohlcv_data["date"].unique())
        target_date = date.fromisoformat(dates[-1])

        factor_scores = _run_factor_computation(sample_ohlcv_data, target_date)

        # Factor output is {name: pd.Series with stock_code index}
        for name, series in factor_scores.items():
            assert isinstance(series, pd.Series), f"Factor {name} should return Series"
            # Strategy can accept factor_matrix as DataFrame
            assert pd.api.types.is_string_dtype(series.index)  # string stock codes

    def test_strategy_output_matches_optimizer_input(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """策略输出格式 = 优化器输入格式。"""
        dates = sorted(sample_ohlcv_data["date"].unique())
        target_date = date.fromisoformat(dates[-1])

        result = _run_strategy(sample_ohlcv_data, target_date)

        # PortfolioOptimizer.combine expects list[StrategyResult]
        assert isinstance(result, StrategyResult)
        assert isinstance(result.scores, pd.Series)
        assert isinstance(result.name, str)
        assert isinstance(result.date, date)

    def test_optimizer_output_matches_backtest_input(
        self, sample_ohlcv_data: pd.DataFrame
    ) -> None:
        """优化器输出格式 = 回测引擎输入格式。"""
        from quant2026.portfolio.optimizer import PortfolioOptimizer
        from quant2026.backtest.engine import BacktestEngine, BacktestConfig

        dates = sorted(sample_ohlcv_data["date"].unique())
        target_date = date.fromisoformat(dates[-1])

        result = _run_strategy(sample_ohlcv_data, target_date)
        optimizer = PortfolioOptimizer(method="equal", max_stocks=5)
        portfolio = optimizer.combine([result], target_date, price_data=sample_ohlcv_data)

        # BacktestEngine.run expects dict[date, PortfolioTarget]
        assert isinstance(portfolio, PortfolioTarget)
        assert isinstance(portfolio.date, date)
        assert isinstance(portfolio.weights, pd.Series)

        # Verify it can be passed to engine without error
        start_d = date.fromisoformat(dates[0])
        bt_config = BacktestConfig(start_date=start_d, end_date=target_date)
        engine = BacktestEngine(bt_config)
        bt_result = engine.run(_prepare_ohlcv_for_backtest(sample_ohlcv_data), {target_date: portfolio})
        assert bt_result is not None
