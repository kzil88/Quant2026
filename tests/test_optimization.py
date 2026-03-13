"""Tests for the strategy parameter optimization framework."""

import numpy as np
import pandas as pd
import pytest

from quant2026.backtest.engine import BacktestConfig, BacktestResult
from quant2026.optimization.param_optimizer import (
    OptimizationResult,
    ParamSpace,
    StrategyOptimizer,
)
from quant2026.strategy.base import Strategy
from quant2026.types import StrategyResult, Signal
from datetime import date


# ── Fixtures ─────────────────────────────────────────────────────


class DummyStrategy(Strategy):
    """Minimal strategy for testing."""

    def __init__(self, x: float = 1.0, y: float = 1.0):
        self.x = x
        self.y = y

    @property
    def name(self) -> str:
        return "dummy"

    def generate(self, data, factor_matrix, target_date):
        return StrategyResult(
            name=self.name,
            date=target_date,
            scores=pd.Series({"A": self.x, "B": self.y}),
        )


def dummy_pipeline(strategy: DummyStrategy, data: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Fake pipeline: score = -(x-3)^2 - (y-2)^2 + 10  (max at x=3, y=2)."""
    score = -(strategy.x - 3.0) ** 2 - (strategy.y - 2.0) ** 2 + 10.0
    result = BacktestResult()
    result.metrics = {
        "sharpe_ratio": f"{score:.2f}",
        "annual_return": f"{score * 0.5:.2f}%",
        "max_drawdown": "-5.00%",
    }
    return result


def make_config() -> BacktestConfig:
    return BacktestConfig(start_date=date(2024, 1, 1), end_date=date(2024, 12, 31))


def make_data() -> pd.DataFrame:
    return pd.DataFrame({"stock_code": ["A"], "date": ["2024-01-01"], "close": [100.0]})


# ── ParamSpace validation ────────────────────────────────────────


class TestParamSpace:
    def test_choice_requires_choices(self):
        with pytest.raises(ValueError, match="choice type requires"):
            ParamSpace(name="a", type="choice", choices=[])

    def test_choice_requires_choices_none(self):
        with pytest.raises(ValueError, match="choice type requires"):
            ParamSpace(name="a", type="choice")

    def test_float_requires_low_high(self):
        with pytest.raises(ValueError, match="requires low and high"):
            ParamSpace(name="a", type="float", low=1.0)

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="must be float/int/choice"):
            ParamSpace(name="a", type="boolean")

    def test_low_gt_high(self):
        with pytest.raises(ValueError, match="low.*>.*high"):
            ParamSpace(name="a", type="float", low=5.0, high=1.0)

    def test_valid_choice(self):
        ps = ParamSpace(name="a", type="choice", choices=[1, 2, 3])
        assert ps.grid_values() == [1, 2, 3]

    def test_grid_values_float(self):
        ps = ParamSpace(name="a", type="float", low=0.0, high=1.0, step=0.5)
        assert ps.grid_values() == pytest.approx([0.0, 0.5, 1.0])

    def test_grid_values_int(self):
        ps = ParamSpace(name="a", type="int", low=1, high=5, step=2)
        vals = ps.grid_values()
        assert all(isinstance(v, (int, np.integer)) for v in vals)

    def test_grid_requires_step(self):
        ps = ParamSpace(name="a", type="float", low=0.0, high=1.0)
        with pytest.raises(ValueError, match="requires step"):
            ps.grid_values()

    def test_sample_random_choice(self):
        ps = ParamSpace(name="a", type="choice", choices=["x", "y"])
        rng = np.random.RandomState(0)
        val = ps.sample_random(rng)
        assert val in ["x", "y"]


# ── Grid Search ──────────────────────────────────────────────────


class TestGridSearch:
    def test_result_count(self):
        spaces = [
            ParamSpace(name="x", type="choice", choices=[1.0, 2.0, 3.0, 4.0]),
            ParamSpace(name="y", type="choice", choices=[1.0, 2.0, 3.0]),
        ]
        opt = StrategyOptimizer(objective="sharpe")
        result = opt.grid_search(
            param_spaces=spaces,
            strategy_factory=lambda p: DummyStrategy(x=p["x"], y=p["y"]),
            data=make_data(),
            backtest_config=make_config(),
            pipeline_fn=dummy_pipeline,
        )
        assert len(result.all_results) == 4 * 3  # 12 combos

    def test_best_is_max(self):
        spaces = [
            ParamSpace(name="x", type="choice", choices=[1.0, 2.0, 3.0, 4.0]),
            ParamSpace(name="y", type="choice", choices=[1.0, 2.0, 3.0]),
        ]
        opt = StrategyOptimizer(objective="sharpe")
        result = opt.grid_search(
            param_spaces=spaces,
            strategy_factory=lambda p: DummyStrategy(x=p["x"], y=p["y"]),
            data=make_data(),
            backtest_config=make_config(),
            pipeline_fn=dummy_pipeline,
        )
        assert result.best_score >= result.all_results["score"].max() - 1e-9
        # Best should be x=3, y=2
        assert result.best_params["x"] == pytest.approx(3.0)
        assert result.best_params["y"] == pytest.approx(2.0)


# ── Random Search ────────────────────────────────────────────────


class TestRandomSearch:
    def test_result_count(self):
        spaces = [
            ParamSpace(name="x", type="float", low=0.0, high=5.0),
            ParamSpace(name="y", type="float", low=0.0, high=5.0),
        ]
        opt = StrategyOptimizer(objective="sharpe")
        n_iter = 15
        result = opt.random_search(
            param_spaces=spaces,
            strategy_factory=lambda p: DummyStrategy(x=p["x"], y=p["y"]),
            data=make_data(),
            backtest_config=make_config(),
            pipeline_fn=dummy_pipeline,
            n_iter=n_iter,
        )
        assert len(result.all_results) == n_iter

    def test_best_score_ge_all(self):
        spaces = [
            ParamSpace(name="x", type="float", low=0.0, high=5.0),
            ParamSpace(name="y", type="float", low=0.0, high=5.0),
        ]
        opt = StrategyOptimizer(objective="sharpe")
        result = opt.random_search(
            param_spaces=spaces,
            strategy_factory=lambda p: DummyStrategy(x=p["x"], y=p["y"]),
            data=make_data(),
            backtest_config=make_config(),
            pipeline_fn=dummy_pipeline,
            n_iter=20,
        )
        assert result.best_score >= result.all_results["score"].max() - 1e-9


# ── Bayesian Search ──────────────────────────────────────────────


class TestBayesianSearch:
    def test_runs_and_returns_result(self):
        spaces = [
            ParamSpace(name="x", type="float", low=0.0, high=5.0),
            ParamSpace(name="y", type="float", low=0.0, high=5.0),
        ]
        opt = StrategyOptimizer(objective="sharpe")
        result = opt.bayesian_search(
            param_spaces=spaces,
            strategy_factory=lambda p: DummyStrategy(x=p["x"], y=p["y"]),
            data=make_data(),
            backtest_config=make_config(),
            pipeline_fn=dummy_pipeline,
            n_iter=15,
            n_initial=8,
        )
        assert len(result.all_results) == 15
        assert result.best_score >= result.all_results["score"].max() - 1e-9


# ── OptimizationResult ───────────────────────────────────────────


class TestOptimizationResult:
    def test_defaults(self):
        r = OptimizationResult()
        assert r.best_score == float("-inf")
        assert r.best_params == {}
