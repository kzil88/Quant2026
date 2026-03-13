"""Tests for Markowitz and Risk Parity portfolio optimizers."""

import numpy as np
import pandas as pd
import pytest

from quant2026.portfolio.markowitz import MarkowitzOptimizer
from quant2026.portfolio.risk_parity import RiskParityOptimizer


def _make_test_data(n: int = 5, seed: int = 42):
    """Generate synthetic expected returns and covariance matrix."""
    rng = np.random.RandomState(seed)
    stocks = [f"stock_{i:03d}" for i in range(n)]
    expected_returns = pd.Series(rng.uniform(0.01, 0.15, n), index=stocks)
    # Generate valid positive-definite cov matrix
    A = rng.randn(n, n) * 0.01
    cov = pd.DataFrame(A @ A.T + np.eye(n) * 0.001, index=stocks, columns=stocks)
    return expected_returns, cov, stocks


class TestMarkowitzOptimizer:
    def test_max_sharpe_weights_sum_to_one(self):
        er, cov, _ = _make_test_data()
        opt = MarkowitzOptimizer(max_single_weight=0.50, min_weight=0.01)
        w = opt.optimize(er, cov, method="max_sharpe")
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self):
        er, cov, _ = _make_test_data()
        opt = MarkowitzOptimizer(max_single_weight=0.50, min_weight=0.01)
        w = opt.optimize(er, cov, method="max_sharpe")
        assert (w >= -1e-8).all()

    def test_max_weight_constraint(self):
        er, cov, _ = _make_test_data(10)
        max_w = 0.15
        opt = MarkowitzOptimizer(max_single_weight=max_w, min_weight=0.01, max_stocks=10)
        w = opt.optimize(er, cov, method="max_sharpe")
        assert w.max() <= max_w + 1e-6

    def test_min_variance(self):
        er, cov, _ = _make_test_data()
        opt = MarkowitzOptimizer(max_single_weight=0.50, min_weight=0.01)
        w = opt.optimize(er, cov, method="min_variance")
        assert abs(w.sum() - 1.0) < 1e-6

    def test_target_return(self):
        er, cov, _ = _make_test_data()
        opt = MarkowitzOptimizer(max_single_weight=0.50, min_weight=0.01)
        target = 0.05
        w = opt.optimize(er, cov, method="target_return", target_return=target)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_single_stock(self):
        er, cov, _ = _make_test_data(1)
        opt = MarkowitzOptimizer(max_single_weight=1.0, min_weight=0.0)
        w = opt.optimize(er, cov)
        assert len(w) == 1
        assert abs(w.iloc[0] - 1.0) < 1e-6

    def test_equal_returns(self):
        """When all expected returns are the same, should still produce valid weights."""
        stocks = [f"s{i}" for i in range(5)]
        er = pd.Series(0.05, index=stocks)
        A = np.eye(5) * 0.01
        cov = pd.DataFrame(A, index=stocks, columns=stocks)
        opt = MarkowitzOptimizer(max_single_weight=0.50, min_weight=0.01)
        w = opt.optimize(er, cov)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-8).all()

    def test_efficient_frontier_format(self):
        er, cov, _ = _make_test_data(8)
        opt = MarkowitzOptimizer(max_single_weight=0.30, min_weight=0.01, max_stocks=8)
        ef = opt.efficient_frontier(er, cov, n_points=10)
        assert isinstance(ef, pd.DataFrame)
        assert "return" in ef.columns
        assert "risk" in ef.columns
        assert "sharpe" in ef.columns
        if len(ef) > 1:
            # Returns should generally be non-decreasing
            assert ef["return"].iloc[-1] >= ef["return"].iloc[0] - 1e-6


class TestRiskParityOptimizer:
    def test_weights_sum_to_one(self):
        _, cov, _ = _make_test_data()
        opt = RiskParityOptimizer(max_single_weight=0.50)
        w = opt.optimize(cov)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_risk_contributions_approximately_equal(self):
        _, cov, stocks = _make_test_data()
        opt = RiskParityOptimizer(max_single_weight=0.50)
        w = opt.optimize(cov)
        wv = w.values
        sigma = cov.values
        port_var = wv @ sigma @ wv
        rc = wv * (sigma @ wv) / port_var
        # Risk contributions should be roughly equal (1/n each)
        assert np.std(rc) < 0.05, f"Risk contributions too uneven: {rc}"

    def test_single_stock(self):
        _, cov, _ = _make_test_data(1)
        opt = RiskParityOptimizer()
        w = opt.optimize(cov)
        assert len(w) == 1
        assert abs(w.iloc[0] - 1.0) < 1e-6

    def test_custom_budget(self):
        _, cov, stocks = _make_test_data()
        budget = pd.Series([0.4, 0.15, 0.15, 0.15, 0.15], index=stocks)
        opt = RiskParityOptimizer(max_single_weight=0.60)
        w = opt.optimize(cov, budget=budget)
        assert abs(w.sum() - 1.0) < 1e-6
        # Stock 0 should have highest risk contribution
        wv = w.values
        sigma = cov.values
        port_var = wv @ sigma @ wv
        rc = wv * (sigma @ wv) / port_var
        assert rc[0] > rc[1] - 0.05  # roughly higher

    def test_max_weight_constraint(self):
        _, cov, _ = _make_test_data(10)
        max_w = 0.15
        opt = RiskParityOptimizer(max_single_weight=max_w)
        w = opt.optimize(cov)
        assert w.max() <= max_w + 1e-6
