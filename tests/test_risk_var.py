"""Tests for VaR/CVaR, stop-loss, and blacklist."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant2026.risk.var import VaRCalculator
from quant2026.risk.stop_loss import StopLossManager
from quant2026.risk.manager import RiskManager
from quant2026.types import PortfolioTarget


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def daily_returns():
    """Simulated daily returns ~ N(0.0005, 0.02)"""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0005, 0.02, 500), name="returns")


@pytest.fixture
def var_calc():
    return VaRCalculator(confidence=0.95)


@pytest.fixture
def returns_matrix():
    np.random.seed(123)
    stocks = ["A", "B", "C", "D"]
    data = np.random.normal(0.0003, 0.02, (250, 4))
    return pd.DataFrame(data, columns=stocks)


# ── VaR Tests ─────────────────────────────────────────────────

class TestVaRCalculator:
    def test_historical_var_negative(self, var_calc, daily_returns):
        var = var_calc.historical_var(daily_returns)
        assert var < 0, "VaR should be negative (loss)"

    def test_historical_var_reasonable(self, var_calc, daily_returns):
        var = var_calc.historical_var(daily_returns)
        assert abs(var) < 0.10, f"Daily VaR {var} seems too large"

    def test_parametric_var_negative(self, var_calc, daily_returns):
        var = var_calc.parametric_var(daily_returns)
        assert var < 0

    def test_parametric_var_same_order(self, var_calc, daily_returns):
        h = var_calc.historical_var(daily_returns)
        p = var_calc.parametric_var(daily_returns)
        ratio = abs(h / p) if p != 0 else 999
        assert 0.3 < ratio < 3.0, f"hist={h}, param={p} differ too much"

    def test_cvar_worse_than_var(self, var_calc, daily_returns):
        var = var_calc.historical_var(daily_returns)
        cvar = var_calc.cvar(daily_returns)
        assert cvar <= var, f"CVaR {cvar} should be <= VaR {var} (more negative)"

    def test_rolling_var_length(self, var_calc, daily_returns):
        rolling = var_calc.rolling_var(daily_returns, window=60)
        assert len(rolling) == len(daily_returns)
        assert rolling.iloc[:59].isna().all()
        assert rolling.iloc[59:].notna().any()

    def test_portfolio_var(self, var_calc, returns_matrix):
        weights = pd.Series([0.3, 0.3, 0.2, 0.2], index=returns_matrix.columns)
        result = var_calc.portfolio_var(weights, returns_matrix)
        assert "var_hist" in result
        assert "cvar" in result
        assert "component_var" in result
        assert result["var_hist"] < 0
        assert result["cvar"] <= result["var_hist"]


# ── StopLoss Tests ────────────────────────────────────────────

class TestStopLossManager:
    def test_stock_stop_loss(self):
        sl = StopLossManager(stock_stop_loss=-0.10)
        holdings = {"A": -0.12, "B": -0.05, "C": -0.15, "D": 0.03}
        triggered = sl.check_stock_stop_loss(holdings)
        assert "A" in triggered
        assert "C" in triggered
        assert "B" not in triggered
        assert "D" not in triggered

    def test_portfolio_stop_loss_triggered(self):
        sl = StopLossManager(portfolio_stop_loss=-0.15)
        # Equity drops 20% from peak
        eq = pd.Series([100, 105, 110, 100, 90, 85])
        assert sl.check_portfolio_stop_loss(eq) is True

    def test_portfolio_stop_loss_not_triggered(self):
        sl = StopLossManager(portfolio_stop_loss=-0.15)
        eq = pd.Series([100, 105, 103, 104, 102])
        assert sl.check_portfolio_stop_loss(eq) is False

    def test_trailing_stop(self):
        sl = StopLossManager(trailing_stop=-0.08)
        prices = {
            "A": pd.Series([10, 12, 13, 14, 12.5]),  # peak 14, current 12.5 => -10.7%
            "B": pd.Series([10, 11, 10.5, 10.8]),     # peak 11, current 10.8 => -1.8%
        }
        entry = {"A": 10.0, "B": 10.0}
        triggered = sl.check_trailing_stop(prices, entry)
        assert "A" in triggered
        assert "B" not in triggered

    def test_blacklist_removes_and_renormalizes(self):
        sl = StopLossManager()
        target = PortfolioTarget(
            date=date(2024, 6, 1),
            weights=pd.Series({"A": 0.3, "B": 0.3, "C": 0.2, "D": 0.2}),
        )
        result = sl.apply_blacklist(target, blacklist={"B", "D"})
        assert "B" not in result.weights.index
        assert "D" not in result.weights.index
        assert abs(result.weights.sum() - 1.0) < 1e-9

    def test_blacklist_industry(self):
        sl = StopLossManager()
        target = PortfolioTarget(
            date=date(2024, 6, 1),
            weights=pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        )
        industry_map = pd.Series({"A": "银行", "B": "白酒", "C": "银行", "D": "科技"})
        result = sl.apply_blacklist(
            target, blacklist=set(), industry_blacklist={"银行"}, industry_map=industry_map
        )
        assert "A" not in result.weights.index
        assert "C" not in result.weights.index
        assert abs(result.weights.sum() - 1.0) < 1e-9


# ── RiskManager Integration ──────────────────────────────────

class TestRiskManagerIntegration:
    def test_post_trade_includes_var(self):
        rm = RiskManager()
        np.random.seed(99)
        eq = pd.Series(
            1_000_000 * np.cumprod(1 + np.random.normal(0.0003, 0.01, 200)),
            index=pd.date_range("2024-01-01", periods=200),
        )
        metrics = rm.check_post_trade(eq)
        assert metrics.var_95 is not None
        assert metrics.cvar_95 is not None
        assert metrics.var_95 < 0
        assert metrics.cvar_95 <= metrics.var_95

    def test_check_stop_loss(self):
        rm = RiskManager(portfolio_stop_loss=-0.10)
        eq = pd.Series([100, 110, 105, 95, 85])
        holdings = {"X": -0.12, "Y": 0.05}
        result = rm.check_stop_loss(eq, holdings=holdings)
        assert result["portfolio_stop"] is True
        assert "X" in result["stock_stops"]
