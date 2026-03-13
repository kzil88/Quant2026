"""Tests for TurnoverConstraint."""

import pandas as pd
import pytest

from quant2026.portfolio.turnover import TurnoverConstraint


@pytest.fixture
def tc():
    return TurnoverConstraint(max_turnover=0.3, penalty_weight=0.01)


class TestApply:
    def test_no_change_when_within_limit(self, tc):
        """换手率在限制内时不改变权重。"""
        current = pd.Series({"A": 0.4, "B": 0.3, "C": 0.3})
        target = pd.Series({"A": 0.35, "B": 0.35, "C": 0.3})
        # turnover = (0.05 + 0.05) / 2 = 0.05 < 0.3
        result = tc.apply(target, current)
        pd.testing.assert_series_equal(result, target, atol=1e-10)

    def test_constrained_when_exceeds_limit(self, tc):
        """换手率超限时被缩减到 max_turnover。"""
        current = pd.Series({"A": 0.8, "B": 0.2})
        target = pd.Series({"A": 0.2, "B": 0.8})
        # turnover = (0.6 + 0.6) / 2 = 0.6 > 0.3
        result = tc.apply(target, current)

        # Check turnover of result
        all_stocks = result.index.union(current.index)
        r = result.reindex(all_stocks, fill_value=0.0)
        c = current.reindex(all_stocks, fill_value=0.0)
        actual_turnover = (r - c).abs().sum() / 2
        assert actual_turnover <= tc.max_turnover + 1e-8

    def test_weights_sum_to_one(self, tc):
        """缩减后权重和仍为 1。"""
        current = pd.Series({"A": 0.8, "B": 0.2})
        target = pd.Series({"A": 0.1, "B": 0.5, "C": 0.4})
        result = tc.apply(target, current)
        assert abs(result.sum() - 1.0) < 1e-8

    def test_new_stocks_added(self, tc):
        """目标中有新股票时正确处理。"""
        current = pd.Series({"A": 1.0})
        target = pd.Series({"A": 0.5, "B": 0.5})
        result = tc.apply(target, current)
        assert abs(result.sum() - 1.0) < 1e-8


class TestTurnoverPenalty:
    def test_zero_when_same(self, tc):
        """惩罚项 = 0 当 current = target。"""
        w = pd.Series({"A": 0.5, "B": 0.5})
        assert tc.turnover_penalty(w, w) == 0.0

    def test_positive_when_different(self, tc):
        """不同权重时惩罚 > 0。"""
        current = pd.Series({"A": 0.5, "B": 0.5})
        target = pd.Series({"A": 0.7, "B": 0.3})
        penalty = tc.turnover_penalty(target, current)
        assert penalty > 0
        # penalty = 0.01 * (0.2 + 0.2) = 0.004
        assert abs(penalty - 0.004) < 1e-10


class TestEstimateCost:
    def test_cost_structure(self, tc):
        """成本估算返回正确结构。"""
        current = pd.Series({"A": 0.6, "B": 0.4})
        target = pd.Series({"A": 0.4, "B": 0.6})
        cost = tc.estimate_cost(target, current)
        assert "turnover" in cost
        assert "buy_turnover" in cost
        assert "sell_turnover" in cost
        assert "total_cost_pct" in cost
        assert "cost_breakdown" in cost

    def test_cost_reasonable(self, tc):
        """成本估算合理。"""
        current = pd.Series({"A": 0.6, "B": 0.4})
        target = pd.Series({"A": 0.4, "B": 0.6})
        cost = tc.estimate_cost(target, current)
        assert cost["turnover"] == pytest.approx(0.2, abs=1e-10)
        assert cost["buy_turnover"] == pytest.approx(0.2, abs=1e-10)
        assert cost["sell_turnover"] == pytest.approx(0.2, abs=1e-10)
        assert cost["total_cost_pct"] > 0

    def test_zero_cost_when_no_change(self, tc):
        """无变化时成本为 0。"""
        w = pd.Series({"A": 0.5, "B": 0.5})
        cost = tc.estimate_cost(w, w)
        assert cost["turnover"] == 0.0
        assert cost["total_cost_pct"] == 0.0


class TestValidation:
    def test_invalid_max_turnover(self):
        with pytest.raises(ValueError):
            TurnoverConstraint(max_turnover=0)
        with pytest.raises(ValueError):
            TurnoverConstraint(max_turnover=1.5)
