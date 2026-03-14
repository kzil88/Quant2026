"""Tests for execution module: T+1, volume constraint, order simulator."""

from datetime import date

import pandas as pd
import pytest

from quant2026.execution.t_plus_one import TPlusOneManager
from quant2026.execution.volume_constraint import VolumeConstraint
from quant2026.execution.order_simulator import Order, Fill, OrderSimulator


# ── T+1 ──────────────────────────────────────────────────────

class TestTPlusOne:
    def test_cannot_sell_same_day(self):
        mgr = TPlusOneManager()
        mgr.record_buy("600519", date(2024, 3, 1))
        assert not mgr.can_sell("600519", date(2024, 3, 1))

    def test_can_sell_next_day(self):
        mgr = TPlusOneManager()
        mgr.record_buy("600519", date(2024, 3, 1))
        assert mgr.can_sell("600519", date(2024, 3, 2))

    def test_can_sell_no_record(self):
        mgr = TPlusOneManager()
        assert mgr.can_sell("600519", date(2024, 3, 1))

    def test_reset(self):
        mgr = TPlusOneManager()
        mgr.record_buy("600519", date(2024, 3, 1))
        mgr.reset()
        assert mgr.can_sell("600519", date(2024, 3, 1))

    def test_filter_sells(self):
        mgr = TPlusOneManager()
        mgr.record_buy("A", date(2024, 3, 1))

        target = pd.Series({"A": 0.0, "B": 1.0})
        current = pd.Series({"A": 0.5, "B": 0.5})

        result = mgr.filter_sells(target, current, date(2024, 3, 1))
        # A cannot be sold, should keep 0.5; then normalized
        assert result["A"] > 0.3  # kept at 0.5 -> normalized 0.5/1.5 ≈ 0.333
        assert abs(result.sum() - 1.0) < 1e-6


# ── Volume Constraint ────────────────────────────────────────

class TestVolumeConstraint:
    def test_under_limit(self):
        vc = VolumeConstraint(max_participation_rate=0.10)
        r = vc.check_executable("600519", 5000, 1_000_000)
        assert r["executable"] is True
        assert r["actual_shares"] == 5000

    def test_over_limit_capped(self):
        vc = VolumeConstraint(max_participation_rate=0.10)
        r = vc.check_executable("600519", 200_000, 1_000_000)
        assert r["executable"] is False
        assert r["max_shares"] == 100_000
        assert r["actual_shares"] == 100_000

    def test_low_volume_rejected(self):
        vc = VolumeConstraint(min_volume_threshold=100_000)
        r = vc.check_executable("600519", 100, 50_000)
        assert r["executable"] is False
        assert r["max_shares"] == 0


# ── Order Simulator ──────────────────────────────────────────

class TestOrderSimulator:
    def _make_ohlcv(self, o=10.0, h=11.0, l=9.0, c=10.5, v=1_000_000):
        return {"open": o, "high": h, "low": l, "close": c, "volume": v}

    def test_market_buy(self):
        sim = OrderSimulator(default_slippage=0.001)
        order = Order("600519", "buy", 10.0, 100, "market", date(2024, 3, 1))
        fill = sim.simulate_market_order(order, self._make_ohlcv())
        assert fill.fill_price == pytest.approx(10.0 * 1.001, rel=1e-6)
        assert fill.fill_shares == 100

    def test_market_sell(self):
        sim = OrderSimulator(default_slippage=0.001)
        order = Order("600519", "sell", 10.0, 100, "market", date(2024, 3, 1))
        fill = sim.simulate_market_order(order, self._make_ohlcv())
        assert fill.fill_price == pytest.approx(10.0 * 0.999, rel=1e-6)

    def test_limit_buy_filled(self):
        """买入限价高于最低价 -> 成交"""
        sim = OrderSimulator(partial_fill=False)
        order = Order("600519", "buy", 9.5, 100, "limit", date(2024, 3, 1))
        fill = sim.simulate_limit_order(order, self._make_ohlcv(l=9.0))
        assert fill is not None
        assert fill.fill_price == 9.5  # min(limit=9.5, open=10) = 9.5

    def test_limit_buy_not_filled(self):
        """买入限价低于最低价 -> 不成交"""
        sim = OrderSimulator(partial_fill=False)
        order = Order("600519", "buy", 8.0, 100, "limit", date(2024, 3, 1))
        fill = sim.simulate_limit_order(order, self._make_ohlcv(l=9.0))
        assert fill is None

    def test_limit_sell_filled(self):
        """卖出限价低于最高价 -> 成交"""
        sim = OrderSimulator(partial_fill=False)
        order = Order("600519", "sell", 10.5, 100, "limit", date(2024, 3, 1))
        fill = sim.simulate_limit_order(order, self._make_ohlcv(h=11.0))
        assert fill is not None
        assert fill.fill_price == 10.5  # max(limit=10.5, open=10) = 10.5

    def test_limit_sell_not_filled(self):
        """卖出限价高于最高价 -> 不成交"""
        sim = OrderSimulator(partial_fill=False)
        order = Order("600519", "sell", 12.0, 100, "limit", date(2024, 3, 1))
        fill = sim.simulate_limit_order(order, self._make_ohlcv(h=11.0))
        assert fill is None

    def test_vwap(self):
        sim = OrderSimulator()
        order = Order("600519", "buy", 10.0, 100, "vwap", date(2024, 3, 1))
        ohlcv = self._make_ohlcv(o=10.0, h=11.0, l=9.0, c=10.5)
        fill = sim.simulate_vwap_order(order, ohlcv)
        expected_vwap = (10.0 + 11.0 + 9.0 + 10.5) / 4
        assert fill.fill_price == pytest.approx(expected_vwap, rel=1e-6)
        assert fill.fill_shares == 100
