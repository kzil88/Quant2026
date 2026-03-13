"""Tests for walk-forward analysis."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant2026.backtest.walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
)
from quant2026.backtest.engine import BacktestConfig
from quant2026.types import PortfolioTarget


# ── Fixtures ───────────────────────────────────────────────────

def _make_mock_data(start: str = "2023-01-01", end: str = "2024-12-31", stocks: list[str] | None = None) -> pd.DataFrame:
    """Generate mock daily OHLCV data."""
    if stocks is None:
        stocks = ["000001", "000002", "000003", "000004", "000005"]

    dates = pd.bdate_range(start, end)
    rows = []
    rng = np.random.RandomState(42)
    for s in stocks:
        price = 10.0 + rng.randn() * 2
        for d in dates:
            ret = rng.randn() * 0.02
            price *= (1 + ret)
            price = max(price, 1.0)
            rows.append({
                "stock_code": s,
                "date": str(d.date()),
                "open": price * (1 - abs(rng.randn() * 0.005)),
                "high": price * (1 + abs(rng.randn() * 0.01)),
                "low": price * (1 - abs(rng.randn() * 0.01)),
                "close": price,
                "volume": rng.randint(1000, 100000),
            })
    return pd.DataFrame(rows)


def _simple_factory(train_data: pd.DataFrame, train_dates: list[date]) -> dict[date, PortfolioTarget]:
    """Simple factory: equal-weight all stocks at end of training."""
    stocks = train_data["stock_code"].unique().tolist()
    weights = pd.Series({s: 1.0 / len(stocks) for s in stocks})
    last_date = max(train_dates)
    return {last_date: PortfolioTarget(date=last_date, weights=weights)}


# ── Tests ──────────────────────────────────────────────────────

class TestWindowGeneration:
    """Test window splitting logic."""

    def test_generates_windows(self):
        config = WalkForwardConfig(train_months=6, test_months=2, step_months=2)
        analyzer = WalkForwardAnalyzer(config)
        windows = analyzer._generate_windows(date(2023, 1, 1), date(2024, 12, 31))
        assert len(windows) >= 2, f"Expected >= 2 windows, got {len(windows)}"

    def test_no_overlap(self):
        config = WalkForwardConfig(train_months=6, test_months=2, step_months=2)
        analyzer = WalkForwardAnalyzer(config)
        windows = analyzer._generate_windows(date(2023, 1, 1), date(2024, 12, 31))
        for ts, te, test_s, test_e in windows:
            assert te < test_s, f"Train end {te} should be before test start {test_s}"

    def test_train_before_test(self):
        config = WalkForwardConfig(train_months=3, test_months=1, step_months=1)
        analyzer = WalkForwardAnalyzer(config)
        windows = analyzer._generate_windows(date(2023, 1, 1), date(2024, 6, 30))
        for ts, te, test_s, test_e in windows:
            assert ts < te < test_s <= test_e

    def test_min_two_windows_with_enough_data(self):
        config = WalkForwardConfig(train_months=6, test_months=2, step_months=2)
        analyzer = WalkForwardAnalyzer(config)
        # 2 years should give multiple windows
        windows = analyzer._generate_windows(date(2023, 1, 1), date(2024, 12, 31))
        assert len(windows) >= 2


class TestWalkForwardRun:
    """Test full walk-forward run with mock data."""

    def test_basic_run(self):
        data = _make_mock_data("2023-01-01", "2024-12-31")
        config = WalkForwardConfig(train_months=6, test_months=2, step_months=2, min_train_days=50)
        bt_config = BacktestConfig(start_date=date(2023, 1, 1), end_date=date(2024, 12, 31))
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(data, _simple_factory, bt_config)

        assert len(result.windows) >= 2
        assert not result.combined_equity.empty

    def test_combined_equity_continuous(self):
        data = _make_mock_data("2023-01-01", "2024-12-31")
        config = WalkForwardConfig(train_months=6, test_months=2, step_months=2, min_train_days=50)
        bt_config = BacktestConfig(start_date=date(2023, 1, 1), end_date=date(2024, 12, 31))
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(data, _simple_factory, bt_config)

        eq = result.combined_equity
        if len(eq) > 1:
            # Should be monotonically indexed (dates in order)
            dates = pd.to_datetime(eq.index)
            assert dates.is_monotonic_increasing

    def test_efficiency_ratio_computed(self):
        data = _make_mock_data("2023-01-01", "2024-12-31")
        config = WalkForwardConfig(train_months=6, test_months=2, step_months=2, min_train_days=50)
        bt_config = BacktestConfig(start_date=date(2023, 1, 1), end_date=date(2024, 12, 31))
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(data, _simple_factory, bt_config)

        # efficiency_ratio should be a finite number
        assert np.isfinite(result.efficiency_ratio)


class TestWalkForwardResult:
    """Test WalkForwardResult methods."""

    def test_summary(self):
        result = WalkForwardResult(
            windows=[
                WalkForwardWindow(window_id=0, train_start=date(2023, 1, 1), train_end=date(2023, 6, 30),
                                  test_start=date(2023, 7, 1), test_end=date(2023, 8, 31)),
            ],
            combined_equity=pd.Series([1.0, 1.01, 1.02]),
            combined_metrics={"sharpe_ratio": "1.50"},
            efficiency_ratio=0.75,
        )
        s = result.summary()
        assert s["num_windows"] == 1
        assert s["overfitting_warning"] is False

    def test_overfitting_warning(self):
        result = WalkForwardResult(efficiency_ratio=0.3)
        assert result.summary()["overfitting_warning"] is True


class TestReport:
    """Test report generation."""

    def test_generate_report(self, tmp_path):
        data = _make_mock_data("2023-01-01", "2024-06-30")
        config = WalkForwardConfig(train_months=3, test_months=1, step_months=1, min_train_days=30)
        bt_config = BacktestConfig(start_date=date(2023, 1, 1), end_date=date(2024, 6, 30))
        analyzer = WalkForwardAnalyzer(config)
        result = analyzer.run(data, _simple_factory, bt_config)

        out = tmp_path / "report.html"
        path = analyzer.generate_report(result, str(out))
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "Walk-Forward" in content
        assert "Efficiency Ratio" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
