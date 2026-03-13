"""Tests for PerformanceAttribution."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant2026.backtest.attribution import PerformanceAttribution


@pytest.fixture
def attr():
    return PerformanceAttribution()


@pytest.fixture
def sample_data():
    """Generate sample data for attribution tests."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", "2024-06-30")
    stocks = [f"stock_{i}" for i in range(20)]
    returns = pd.DataFrame(
        np.random.randn(len(dates), len(stocks)) * 0.02,
        index=dates,
        columns=stocks,
    )
    industries = ["Tech"] * 5 + ["Finance"] * 5 + ["Consumer"] * 5 + ["Industrial"] * 5
    industry_map = pd.Series(industries, index=stocks)

    # Portfolio weights at two rebalance dates
    pw1 = pd.Series(np.random.dirichlet(np.ones(20)), index=stocks)
    pw2 = pd.Series(np.random.dirichlet(np.ones(20)), index=stocks)
    portfolio_weights = {
        dates[0].date(): pw1,
        dates[60].date(): pw2,
    }

    benchmark_weights = pd.Series(1.0 / len(stocks), index=stocks)

    return {
        "returns": returns,
        "industry_map": industry_map,
        "portfolio_weights": portfolio_weights,
        "benchmark_weights": benchmark_weights,
        "dates": dates,
        "stocks": stocks,
    }


class TestSectorAttribution:
    def test_columns(self, attr, sample_data):
        result = attr.sector_attribution(
            sample_data["portfolio_weights"],
            sample_data["benchmark_weights"],
            sample_data["returns"],
            sample_data["industry_map"],
        )
        assert list(result.columns) == ["allocation", "selection", "interaction", "total"]

    def test_total_equals_sum(self, attr, sample_data):
        """allocation + selection + interaction ≈ total for each industry."""
        result = attr.sector_attribution(
            sample_data["portfolio_weights"],
            sample_data["benchmark_weights"],
            sample_data["returns"],
            sample_data["industry_map"],
        )
        computed_total = result["allocation"] + result["selection"] + result["interaction"]
        pd.testing.assert_series_equal(
            computed_total, result["total"], check_names=False, atol=1e-10
        )

    def test_industries_present(self, attr, sample_data):
        result = attr.sector_attribution(
            sample_data["portfolio_weights"],
            sample_data["benchmark_weights"],
            sample_data["returns"],
            sample_data["industry_map"],
        )
        assert set(result.index) == {"Tech", "Finance", "Consumer", "Industrial"}


class TestFactorAttribution:
    def test_r_squared_in_range(self, attr, sample_data):
        np.random.seed(42)
        dates = sample_data["dates"]
        portfolio_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        factor_returns = pd.DataFrame({
            "market": np.random.randn(len(dates)) * 0.01,
            "size": np.random.randn(len(dates)) * 0.005,
            "momentum": np.random.randn(len(dates)) * 0.005,
        }, index=dates)

        result = attr.factor_attribution(portfolio_returns, factor_returns)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_keys_present(self, attr, sample_data):
        dates = sample_data["dates"]
        portfolio_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        factor_returns = pd.DataFrame({
            "market": np.random.randn(len(dates)) * 0.01,
        }, index=dates)

        result = attr.factor_attribution(portfolio_returns, factor_returns)
        assert "factor_exposures" in result
        assert "factor_contributions" in result
        assert "alpha" in result
        assert "r_squared" in result

    def test_exposures_match_factors(self, attr, sample_data):
        dates = sample_data["dates"]
        portfolio_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        factor_returns = pd.DataFrame({
            "market": np.random.randn(len(dates)) * 0.01,
            "size": np.random.randn(len(dates)) * 0.005,
        }, index=dates)

        result = attr.factor_attribution(portfolio_returns, factor_returns)
        assert set(result["factor_exposures"].keys()) == {"market", "size"}


class TestMonthlyAttribution:
    def test_excess_equals_diff(self, attr):
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        np.random.seed(42)
        portfolio_returns = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        benchmark_returns = pd.Series(np.random.randn(len(dates)) * 0.008, index=dates)

        result = attr.monthly_attribution(portfolio_returns, benchmark_returns)
        # excess should equal portfolio - benchmark
        diff = (result["portfolio"] - result["benchmark"] - result["excess"]).abs()
        assert (diff < 1e-10).all(), f"Excess mismatch: {diff.max()}"

    def test_columns(self, attr):
        dates = pd.bdate_range("2024-01-01", "2024-03-31")
        pr = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        br = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        result = attr.monthly_attribution(pr, br)
        assert list(result.columns) == ["portfolio", "benchmark", "excess", "cumulative_excess"]

    def test_cumulative_excess(self, attr):
        dates = pd.bdate_range("2024-01-01", "2024-06-30")
        np.random.seed(42)
        pr = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        br = pd.Series(np.random.randn(len(dates)) * 0.008, index=dates)
        result = attr.monthly_attribution(pr, br)
        # Cumulative should be running sum of excess
        expected_cum = result["excess"].cumsum()
        diff = (result["cumulative_excess"] - expected_cum).abs()
        assert (diff < 1e-10).all()


class TestReportGeneration:
    def test_generates_html(self, attr, sample_data, tmp_path):
        sector_attr = attr.sector_attribution(
            sample_data["portfolio_weights"],
            sample_data["benchmark_weights"],
            sample_data["returns"],
            sample_data["industry_map"],
        )
        dates = sample_data["dates"]
        pr = pd.Series(np.random.randn(len(dates)) * 0.01, index=dates)
        fr = pd.DataFrame({"market": np.random.randn(len(dates)) * 0.01}, index=dates)
        br = pd.Series(np.random.randn(len(dates)) * 0.008, index=dates)

        factor_attr = attr.factor_attribution(pr, fr)
        monthly_attr = attr.monthly_attribution(pr, br)

        out = tmp_path / "report.html"
        result = attr.generate_report(sector_attr, factor_attr, monthly_attr, str(out))
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "Sector Attribution" in content
        assert "Factor Attribution" in content
        assert "base64" in content
