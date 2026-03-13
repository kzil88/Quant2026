"""Tests for all factors using mock data."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant2026.factor import (
    MomentumFactor,
    VolatilityFactor,
    TurnoverFactor,
    VolumePriceFactor,
    PEFactor,
    PBFactor,
    DividendYieldFactor,
    ROEFactor,
    GrossMarginFactor,
    DebtRatioFactor,
    RevenueGrowthFactor,
    ProfitGrowthFactor,
    RSIFactor,
    MACDFactor,
    BollingerFactor,
)

TARGET_DATE = date(2026, 3, 10)
STOCKS = ["000001", "000002", "000003"]


def _make_daily_data(n_days: int = 60) -> pd.DataFrame:
    """Generate mock daily OHLCV data for 3 stocks over n_days."""
    rows = []
    np.random.seed(42)
    for stock in STOCKS:
        base_price = np.random.uniform(10, 50)
        for i in range(n_days):
            d = date(2026, 1, 1 + i) if i < 31 else date(2026, 2, i - 30) if i < 59 else date(2026, 3, 10)
            # simple random walk
            base_price *= 1 + np.random.normal(0, 0.02)
            o = base_price * (1 + np.random.normal(0, 0.005))
            h = max(o, base_price) * (1 + abs(np.random.normal(0, 0.005)))
            l = min(o, base_price) * (1 - abs(np.random.normal(0, 0.005)))
            rows.append({
                "stock_code": stock,
                "date": str(d),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(l, 2),
                "close": round(base_price, 2),
                "volume": int(np.random.uniform(1e6, 1e7)),
                "amount": round(np.random.uniform(1e7, 1e8), 2),
            })
    return pd.DataFrame(rows)


def _make_financial_data() -> pd.DataFrame:
    """Generate mock financial data for 3 stocks."""
    return pd.DataFrame({
        "stock_code": STOCKS,
        "eps": [1.5, 2.0, 0.8],
        "bps": [10.0, 15.0, 5.0],
        "dps": [0.5, 0.8, 0.2],
        "roe": [0.15, 0.20, 0.08],
        "gross_margin": [0.35, 0.45, 0.25],
        "debt_ratio": [0.40, 0.60, 0.30],
        "revenue_growth": [0.12, 0.25, -0.05],
        "profit_growth": [0.10, 0.30, -0.10],
    })


@pytest.fixture
def daily_data() -> pd.DataFrame:
    return _make_daily_data()


@pytest.fixture
def financial_data() -> pd.DataFrame:
    return _make_financial_data()


# ---------------------------------------------------------------------------
# Existing factors (backward compat — no financial_data needed)
# ---------------------------------------------------------------------------

class TestExistingFactors:
    def test_momentum(self, daily_data):
        f = MomentumFactor(window=20)
        result = f.compute(daily_data, TARGET_DATE)
        assert len(result) == 3
        assert not result.isna().all()

    def test_volatility(self, daily_data):
        f = VolatilityFactor(window=20)
        result = f.compute(daily_data, TARGET_DATE)
        assert len(result) == 3
        assert (result > 0).all()

    def test_turnover(self, daily_data):
        f = TurnoverFactor(window=20)
        result = f.compute(daily_data, TARGET_DATE)
        assert len(result) == 3

    def test_volume_price(self, daily_data):
        f = VolumePriceFactor(window=20)
        result = f.compute(daily_data, TARGET_DATE)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Value factors
# ---------------------------------------------------------------------------

class TestValueFactors:
    def test_pe_factor(self, daily_data, financial_data):
        f = PEFactor()
        result = f.compute(daily_data, TARGET_DATE, financial_data)
        assert len(result) == 3
        assert f.category == "value"
        assert f.name == "ep"
        # EP = eps / close, should be positive for positive eps
        assert (result > 0).all()

    def test_pb_factor(self, daily_data, financial_data):
        f = PBFactor()
        result = f.compute(daily_data, TARGET_DATE, financial_data)
        assert len(result) == 3
        assert f.name == "bp"

    def test_dividend_yield(self, daily_data, financial_data):
        f = DividendYieldFactor()
        result = f.compute(daily_data, TARGET_DATE, financial_data)
        assert len(result) == 3
        assert (result > 0).all()

    def test_value_no_financial_data(self, daily_data):
        """Value factors should return empty series without financial_data."""
        f = PEFactor()
        result = f.compute(daily_data, TARGET_DATE)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Quality factors
# ---------------------------------------------------------------------------

class TestQualityFactors:
    def test_roe(self, daily_data, financial_data):
        f = ROEFactor()
        result = f.compute(daily_data, TARGET_DATE, financial_data)
        assert len(result) == 3
        assert f.category == "quality"

    def test_gross_margin(self, daily_data, financial_data):
        f = GrossMarginFactor()
        result = f.compute(daily_data, TARGET_DATE, financial_data)
        assert len(result) == 3

    def test_debt_ratio(self, daily_data, financial_data):
        f = DebtRatioFactor()
        result = f.compute(daily_data, TARGET_DATE, financial_data)
        assert len(result) == 3
        # Should be negative (negated debt ratio)
        assert (result < 0).all()


# ---------------------------------------------------------------------------
# Growth factors
# ---------------------------------------------------------------------------

class TestGrowthFactors:
    def test_revenue_growth(self, daily_data, financial_data):
        f = RevenueGrowthFactor()
        result = f.compute(daily_data, TARGET_DATE, financial_data)
        assert len(result) == 3
        assert f.category == "growth"

    def test_profit_growth(self, daily_data, financial_data):
        f = ProfitGrowthFactor()
        result = f.compute(daily_data, TARGET_DATE, financial_data)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Technical factors
# ---------------------------------------------------------------------------

class TestTechnicalFactors:
    def test_rsi(self, daily_data):
        f = RSIFactor(window=14)
        result = f.compute(daily_data, TARGET_DATE)
        assert len(result) == 3
        assert f.category == "technical"
        # RSI should be between 0 and 100
        assert (result >= 0).all() and (result <= 100).all()

    def test_macd(self, daily_data):
        f = MACDFactor()
        result = f.compute(daily_data, TARGET_DATE)
        assert len(result) == 3
        assert not result.isna().all()

    def test_bollinger(self, daily_data):
        f = BollingerFactor(window=20)
        result = f.compute(daily_data, TARGET_DATE)
        assert len(result) == 3
        assert not result.isna().all()
        # Should be roughly in [-1, 1] range for normal data
        assert (result.abs() < 2).all()
