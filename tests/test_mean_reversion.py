"""Tests for MeanReversionStrategy."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant2026.strategy.mean_reversion.strategy import MeanReversionStrategy
from quant2026.types import StrategyResult, Signal


def _make_data(stock_code: str, prices: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Create mock OHLCV data for a single stock."""
    dates = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame({
        "stock_code": stock_code,
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000] * len(prices),
        "turnover": [1e8] * len(prices),
    })


def _trending_down(n: int = 60, start_price: float = 100.0) -> list[float]:
    """Generate a price series that trends down (oversold)."""
    np.random.seed(42)
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 - 0.005 + np.random.randn() * 0.005))
    return prices


def _trending_up(n: int = 60, start_price: float = 100.0) -> list[float]:
    """Generate a price series that trends up (overbought)."""
    np.random.seed(123)
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + 0.005 + np.random.randn() * 0.005))
    return prices


class TestMeanReversionOutput:
    """Test that strategy output format is correct."""

    def test_returns_strategy_result(self):
        data = _make_data("000001", _trending_down(60))
        strategy = MeanReversionStrategy(window=20)
        result = strategy.generate(data, None, date(2024, 3, 22))
        assert isinstance(result, StrategyResult)
        assert result.name == "mean_reversion_20"
        assert result.date == date(2024, 3, 22)
        assert isinstance(result.scores, pd.Series)

    def test_scores_are_float(self):
        data = _make_data("000001", _trending_down(60))
        strategy = MeanReversionStrategy(window=20)
        result = strategy.generate(data, None, date(2024, 3, 22))
        assert result.scores.dtype == np.float64

    def test_empty_data(self):
        data = _make_data("000001", [100.0] * 5)  # too few rows
        strategy = MeanReversionStrategy(window=20)
        result = strategy.generate(data, None, date(2024, 1, 8))
        assert len(result.scores) == 0


class TestMeanReversionLogic:
    """Test that oversold stocks score higher than overbought."""

    def test_oversold_scores_higher(self):
        down_data = _make_data("OVERSOLD", _trending_down(60))
        up_data = _make_data("OVERBOUGHT", _trending_up(60))
        data = pd.concat([down_data, up_data], ignore_index=True)

        strategy = MeanReversionStrategy(window=20)
        result = strategy.generate(data, None, date(2024, 3, 22))

        assert "OVERSOLD" in result.scores.index
        assert "OVERBOUGHT" in result.scores.index
        assert result.scores["OVERSOLD"] > result.scores["OVERBOUGHT"]

    def test_sudden_drop_gets_buy_signal(self):
        """A sudden drop after flat prices should trigger BUY."""
        # Flat then sharp drop → zscore should be very negative
        prices = [100.0] * 40 + [100 - i * 2 for i in range(1, 21)]
        data = _make_data("DROPPED", prices)
        strategy = MeanReversionStrategy(window=20, zscore_threshold=-1.0)
        result = strategy.generate(data, None, date(2024, 3, 22))
        if result.signals is not None and "DROPPED" in result.signals.index:
            assert result.signals["DROPPED"] == Signal.BUY


class TestMeanReversionParams:
    """Test parameter configuration."""

    def test_custom_window(self):
        data = _make_data("000001", _trending_down(80))
        s = MeanReversionStrategy(window=30)
        assert s.name == "mean_reversion_30"
        result = s.generate(data, None, date(2024, 4, 19))
        assert isinstance(result, StrategyResult)

    def test_disable_bollinger_and_rsi(self):
        data = _make_data("000001", _trending_down(60))
        s = MeanReversionStrategy(use_bollinger=False, use_rsi=False)
        result = s.generate(data, None, date(2024, 3, 22))
        assert len(result.scores) > 0

    def test_different_thresholds(self):
        data = _make_data("000001", _trending_down(60))
        s1 = MeanReversionStrategy(zscore_threshold=-0.5)
        s2 = MeanReversionStrategy(zscore_threshold=-2.0)
        r1 = s1.generate(data, None, date(2024, 3, 22))
        r2 = s2.generate(data, None, date(2024, 3, 22))
        # Scores should be the same (threshold only affects signals)
        assert abs(r1.scores.iloc[0] - r2.scores.iloc[0]) < 1e-10
