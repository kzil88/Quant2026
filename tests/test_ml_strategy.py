"""Tests for ML strategy (fit + generate) with mock data."""

import numpy as np
import pandas as pd
import pytest
from datetime import date

from quant2026.strategy.ml_model.strategy import MLStrategy
from quant2026.strategy.ml_model.trainer import MLTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_data(n_stocks: int = 50, n_days: int = 300) -> pd.DataFrame:
    """Create synthetic OHLCV data."""
    rng = np.random.RandomState(42)
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    records = []
    for code in [f"{i:06d}" for i in range(1, n_stocks + 1)]:
        price = 10.0 + rng.randn() * 2
        for dt in dates:
            ret = rng.randn() * 0.02
            price *= (1 + ret)
            records.append({
                "stock_code": code,
                "date": str(dt.date()),
                "open": price * (1 + rng.randn() * 0.005),
                "high": price * (1 + abs(rng.randn()) * 0.01),
                "low": price * (1 - abs(rng.randn()) * 0.01),
                "close": price,
                "volume": rng.randint(1000, 100000),
            })
    return pd.DataFrame(records)


def _make_mock_factors(data: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Create simple mock factors from data."""
    rng = np.random.RandomState(hash(str(target_date)) % 2**31)
    df = data[data["date"] <= str(target_date)]
    stocks = df["stock_code"].unique()
    n_factors = 5
    return pd.DataFrame(
        rng.randn(len(stocks), n_factors),
        index=stocks,
        columns=[f"factor_{i}" for i in range(n_factors)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLStrategy:
    def _fit_strategy(self, model_type: str = "lightgbm") -> tuple[MLStrategy, pd.DataFrame]:
        data = _make_mock_data(n_stocks=30, n_days=200)
        strategy = MLStrategy(model_type=model_type, forward_days=10, top_n=5)

        # Build training data
        all_dates = sorted(data["date"].unique())
        train_dates = [date.fromisoformat(d) for d in all_dates[60:150:10]]

        factor_matrices = {}
        returns_dict = {}
        trainer = MLTrainer(forward_days=10)
        fwd = trainer._compute_forward_returns(data)

        for dt in train_dates:
            fm = _make_mock_factors(data, dt)
            dt_str = str(dt)
            if dt_str in fwd:
                common = fm.index.intersection(fwd[dt_str].index)
                if not common.empty:
                    factor_matrices[dt] = fm.loc[common]
                    returns_dict[dt] = fwd[dt_str].loc[common]

        strategy.fit(factor_matrices, returns_dict)
        return strategy, data

    def test_fit_and_generate_lightgbm(self):
        strategy, data = self._fit_strategy("lightgbm")
        assert strategy._model is not None
        assert strategy.name == "ml_lightgbm"

        target = date(2023, 8, 1)
        fm = _make_mock_factors(data, target)
        result = strategy.generate(data, fm, target)
        assert len(result.scores) > 0
        assert "feature_importance" in result.metadata

    def test_fit_and_generate_xgboost(self):
        strategy, data = self._fit_strategy("xgboost")
        assert strategy._model is not None
        assert strategy.name == "ml_xgboost"

        target = date(2023, 8, 1)
        fm = _make_mock_factors(data, target)
        result = strategy.generate(data, fm, target)
        assert len(result.scores) > 0

    def test_generate_without_fit_raises(self):
        strategy = MLStrategy()
        with pytest.raises(RuntimeError, match="not fitted"):
            strategy.generate(pd.DataFrame(), pd.DataFrame(), date.today())

    def test_top_n_signals(self):
        strategy, data = self._fit_strategy("lightgbm")
        target = date(2023, 8, 1)
        fm = _make_mock_factors(data, target)
        result = strategy.generate(data, fm, target)
        from quant2026.types import Signal
        buy_count = (result.signals == Signal.BUY).sum()
        assert buy_count <= strategy.top_n

    def test_invalid_model_type(self):
        with pytest.raises(ValueError):
            MLStrategy(model_type="random_forest")


class TestMLTrainer:
    def test_build_dataset(self):
        data = _make_mock_data(n_stocks=20, n_days=150)
        trainer = MLTrainer(forward_days=10)
        all_dates = sorted(data["date"].unique())
        dates = [date.fromisoformat(d) for d in all_dates[30:100:10]]

        fm_dict, ret_dict = trainer.build_dataset(
            data,
            factor_computer=_make_mock_factors,
            dates=dates,
        )
        assert len(fm_dict) > 0
        assert len(ret_dict) > 0
        assert set(fm_dict.keys()) == set(ret_dict.keys())
