"""Tests for FactorEvaluator."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant2026.factor.evaluation import FactorEvaluator


@pytest.fixture
def evaluator():
    return FactorEvaluator()


def _make_stocks(n=50):
    return [f"S{i:03d}" for i in range(n)]


class TestComputeICSeries:
    """Test Rank IC computation."""

    def test_perfect_positive_correlation(self, evaluator):
        """If factor perfectly predicts returns, IC ~ 1."""
        stocks = _make_stocks(100)
        dt = date(2024, 6, 1)
        vals = pd.Series(np.arange(100, dtype=float), index=stocks)
        # Forward returns perfectly correlated
        fwd = pd.Series(np.arange(100, dtype=float), index=stocks)
        ic_s = evaluator.compute_ic_series({dt: vals}, {dt: fwd})
        assert len(ic_s) == 1
        assert ic_s.iloc[0] == pytest.approx(1.0, abs=0.01)

    def test_perfect_negative_correlation(self, evaluator):
        stocks = _make_stocks(100)
        dt = date(2024, 6, 1)
        vals = pd.Series(np.arange(100, dtype=float), index=stocks)
        fwd = pd.Series(np.arange(100, dtype=float)[::-1], index=stocks)
        ic_s = evaluator.compute_ic_series({dt: vals}, {dt: fwd})
        assert ic_s.iloc[0] == pytest.approx(-1.0, abs=0.01)

    def test_random_factor_ic_near_zero(self, evaluator):
        """Random factor should have IC close to 0 on average."""
        rng = np.random.default_rng(42)
        stocks = _make_stocks(200)
        ic_list = []
        for i in range(50):
            dt = date(2024, 1, 1 + i) if i < 28 else date(2024, 2, i - 27)
            # Use unique dates; exact date doesn't matter
            dt = date(2024, 1, 1) + pd.Timedelta(days=i)
            dt = dt.date() if hasattr(dt, 'date') else dt
            vals = pd.Series(rng.standard_normal(200), index=stocks)
            fwd = pd.Series(rng.standard_normal(200), index=stocks)
            ic_s = evaluator.compute_ic_series({dt: vals}, {dt: fwd})
            if len(ic_s) > 0:
                ic_list.append(ic_s.iloc[0])
        mean_ic = np.mean(ic_list)
        assert abs(mean_ic) < 0.1  # should be close to 0

    def test_missing_dates_skipped(self, evaluator):
        stocks = _make_stocks(30)
        dt1 = date(2024, 1, 1)
        dt2 = date(2024, 1, 2)
        fv = {dt1: pd.Series(np.arange(30.0), index=stocks)}
        fr = {dt2: pd.Series(np.arange(30.0), index=stocks)}  # different date
        ic_s = evaluator.compute_ic_series(fv, fr)
        assert len(ic_s) == 0

    def test_too_few_stocks_skipped(self, evaluator):
        stocks = _make_stocks(3)
        dt = date(2024, 1, 1)
        fv = {dt: pd.Series([1.0, 2.0, 3.0], index=stocks)}
        fr = {dt: pd.Series([3.0, 2.0, 1.0], index=stocks)}
        ic_s = evaluator.compute_ic_series(fv, fr)
        assert len(ic_s) == 0  # < 5 stocks


class TestIR:
    def test_basic_ir(self, evaluator):
        ic_s = pd.Series([0.05, 0.04, 0.06, 0.03, 0.05])
        ir = evaluator.compute_ir(ic_s)
        expected = ic_s.mean() / ic_s.std()
        assert ir == pytest.approx(expected, rel=1e-6)

    def test_zero_std(self, evaluator):
        ic_s = pd.Series([0.05, 0.05, 0.05])
        assert evaluator.compute_ir(ic_s) == 0.0


class TestICSummary:
    def test_summary_keys(self, evaluator):
        ic_s = pd.Series([0.03, -0.01, 0.05, 0.02, -0.03, 0.04])
        s = evaluator.ic_summary(ic_s)
        assert set(s.keys()) == {
            "ic_mean", "ic_std", "ir", "ic_positive_ratio",
            "ic_abs_gt_002_ratio", "t_stat", "n_periods",
        }
        assert s["n_periods"] == 6
        assert 0 < s["ic_positive_ratio"] < 1

    def test_empty_series(self, evaluator):
        s = evaluator.ic_summary(pd.Series(dtype=float))
        assert s["n_periods"] == 0
        assert np.isnan(s["ic_mean"])


class TestICDecay:
    def test_decay_returns_correct_shape(self, evaluator):
        """Build synthetic daily data and test decay."""
        rng = np.random.default_rng(42)
        stocks = _make_stocks(20)
        dates = pd.bdate_range("2024-01-01", periods=120)
        rows = []
        for s in stocks:
            price = 100.0
            for d in dates:
                price *= 1 + rng.normal(0, 0.02)
                rows.append({"stock_code": s, "date": str(d.date()), "close": price, "volume": 1000})
        data = pd.DataFrame(rows)

        dt = date(2024, 3, 1)
        fv = {dt: pd.Series(rng.standard_normal(20), index=stocks)}
        decay = evaluator.ic_decay(data, fv, periods=[5, 10, 20])
        assert list(decay.index) == [5, 10, 20]
        assert list(decay.columns) == ["ic_mean", "ic_std", "ir"]


class TestFactorCorrelation:
    def test_identity_correlation(self, evaluator):
        stocks = _make_stocks(50)
        vals = np.arange(50, dtype=float)
        fm = pd.DataFrame({"A": vals, "B": vals, "C": -vals}, index=stocks)
        corr = evaluator.factor_correlation(fm)
        assert corr.loc["A", "B"] == pytest.approx(1.0, abs=0.01)
        assert corr.loc["A", "C"] == pytest.approx(-1.0, abs=0.01)

    def test_symmetric(self, evaluator):
        rng = np.random.default_rng(42)
        stocks = _make_stocks(30)
        fm = pd.DataFrame({
            "X": rng.standard_normal(30),
            "Y": rng.standard_normal(30),
        }, index=stocks)
        corr = evaluator.factor_correlation(fm)
        assert corr.loc["X", "Y"] == pytest.approx(corr.loc["Y", "X"])


class TestGenerateReport:
    def test_report_creates_file(self, evaluator, tmp_path):
        ic_summaries = {
            "test_factor": evaluator.ic_summary(pd.Series([0.03, 0.05, -0.01, 0.04]))
        }
        ic_series_dict = {"test_factor": pd.Series([0.03, 0.05, -0.01, 0.04])}
        decay_dict = {"test_factor": pd.DataFrame(
            {"ic_mean": [0.04, 0.03], "ic_std": [0.02, 0.03], "ir": [2.0, 1.0]},
            index=pd.Index([5, 10], name="period"),
        )}
        corr = pd.DataFrame({"test_factor": [1.0]}, index=["test_factor"])
        path = evaluator.generate_report(
            ic_summaries, ic_series_dict, decay_dict, corr, str(tmp_path)
        )
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "test_factor" in content
        assert "<img" in content
