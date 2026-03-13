"""Tests for stat_arb strategy & cointegration analyzer."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quant2026.strategy.stat_arb.cointegration import CointegrationAnalyzer
from quant2026.strategy.stat_arb.strategy import StatArbStrategy
from quant2026.types import StrategyResult


# ── Fixtures ─────────────────────────────────────────────────────

def _make_cointegrated_pair(n: int = 300, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    """Create two cointegrated series: b = random walk, a = 2*b + mean-reverting noise."""
    rng = np.random.RandomState(seed)
    b = 100 + np.cumsum(rng.randn(n) * 0.5)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.5 * noise[i - 1] + rng.randn()  # mean-reverting
    a = 2.0 * b + noise + 50
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


def _make_independent_pair(n: int = 300, seed: int = 99) -> tuple[pd.Series, pd.Series]:
    """Two independent random walks — should NOT be cointegrated."""
    rng = np.random.RandomState(seed)
    a = 100 + np.cumsum(rng.randn(n))
    b = 100 + np.cumsum(rng.randn(n))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


# ── CointegrationAnalyzer tests ──────────────────────────────────

class TestCointegrationAnalyzer:
    def test_cointegrated_pair_detected(self):
        analyzer = CointegrationAnalyzer(significance=0.05)
        a, b = _make_cointegrated_pair()
        result = analyzer.test_pair(a, b)
        assert result["cointegrated"] is True
        assert result["p_value"] < 0.05
        assert np.isfinite(result["hedge_ratio"])
        assert result["half_life"] > 0

    def test_independent_pair_rejected(self):
        analyzer = CointegrationAnalyzer(significance=0.05)
        a, b = _make_independent_pair()
        result = analyzer.test_pair(a, b)
        # Independent walks should generally not be cointegrated
        # (small chance of false positive, but seed is fixed)
        assert result["p_value"] > 0.05

    def test_spread_zscore(self):
        analyzer = CointegrationAnalyzer()
        a, b = _make_cointegrated_pair()
        result = analyzer.test_pair(a, b)
        spread = analyzer.compute_spread(a, b, result["hedge_ratio"])
        assert len(spread) == len(a)
        # Z-score should be roughly mean 0, std 1
        assert abs(spread.mean()) < 0.2
        assert abs(spread.std() - 1.0) < 0.2

    def test_half_life_positive(self):
        analyzer = CointegrationAnalyzer()
        a, b = _make_cointegrated_pair()
        spread = a - 2.0 * b
        hl = analyzer.compute_half_life(spread)
        assert hl > 0
        assert hl < 100  # Should be reasonable for mean-reverting process

    def test_find_pairs(self):
        a, b = _make_cointegrated_pair()
        c, _ = _make_independent_pair()
        matrix = pd.DataFrame({"A": a, "B": b, "C": c})
        analyzer = CointegrationAnalyzer()
        pairs = analyzer.find_pairs(matrix, min_obs=100)
        # Should find A-B pair
        assert len(pairs) >= 1
        pair_set = {(p["stock_a"], p["stock_b"]) for p in pairs}
        assert ("A", "B") in pair_set or ("B", "A") in pair_set


# ── StatArbStrategy tests ───────────────────────────────────────

class TestStatArbStrategy:
    def _build_data(self) -> pd.DataFrame:
        """Build minimal data DataFrame for strategy."""
        a, b = _make_cointegrated_pair(n=200)
        c, _ = _make_independent_pair(n=200)
        dates = a.index.strftime("%Y-%m-%d")
        rows = []
        for i, dt in enumerate(dates):
            rows.append({"stock_code": "A", "date": dt, "close": a.iloc[i], "open": a.iloc[i], "high": a.iloc[i], "low": a.iloc[i], "volume": 1e6})
            rows.append({"stock_code": "B", "date": dt, "close": b.iloc[i], "open": b.iloc[i], "high": b.iloc[i], "low": b.iloc[i], "volume": 1e6})
            rows.append({"stock_code": "C", "date": dt, "close": c.iloc[i], "open": c.iloc[i], "high": c.iloc[i], "low": c.iloc[i], "volume": 1e6})
        return pd.DataFrame(rows)

    def test_generate_returns_strategy_result(self):
        strategy = StatArbStrategy(lookback=120, max_pairs=5, recalc_interval=60)
        data = self._build_data()
        target = date(2023, 10, 1)
        result = strategy.generate(data, None, target)

        assert isinstance(result, StrategyResult)
        assert result.name == "stat_arb"
        assert result.date == target
        assert isinstance(result.scores, pd.Series)
        assert "active_pairs" in result.metadata

    def test_scores_are_non_negative(self):
        strategy = StatArbStrategy(lookback=120, entry_zscore=1.0)
        data = self._build_data()
        target = date(2023, 10, 1)
        result = strategy.generate(data, None, target)
        if not result.scores.empty:
            assert (result.scores >= 0).all()
