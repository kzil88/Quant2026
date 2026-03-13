"""Built-in factor library: common factors for A-share market."""

from datetime import date

import numpy as np
import pandas as pd

from .base import Factor


class MomentumFactor(Factor):
    """N-day price momentum (动量因子)."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"momentum_{self._window}d"

    @property
    def category(self) -> str:
        return "momentum"

    def compute(self, data: pd.DataFrame, target_date: date) -> pd.Series:
        df = data[data["date"] <= str(target_date)].copy()
        latest = df.groupby("stock_code").tail(self._window + 1)
        returns = latest.groupby("stock_code")["close"].apply(
            lambda x: x.iloc[-1] / x.iloc[0] - 1 if len(x) > 1 else np.nan
        )
        return returns


class VolatilityFactor(Factor):
    """N-day return volatility (波动率因子)."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"volatility_{self._window}d"

    @property
    def category(self) -> str:
        return "volatility"

    def compute(self, data: pd.DataFrame, target_date: date) -> pd.Series:
        df = data[data["date"] <= str(target_date)].copy()
        latest = df.groupby("stock_code").tail(self._window)
        ret = latest.groupby("stock_code")["close"].pct_change()
        return ret.groupby(latest["stock_code"]).std()


class TurnoverFactor(Factor):
    """Average turnover ratio (换手率因子)."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"turnover_{self._window}d"

    @property
    def category(self) -> str:
        return "liquidity"

    def compute(self, data: pd.DataFrame, target_date: date) -> pd.Series:
        df = data[data["date"] <= str(target_date)].copy()
        latest = df.groupby("stock_code").tail(self._window)
        return latest.groupby("stock_code")["volume"].mean()


class VolumePriceFactor(Factor):
    """Volume-price correlation (量价相关性因子)."""

    def __init__(self, window: int = 20):
        self._window = window

    @property
    def name(self) -> str:
        return f"vol_price_corr_{self._window}d"

    @property
    def category(self) -> str:
        return "technical"

    def compute(self, data: pd.DataFrame, target_date: date) -> pd.Series:
        df = data[data["date"] <= str(target_date)].copy()
        latest = df.groupby("stock_code").tail(self._window)

        def _corr(g):
            if len(g) < 5:
                return np.nan
            return g["close"].pct_change().corr(g["volume"].pct_change())

        return latest.groupby("stock_code").apply(_corr)
