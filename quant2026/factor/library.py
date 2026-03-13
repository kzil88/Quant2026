"""Built-in factor library: common factors for A-share market."""

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from .base import Factor


# ---------------------------------------------------------------------------
# Existing factors (updated signature)
# ---------------------------------------------------------------------------

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

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
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

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
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

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
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

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        df = data[data["date"] <= str(target_date)].copy()
        latest = df.groupby("stock_code").tail(self._window)

        def _corr(g):
            if len(g) < 5:
                return np.nan
            return g["close"].pct_change().corr(g["volume"].pct_change())

        return latest.groupby("stock_code").apply(_corr)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_financial_field(financial_data: pd.DataFrame | None, field: str) -> pd.Series:
    """Extract a field from financial_data, indexed by stock_code."""
    if financial_data is None or field not in financial_data.columns:
        logger.warning(f"financial_data missing or no '{field}' column")
        return pd.Series(dtype=float)
    return financial_data.set_index("stock_code")[field]


def _get_latest_close(data: pd.DataFrame, target_date: date) -> pd.Series:
    """Get latest close price per stock as of target_date."""
    df = data[data["date"] <= str(target_date)]
    idx = df.groupby("stock_code")["date"].idxmax()
    return df.loc[idx].set_index("stock_code")["close"]


# ---------------------------------------------------------------------------
# Value factors (价值因子)
# ---------------------------------------------------------------------------

class PEFactor(Factor):
    """Earnings-to-price ratio (EP = eps / close)."""

    @property
    def name(self) -> str:
        return "ep"

    @property
    def category(self) -> str:
        return "value"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        close = _get_latest_close(data, target_date)
        eps = _get_financial_field(financial_data, "eps")
        common = close.index.intersection(eps.index)
        return (eps[common] / close[common]).rename(self.name)


class PBFactor(Factor):
    """Book-to-price ratio (BP = bps / close)."""

    @property
    def name(self) -> str:
        return "bp"

    @property
    def category(self) -> str:
        return "value"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        close = _get_latest_close(data, target_date)
        bps = _get_financial_field(financial_data, "bps")
        common = close.index.intersection(bps.index)
        return (bps[common] / close[common]).rename(self.name)


class DividendYieldFactor(Factor):
    """Dividend yield (dps / close)."""

    @property
    def name(self) -> str:
        return "dividend_yield"

    @property
    def category(self) -> str:
        return "value"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        close = _get_latest_close(data, target_date)
        dps = _get_financial_field(financial_data, "dps")
        common = close.index.intersection(dps.index)
        return (dps[common] / close[common]).rename(self.name)


# ---------------------------------------------------------------------------
# Quality factors (质量因子)
# ---------------------------------------------------------------------------

class ROEFactor(Factor):
    """Return on equity."""

    @property
    def name(self) -> str:
        return "roe"

    @property
    def category(self) -> str:
        return "quality"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        return _get_financial_field(financial_data, "roe").rename(self.name)


class GrossMarginFactor(Factor):
    """Gross profit margin."""

    @property
    def name(self) -> str:
        return "gross_margin"

    @property
    def category(self) -> str:
        return "quality"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        return _get_financial_field(financial_data, "gross_margin").rename(self.name)


class DebtRatioFactor(Factor):
    """Negative debt-to-asset ratio (lower debt is better)."""

    @property
    def name(self) -> str:
        return "neg_debt_ratio"

    @property
    def category(self) -> str:
        return "quality"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        dr = _get_financial_field(financial_data, "debt_ratio")
        return (-dr).rename(self.name)


# ---------------------------------------------------------------------------
# Growth factors (成长因子)
# ---------------------------------------------------------------------------

class RevenueGrowthFactor(Factor):
    """Year-over-year revenue growth."""

    @property
    def name(self) -> str:
        return "revenue_growth"

    @property
    def category(self) -> str:
        return "growth"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        return _get_financial_field(financial_data, "revenue_growth").rename(self.name)


class ProfitGrowthFactor(Factor):
    """Year-over-year net profit growth."""

    @property
    def name(self) -> str:
        return "profit_growth"

    @property
    def category(self) -> str:
        return "growth"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        return _get_financial_field(financial_data, "profit_growth").rename(self.name)


# ---------------------------------------------------------------------------
# Technical factors (技术因子)
# ---------------------------------------------------------------------------

class RSIFactor(Factor):
    """14-day Relative Strength Index."""

    def __init__(self, window: int = 14):
        self._window = window

    @property
    def name(self) -> str:
        return f"rsi_{self._window}d"

    @property
    def category(self) -> str:
        return "technical"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        df = data[data["date"] <= str(target_date)].copy()
        # Need window+1 rows to get window deltas
        latest = df.groupby("stock_code").tail(self._window + 1)

        def _rsi(g: pd.DataFrame) -> float:
            closes = g.sort_values("date")["close"]
            delta = closes.diff().dropna()
            if len(delta) < self._window:
                return np.nan
            gain = delta.clip(lower=0).rolling(self._window).mean().iloc[-1]
            loss = (-delta.clip(upper=0)).rolling(self._window).mean().iloc[-1]
            if loss == 0:
                return 100.0
            rs = gain / loss
            return 100 - 100 / (1 + rs)

        return latest.groupby("stock_code").apply(_rsi)


class MACDFactor(Factor):
    """MACD histogram (DIF - DEA). DIF=EMA12-EMA26, DEA=EMA9(DIF)."""

    @property
    def name(self) -> str:
        return "macd_hist"

    @property
    def category(self) -> str:
        return "technical"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        df = data[data["date"] <= str(target_date)].copy()
        # Need enough history for EMA26 + EMA9
        latest = df.groupby("stock_code").tail(60)

        def _macd(g: pd.DataFrame) -> float:
            closes = g.sort_values("date")["close"]
            if len(closes) < 26:
                return np.nan
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            return (dif.iloc[-1] - dea.iloc[-1])

        return latest.groupby("stock_code").apply(_macd)


class BollingerFactor(Factor):
    """Position within Bollinger Bands: (close - mid) / (upper - lower)."""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self._window = window
        self._num_std = num_std

    @property
    def name(self) -> str:
        return "bollinger_pos"

    @property
    def category(self) -> str:
        return "technical"

    def compute(self, data: pd.DataFrame, target_date: date, financial_data: pd.DataFrame | None = None) -> pd.Series:
        df = data[data["date"] <= str(target_date)].copy()
        latest = df.groupby("stock_code").tail(self._window)

        def _boll(g: pd.DataFrame) -> float:
            closes = g.sort_values("date")["close"]
            if len(closes) < self._window:
                return np.nan
            mid = closes.mean()
            std = closes.std()
            upper = mid + self._num_std * std
            lower = mid - self._num_std * std
            band_width = upper - lower
            if band_width == 0:
                return 0.0
            return (closes.iloc[-1] - mid) / band_width

        return latest.groupby("stock_code").apply(_boll)
