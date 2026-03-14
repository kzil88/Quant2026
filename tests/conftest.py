"""Shared pytest fixtures for Quant2026 test suite.

Provides reusable mock data generators for OHLCV, factor matrices,
returns, configs, and stock pools — all deterministic, no network needed.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from loguru import logger


# ── Stock Pools ──────────────────────────────────────────────────


@pytest.fixture
def sample_stock_codes() -> list[str]:
    """30只蓝筹股代码（沪深混合）。"""
    return [
        "600519", "601318", "000858", "000333", "600036",
        "601166", "600276", "000651", "601888", "600887",
        "002415", "000568", "600309", "601012", "600900",
        "002304", "000002", "601398", "600000", "601288",
        "000001", "002714", "600585", "601668", "600048",
        "002352", "600031", "601899", "000725", "601601",
    ]


# ── OHLCV Data ───────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """模拟 OHLCV 日线数据。

    Returns:
        DataFrame with columns=[stock_code, date, open, high, low, close, volume, amount]
        200 个交易日，10只股票，确定性随机种子。
    """
    rng = np.random.RandomState(42)
    stocks = [
        "600519", "601318", "000858", "000333", "600036",
        "000651", "601888", "600887", "002415", "000568",
    ]
    n_days = 200
    base_date = date(2025, 1, 2)
    trading_dates = [base_date + timedelta(days=i) for i in range(n_days * 2)]
    # 过滤掉周末
    trading_dates = [d for d in trading_dates if d.weekday() < 5][:n_days]

    rows: list[dict] = []
    for stock in stocks:
        base_price = rng.uniform(15, 200)
        for d in trading_dates:
            base_price *= 1 + rng.normal(0, 0.02)
            base_price = max(base_price, 1.0)
            o = base_price * (1 + rng.normal(0, 0.005))
            h = max(o, base_price) * (1 + abs(rng.normal(0, 0.008)))
            low = min(o, base_price) * (1 - abs(rng.normal(0, 0.008)))
            vol = rng.randint(50_000, 500_000)
            rows.append({
                "stock_code": stock,
                "date": str(d),
                "open": round(o, 2),
                "high": round(h, 2),
                "low": round(low, 2),
                "close": round(base_price, 2),
                "volume": vol,
                "amount": round(vol * base_price, 2),
            })

    df = pd.DataFrame(rows)
    logger.debug(f"sample_ohlcv_data: {len(df)} rows, {len(stocks)} stocks, {n_days} days")
    return df


# ── Factor Matrix ────────────────────────────────────────────────


@pytest.fixture
def sample_factor_matrix() -> pd.DataFrame:
    """模拟因子矩阵。

    Returns:
        DataFrame: index=date, columns=MultiIndex(stock, factor)
        50 个交易日，5只股票，3个因子。
    """
    rng = np.random.RandomState(123)
    stocks = ["600519", "601318", "000858", "000333", "600036"]
    factors = ["momentum", "value", "quality"]
    n_days = 50
    base_date = date(2025, 6, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_days * 2)]
    dates = [d for d in dates if d.weekday() < 5][:n_days]

    arrays = [
        [s for s in stocks for _ in factors],
        factors * len(stocks),
    ]
    columns = pd.MultiIndex.from_arrays(arrays, names=["stock", "factor"])
    data = rng.randn(n_days, len(stocks) * len(factors))
    return pd.DataFrame(data, index=dates, columns=columns)


# ── Returns ──────────────────────────────────────────────────────


@pytest.fixture
def sample_returns() -> pd.Series:
    """模拟日收益率序列（单资产，252 天）。"""
    rng = np.random.RandomState(99)
    n = 252
    base_date = date(2025, 1, 2)
    dates = pd.bdate_range(start=str(base_date), periods=n)
    returns = rng.normal(0.0003, 0.015, n)
    return pd.Series(returns, index=dates, name="daily_return")


# ── Config ───────────────────────────────────────────────────────


@pytest.fixture
def sample_config():
    """默认 Quant2026Config 实例。"""
    from quant2026.config import Quant2026Config, DataConfig, StrategyConfig

    return Quant2026Config(
        data=DataConfig(
            stock_pool=["600519", "601318", "000858"],
            start_date="2025-01-01",
            end_date="2025-06-30",
            cache_enabled=False,
        ),
        strategies=[
            StrategyConfig(name="multi_factor", type="MultiFactorStrategy"),
        ],
    )


# ── Temp Dirs ────────────────────────────────────────────────────


@pytest.fixture
def tmp_output_dir(tmp_path):
    """临时输出目录，测试结束后自动清理。"""
    out = tmp_path / "output"
    out.mkdir()
    return out
