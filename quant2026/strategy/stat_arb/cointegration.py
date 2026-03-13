"""Cointegration analysis tools for pairs trading."""

from itertools import combinations
from math import log

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import coint


class CointegrationAnalyzer:
    """配对交易的协整分析工具。"""

    def __init__(self, significance: float = 0.05) -> None:
        self.significance = significance

    # ── public API ───────────────────────────────────────────────

    def test_pair(self, series_a: pd.Series, series_b: pd.Series) -> dict:
        """Engle-Granger 两步法检验一对股票的协整关系。

        Returns:
            dict with keys: cointegrated, p_value, hedge_ratio, half_life
        """
        a = series_a.dropna().values.astype(float)
        b = series_b.dropna().values.astype(float)
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        # Hedge ratio via OLS: a = hedge_ratio * b + intercept
        hedge_ratio = float(np.polyfit(b, a, 1)[0])

        # Cointegration test
        t_stat, p_value, _ = coint(a, b)
        p_value = float(p_value)

        # Spread & half-life
        spread = pd.Series(a - hedge_ratio * b)
        half_life = self.compute_half_life(spread)

        return {
            "cointegrated": p_value < self.significance,
            "p_value": p_value,
            "hedge_ratio": hedge_ratio,
            "half_life": half_life,
        }

    def find_pairs(
        self, price_matrix: pd.DataFrame, min_obs: int = 120
    ) -> list[dict]:
        """从价格矩阵中找所有协整对。

        Args:
            price_matrix: index=date, columns=stock_code, values=close
            min_obs: 最少观测数

        Returns:
            按 p_value 升序排列的配对列表。
        """
        # Drop columns with insufficient data
        valid = price_matrix.dropna(axis=1, thresh=min_obs)
        cols = list(valid.columns)
        logger.debug(f"Testing {len(cols)}C2 = {len(cols)*(len(cols)-1)//2} pairs")

        pairs: list[dict] = []
        for col_a, col_b in combinations(cols, 2):
            sa = valid[col_a].dropna()
            sb = valid[col_b].dropna()
            common = sa.index.intersection(sb.index)
            if len(common) < min_obs:
                continue
            sa, sb = sa.loc[common], sb.loc[common]

            try:
                result = self.test_pair(sa, sb)
            except Exception:
                continue

            if not result["cointegrated"]:
                continue

            corr = float(sa.corr(sb))
            pairs.append({
                "stock_a": col_a,
                "stock_b": col_b,
                "p_value": result["p_value"],
                "hedge_ratio": result["hedge_ratio"],
                "half_life": result["half_life"],
                "correlation": corr,
            })

        pairs.sort(key=lambda x: x["p_value"])
        logger.info(f"Found {len(pairs)} cointegrated pairs from {len(cols)} stocks")
        return pairs

    def compute_spread(
        self, series_a: pd.Series, series_b: pd.Series, hedge_ratio: float
    ) -> pd.Series:
        """计算标准化价差 (z-score)。"""
        spread = series_a - hedge_ratio * series_b
        mean = spread.mean()
        std = spread.std()
        if std == 0 or not np.isfinite(std):
            return pd.Series(0.0, index=spread.index)
        return (spread - mean) / std

    def compute_half_life(self, spread: pd.Series) -> float:
        """均值回归半衰期 (Ornstein-Uhlenbeck AR(1))。"""
        spread = spread.dropna()
        if len(spread) < 3:
            return float("inf")
        lag = spread.shift(1).dropna()
        cur = spread.iloc[1:]
        # AR(1): cur = rho * lag + eps
        rho = float(np.polyfit(lag.values, cur.values, 1)[0])
        if rho >= 1 or rho <= 0:
            return float("inf")
        return -log(2) / log(rho)
