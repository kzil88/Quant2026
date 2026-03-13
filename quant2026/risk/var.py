"""VaR (Value at Risk) and CVaR (Conditional VaR) calculations."""

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm


class VaRCalculator:
    """Value at Risk / Conditional VaR 计算"""

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def historical_var(self, returns: pd.Series) -> float:
        """历史模拟法 VaR（返回负值，表示损失）"""
        return float(np.percentile(returns.dropna(), (1 - self.confidence) * 100))

    def parametric_var(self, returns: pd.Series) -> float:
        """参数法 VaR（假设正态分布，返回负值）"""
        r = returns.dropna()
        mu = r.mean()
        sigma = r.std()
        return float(mu + norm.ppf(1 - self.confidence) * sigma)

    def cvar(self, returns: pd.Series) -> float:
        """Conditional VaR (Expected Shortfall)：超过VaR后的平均损失"""
        var = self.historical_var(returns)
        r = returns.dropna()
        tail = r[r <= var]
        if tail.empty:
            return var
        return float(tail.mean())

    def rolling_var(self, returns: pd.Series, window: int = 60) -> pd.Series:
        """滚动 VaR 时序"""
        q = (1 - self.confidence) * 100
        return returns.rolling(window).apply(
            lambda x: np.percentile(x.dropna(), q) if len(x.dropna()) > 0 else np.nan,
            raw=False,
        )

    def portfolio_var(
        self,
        weights: pd.Series,
        returns_matrix: pd.DataFrame,
    ) -> dict:
        """组合级 VaR/CVaR

        Args:
            weights: stock_code -> weight
            returns_matrix: DatetimeIndex x stock_code 收益率矩阵

        Returns:
            {var_hist, var_param, cvar, component_var: {stock: contribution}}
        """
        common = weights.index.intersection(returns_matrix.columns)
        w = weights.loc[common]
        R = returns_matrix[common]

        # 组合收益率序列
        port_returns = R.dot(w)

        var_hist = self.historical_var(port_returns)
        var_param = self.parametric_var(port_returns)
        cvar_val = self.cvar(port_returns)

        # Component VaR: marginal contribution ≈ w_i * sigma_i * rho(i, port) / port_sigma * VaR
        cov = R.cov()
        port_sigma = float(np.sqrt(w.T @ cov @ w))
        component_var: dict[str, float] = {}
        if port_sigma > 0:
            marginal = cov @ w / port_sigma
            comp = w * marginal * (var_hist / port_sigma) if port_sigma != 0 else w * 0
            component_var = comp.to_dict()

        logger.info(
            f"Portfolio VaR(hist={var_hist:.4f}, param={var_param:.4f}), "
            f"CVaR={cvar_val:.4f}"
        )

        return {
            "var_hist": var_hist,
            "var_param": var_param,
            "cvar": cvar_val,
            "component_var": component_var,
        }
