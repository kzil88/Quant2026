"""Markowitz Mean-Variance Portfolio Optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize


class MarkowitzOptimizer:
    """Markowitz Mean-Variance Portfolio Optimization.

    Uses scipy SLSQP to find optimal weights on the efficient frontier.
    Supports max_sharpe, min_variance, and target_return methods.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.025,
        max_single_weight: float = 0.10,
        min_weight: float = 0.01,
        max_stocks: int = 30,
    ):
        self.risk_free_rate = risk_free_rate
        self.max_single_weight = max_single_weight
        self.min_weight = min_weight
        self.max_stocks = max_stocks

    def optimize(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        method: str = "max_sharpe",
        target_return: float | None = None,
        current_weights: pd.Series | None = None,
        turnover_penalty_weight: float = 0.0,
    ) -> pd.Series:
        """Return optimal weights (stock_code -> weight).

        Args:
            expected_returns: Expected return per stock.
            cov_matrix: Covariance matrix of returns.
            method: "max_sharpe" | "min_variance" | "target_return"
            target_return: Required when method="target_return".

        Returns:
            pd.Series of weights summing to 1.
        """
        # Align universe
        common = expected_returns.index.intersection(cov_matrix.index)
        if len(common) == 0:
            raise ValueError("No common stocks between returns and cov_matrix")

        # Select top N by expected return
        top = expected_returns.loc[common].nlargest(self.max_stocks)
        stocks = top.index.tolist()
        n = len(stocks)

        if n == 1:
            return pd.Series([1.0], index=stocks)

        mu = expected_returns.loc[stocks].values.astype(float)
        sigma = cov_matrix.loc[stocks, stocks].values.astype(float)
        sigma = _ensure_positive_definite(sigma)

        rf = self.risk_free_rate
        bounds = [(self.min_weight, self.max_single_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        # Prepare current weights vector for turnover penalty
        cw = np.zeros(n)
        if current_weights is not None and turnover_penalty_weight > 0:
            cw_aligned = current_weights.reindex(stocks, fill_value=0.0).values.astype(float)
            cw = cw_aligned

        w0 = np.ones(n) / n

        if method == "min_variance":
            obj = lambda w: w @ sigma @ w + turnover_penalty_weight * np.sum(np.abs(w - cw))
        elif method == "target_return":
            if target_return is None:
                raise ValueError("target_return required for method='target_return'")
            obj = lambda w: w @ sigma @ w + turnover_penalty_weight * np.sum(np.abs(w - cw))
            constraints.append({"type": "eq", "fun": lambda w: w @ mu - target_return})
        else:  # max_sharpe
            def neg_sharpe(w: np.ndarray) -> float:
                ret = w @ mu
                vol = np.sqrt(w @ sigma @ w)
                penalty = turnover_penalty_weight * np.sum(np.abs(w - cw))
                return -(ret - rf) / vol + penalty if vol > 1e-12 else 1e6
            obj = neg_sharpe

        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-12})

        if not res.success:
            logger.warning(f"Markowitz optimization did not converge: {res.message}")

        weights = np.maximum(res.x, 0.0)
        weights /= weights.sum()

        result = pd.Series(weights, index=stocks)
        result = result[result > 1e-6]
        result /= result.sum()

        logger.info(f"Markowitz({method}): {len(result)} stocks, sharpe proxy={_sharpe(result.values, mu[:len(result)], sigma[:len(result),:len(result)], rf):.4f}")
        return result

    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        n_points: int = 50,
    ) -> pd.DataFrame:
        """Compute the efficient frontier.

        Returns:
            DataFrame with columns [return, risk, sharpe, weights].
        """
        common = expected_returns.index.intersection(cov_matrix.index)
        top = expected_returns.loc[common].nlargest(self.max_stocks)
        stocks = top.index.tolist()
        n = len(stocks)

        mu = expected_returns.loc[stocks].values.astype(float)
        sigma = cov_matrix.loc[stocks, stocks].values.astype(float)
        sigma = _ensure_positive_definite(sigma)

        # Get min/max feasible returns
        min_ret = self.optimize(expected_returns, cov_matrix, method="min_variance")
        min_r = float(min_ret.reindex(stocks, fill_value=0).values @ mu)
        max_r = float(mu.max() * self.max_single_weight + mu.mean() * (1 - self.max_single_weight))

        target_returns = np.linspace(min_r, max_r * 0.95, n_points)
        rows = []

        for tr in target_returns:
            try:
                w = self.optimize(expected_returns, cov_matrix, method="target_return", target_return=tr)
                wv = w.reindex(stocks, fill_value=0).values
                ret = float(wv @ mu)
                risk = float(np.sqrt(wv @ sigma @ wv))
                sharpe = (ret - self.risk_free_rate) / risk if risk > 1e-12 else 0.0
                rows.append({"return": ret, "risk": risk, "sharpe": sharpe, "weights": w.to_dict()})
            except Exception:
                continue

        return pd.DataFrame(rows)


def _sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray, rf: float) -> float:
    ret = w @ mu
    vol = np.sqrt(w @ sigma @ w)
    return (ret - rf) / vol if vol > 1e-12 else 0.0


def _ensure_positive_definite(m: np.ndarray) -> np.ndarray:
    """Fix non-positive-definite matrix via eigenvalue clipping."""
    try:
        np.linalg.cholesky(m)
        return m
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(m)
        eigvals = np.maximum(eigvals, 1e-8)
        fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
        logger.debug("Fixed non-positive-definite covariance matrix")
        return (fixed + fixed.T) / 2
