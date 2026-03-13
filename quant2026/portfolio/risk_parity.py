"""Risk Parity Portfolio Optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy.optimize import minimize


class RiskParityOptimizer:
    """Risk Parity: each asset contributes equal risk to the portfolio.

    Risk contribution: RC_i = w_i * (Σw)_i / (w'Σw)
    Objective: minimize deviation of RC_i from target budget.
    """

    def __init__(self, max_single_weight: float = 0.15):
        self.max_single_weight = max_single_weight

    def optimize(
        self,
        cov_matrix: pd.DataFrame,
        budget: pd.Series | None = None,
    ) -> pd.Series:
        """Return risk-parity weights.

        Args:
            cov_matrix: Covariance matrix.
            budget: Risk budget per asset (default: equal).

        Returns:
            pd.Series of weights summing to 1.
        """
        stocks = cov_matrix.index.tolist()
        n = len(stocks)

        if n == 1:
            return pd.Series([1.0], index=stocks)

        sigma = cov_matrix.values.astype(float)
        sigma = _ensure_positive_definite(sigma)

        if budget is not None:
            b = budget.reindex(stocks, fill_value=1.0 / n).values.astype(float)
        else:
            b = np.ones(n) / n
        b = b / b.sum()

        def objective(w: np.ndarray) -> float:
            port_var = w @ sigma @ w
            if port_var < 1e-16:
                return 0.0
            marginal = sigma @ w
            rc = w * marginal / port_var
            return float(np.sum((rc - b) ** 2))

        w0 = np.ones(n) / n
        bounds = [(1e-6, self.max_single_weight)] * n
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-14})

        if not res.success:
            logger.warning(f"Risk parity optimization did not converge: {res.message}")

        weights = np.maximum(res.x, 0.0)
        weights /= weights.sum()

        result = pd.Series(weights, index=stocks)
        logger.info(f"RiskParity: {n} stocks, max weight={result.max():.4f}")
        return result


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
