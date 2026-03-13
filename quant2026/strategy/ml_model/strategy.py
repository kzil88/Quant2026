"""ML-based stock selection strategy using LightGBM or XGBoost."""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from quant2026.strategy.base import Strategy
from quant2026.types import Signal, StrategyResult


class MLStrategy(Strategy):
    """Machine-learning stock selection strategy.

    Uses LightGBM or XGBoost to predict forward returns from factor features.
    Call ``fit()`` with historical data before ``generate()``.

    Args:
        model_type: ``"lightgbm"`` or ``"xgboost"``.
        forward_days: Number of forward trading days for return label.
        train_window: Training window in trading days (unused internally but
            exposed for pipeline control).
        top_n: Number of stocks to select.
        model_params: Override default hyper-parameters.
    """

    DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
        "lightgbm": dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                         random_state=42, verbosity=-1, n_jobs=-1),
        "xgboost": dict(n_estimators=200, max_depth=5, learning_rate=0.05,
                        random_state=42, verbosity=0, n_jobs=-1),
    }

    def __init__(
        self,
        model_type: str = "lightgbm",
        forward_days: int = 20,
        train_window: int = 252,
        top_n: int = 30,
        model_params: dict | None = None,
    ) -> None:
        if model_type not in ("lightgbm", "xgboost"):
            raise ValueError(f"model_type must be 'lightgbm' or 'xgboost', got '{model_type}'")
        self._model_type = model_type
        self.forward_days = forward_days
        self.train_window = train_window
        self.top_n = top_n
        self._model_params = {**self.DEFAULT_PARAMS[model_type], **(model_params or {})}
        self._model: Any = None
        self._feature_names: list[str] = []
        self._feature_importances: pd.Series | None = None

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"ml_{self._model_type}"

    def generate(
        self,
        data: pd.DataFrame,
        factor_matrix: pd.DataFrame | None,
        target_date: date,
    ) -> StrategyResult:
        """Predict forward returns and rank stocks.

        ``fit()`` must be called before ``generate()``.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() before generate().")
        if factor_matrix is None or factor_matrix.empty:
            raise ValueError("factor_matrix is required for MLStrategy.generate()")

        # Align features
        fm = factor_matrix.reindex(columns=self._feature_names)
        valid = fm.dropna()
        if valid.empty:
            return StrategyResult(name=self.name, date=target_date, scores=pd.Series(dtype=float))

        preds = pd.Series(self._model.predict(valid), index=valid.index, name="pred_return")
        # Top-N signals
        top = preds.nlargest(self.top_n).index
        signals = pd.Series(Signal.HOLD, index=preds.index)
        signals[top] = Signal.BUY

        metadata: dict[str, Any] = {"model_type": self._model_type}
        if self._feature_importances is not None:
            metadata["feature_importance"] = self._feature_importances.to_dict()

        return StrategyResult(
            name=self.name,
            date=target_date,
            scores=preds,
            signals=signals,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        historical_factor_matrices: dict[date, pd.DataFrame],
        returns: dict[date, pd.Series],
    ) -> None:
        """Train the model on historical factor values and forward returns.

        Args:
            historical_factor_matrices: ``{date: DataFrame(stock_code x factors)}``
            returns: ``{date: Series(stock_code -> forward return)}``
        """
        t0 = time.perf_counter()

        # Build training set
        X_parts: list[pd.DataFrame] = []
        y_parts: list[pd.Series] = []
        common_dates = sorted(set(historical_factor_matrices) & set(returns))

        for dt in common_dates:
            fm = historical_factor_matrices[dt]
            ret = returns[dt]
            common_idx = fm.index.intersection(ret.index)
            if common_idx.empty:
                continue
            X_parts.append(fm.loc[common_idx])
            y_parts.append(ret.loc[common_idx])

        if not X_parts:
            raise ValueError("No valid training samples after aligning factors and returns.")

        X = pd.concat(X_parts)
        y = pd.concat(y_parts)

        # Drop NaN
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]
        if len(X) < 10:
            raise ValueError(f"Too few training samples ({len(X)}). Need at least 10.")

        self._feature_names = list(X.columns)

        # Build model
        model = self._create_model()
        model.fit(X, y)
        self._model = model

        # Feature importance
        imp = model.feature_importances_
        self._feature_importances = pd.Series(imp, index=self._feature_names).sort_values(ascending=False)

        elapsed = time.perf_counter() - t0
        logger.info(
            f"MLStrategy.fit done: model={self._model_type}, samples={len(X)}, "
            f"features={len(self._feature_names)}, time={elapsed:.2f}s"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_model(self) -> Any:
        if self._model_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self._model_params)
        else:
            from xgboost import XGBRegressor
            return XGBRegressor(**self._model_params)
