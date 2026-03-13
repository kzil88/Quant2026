"""Training utilities for ML strategy."""

from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger


class MLTrainer:
    """Helper to build (features, labels) datasets for ML strategies.

    Args:
        forward_days: Number of forward trading days for return label.
    """

    def __init__(self, forward_days: int = 20) -> None:
        self.forward_days = forward_days

    def build_dataset(
        self,
        data: pd.DataFrame,
        factor_computer: callable,
        dates: list[date],
    ) -> tuple[dict[date, pd.DataFrame], dict[date, pd.Series]]:
        """Build factor matrices and forward-return labels for given dates.

        Args:
            data: Daily OHLCV DataFrame with columns ``[stock_code, date, close, ...]``.
            factor_computer: ``(data, date) -> pd.DataFrame`` returning factor matrix
                indexed by stock_code.
            dates: List of dates to build samples for.

        Returns:
            ``(factor_matrices, returns)`` dicts keyed by date.
        """
        factor_matrices: dict[date, pd.DataFrame] = {}
        returns: dict[date, pd.Series] = {}

        # Pre-compute forward returns per stock
        fwd_returns = self._compute_forward_returns(data)

        for dt in dates:
            try:
                fm = factor_computer(data, dt)
            except Exception as e:
                logger.debug(f"Factor computation failed for {dt}: {e}")
                continue
            if fm is None or fm.empty:
                continue

            dt_str = str(dt)
            if dt_str not in fwd_returns:
                continue
            ret = fwd_returns[dt_str]
            common = fm.index.intersection(ret.index)
            if common.empty:
                continue

            factor_matrices[dt] = fm.loc[common]
            returns[dt] = ret.loc[common]

        logger.info(f"MLTrainer.build_dataset: {len(factor_matrices)} dates with valid data")
        return factor_matrices, returns

    def _compute_forward_returns(self, data: pd.DataFrame) -> dict[str, pd.Series]:
        """Compute forward N-day returns for each trading date.

        Returns:
            ``{date_str: Series(stock_code -> forward return)}``
        """
        df = data[["stock_code", "date", "close"]].copy()
        df["date"] = df["date"].astype(str)
        pivot = df.pivot_table(index="date", columns="stock_code", values="close")
        pivot = pivot.sort_index()

        fwd = pivot.shift(-self.forward_days) / pivot - 1
        result: dict[str, pd.Series] = {}
        for dt_str in fwd.index:
            row = fwd.loc[dt_str].dropna()
            if not row.empty:
                result[dt_str] = row
        return result
