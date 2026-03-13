"""Base factor interface."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class Factor(ABC):
    """All factors implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique factor name."""
        ...

    @property
    def category(self) -> str:
        """Factor category: value / momentum / quality / growth / volatility / ..."""
        return "unknown"

    @abstractmethod
    def compute(
        self,
        data: pd.DataFrame,
        target_date: date,
        financial_data: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Compute factor values for all stocks.

        Args:
            data: cleaned daily quotes (stock_code, date, open, high, low, close, volume, amount)
            target_date: compute factor as of this date
            financial_data: optional financial statement data (stock_code, eps, bps, dps, roe, ...)

        Returns:
            Series: stock_code -> factor_value
        """
        ...
