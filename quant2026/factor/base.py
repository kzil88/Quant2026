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
    def compute(self, data: pd.DataFrame, target_date: date) -> pd.Series:
        """Compute factor values for all stocks.

        Args:
            data: cleaned daily quotes (output from data layer)
            target_date: compute factor as of this date

        Returns:
            Series: stock_code -> factor_value
        """
        ...
