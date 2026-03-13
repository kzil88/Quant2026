"""Base strategy interface."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from quant2026.types import StrategyResult


class Strategy(ABC):
    """All strategies implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(
        self,
        data: pd.DataFrame,
        factor_matrix: pd.DataFrame | None,
        target_date: date,
    ) -> StrategyResult:
        """Generate strategy scores/signals.

        Args:
            data: cleaned market data
            factor_matrix: preprocessed factor values (optional, not all strategies need it)
            target_date: as of this date

        Returns:
            StrategyResult with standardized scores
        """
        ...
