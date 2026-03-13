"""Factor registry: manage and compute all registered factors."""

from datetime import date

import pandas as pd
from loguru import logger

from .base import Factor
from .preprocessing import FactorPreprocessor


class FactorRegistry:
    """Central registry for all factors."""

    def __init__(self):
        self._factors: dict[str, Factor] = {}

    def register(self, factor: Factor) -> None:
        self._factors[factor.name] = factor
        logger.info(f"Registered factor: {factor.name} ({factor.category})")

    def compute_all(
        self,
        data: pd.DataFrame,
        target_date: date,
        industry: pd.Series | None = None,
        preprocess: bool = True,
    ) -> pd.DataFrame:
        """Compute all registered factors, return a stock_code x factor matrix."""
        results = {}
        for name, factor in self._factors.items():
            try:
                values = factor.compute(data, target_date)
                results[name] = values
            except Exception as e:
                logger.error(f"Factor {name} failed: {e}")

        df = pd.DataFrame(results)

        if preprocess:
            preprocessor = FactorPreprocessor()
            df = preprocessor.full_pipeline(df, industry=industry)

        return df

    @property
    def factor_names(self) -> list[str]:
        return list(self._factors.keys())
