"""Factor preprocessing: winsorize, standardize, neutralize."""

import numpy as np
import pandas as pd


class FactorPreprocessor:
    """Standard preprocessing pipeline for raw factor values."""

    @staticmethod
    def winsorize(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
        """Clip extreme values (去极值) using MAD method."""
        median = series.median()
        mad = (series - median).abs().median()
        upper = median + n_sigma * 1.4826 * mad
        lower = median - n_sigma * 1.4826 * mad
        return series.clip(lower, upper)

    @staticmethod
    def standardize(series: pd.Series) -> pd.Series:
        """Z-score standardization (标准化)."""
        std = series.std()
        if std == 0:
            return series * 0
        return (series - series.mean()) / std

    @staticmethod
    def neutralize(
        factor_df: pd.DataFrame, industry: pd.Series
    ) -> pd.DataFrame:
        """Industry + market-cap neutralization (行业中性化).

        Regress out industry dummies from each factor.
        """
        dummies = pd.get_dummies(industry, prefix="ind", dtype=float)
        aligned = dummies.reindex(factor_df.index).fillna(0)

        result = factor_df.copy()
        for col in factor_df.columns:
            y = factor_df[col].dropna()
            X = aligned.loc[y.index]
            try:
                beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
                residual = y - X.values @ beta
                result[col] = residual
            except Exception:
                pass
        return result

    def full_pipeline(
        self,
        df: pd.DataFrame,
        industry: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Run full preprocessing: winsorize -> standardize -> neutralize."""
        result = df.copy()
        for col in result.columns:
            result[col] = self.winsorize(result[col])
            result[col] = self.standardize(result[col])

        if industry is not None:
            result = self.neutralize(result, industry)

        return result
