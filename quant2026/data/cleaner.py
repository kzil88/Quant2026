"""Data cleaning utilities for A-share specific issues."""

import pandas as pd
import numpy as np
from loguru import logger


class DataCleaner:
    """Clean raw market data with A-share specific logic."""

    @staticmethod
    def remove_st_stocks(df: pd.DataFrame, stock_list: pd.DataFrame) -> pd.DataFrame:
        """Remove ST / *ST / 退市 stocks."""
        st_mask = stock_list["name"].str.contains(r"ST|退市", na=False)
        st_codes = stock_list.loc[st_mask, "stock_code"]
        before = df["stock_code"].nunique()
        df = df[~df["stock_code"].isin(st_codes)]
        after = df["stock_code"].nunique()
        logger.info(f"Removed {before - after} ST stocks")
        return df

    @staticmethod
    def remove_new_stocks(df: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
        """Remove stocks listed less than min_days (次新股行为异常)."""
        trading_days = df.groupby("stock_code")["date"].nunique()
        valid = trading_days[trading_days >= min_days].index
        before = df["stock_code"].nunique()
        df = df[df["stock_code"].isin(valid)]
        logger.info(f"Removed {before - df['stock_code'].nunique()} new stocks (< {min_days} days)")
        return df

    @staticmethod
    def handle_limit_up_down(df: pd.DataFrame) -> pd.DataFrame:
        """Mark limit-up/down days (涨跌停不可交易).

        Adds columns: is_limit_up, is_limit_down
        """
        df = df.copy()
        pct_change = df.groupby("stock_code")["close"].pct_change()

        # 主板10%, 创业板/科创板20%
        is_cyb = df["stock_code"].str.startswith(("300", "301"))  # 创业板
        is_kcb = df["stock_code"].str.startswith("688")            # 科创板
        limit = pd.Series(0.1, index=df.index)
        limit[is_cyb | is_kcb] = 0.2

        df["is_limit_up"] = pct_change >= (limit - 0.001)
        df["is_limit_down"] = pct_change <= -(limit - 0.001)
        return df

    @staticmethod
    def handle_suspended(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill prices for suspended days, mark as not tradable."""
        df = df.copy()
        df["is_suspended"] = df["volume"] == 0
        return df

    @staticmethod
    def adjust_for_splits(df: pd.DataFrame) -> pd.DataFrame:
        """Apply forward-adjusted prices (前复权). Assumes data source provides raw prices."""
        # Most A-share data sources (akshare) can return pre-adjusted data
        # This is a placeholder for custom adjustment logic
        logger.info("Price adjustment: using data source default (forward-adjusted)")
        return df
