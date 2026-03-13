"""Base data provider interface."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Abstract data provider. All data sources implement this."""

    @abstractmethod
    def get_daily_quotes(
        self, stock_codes: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Fetch daily OHLCV data.

        Returns:
            DataFrame with columns: [stock_code, date, open, high, low, close, volume, amount]
        """
        ...

    @abstractmethod
    def get_financial_data(
        self, stock_codes: list[str], report_date: date
    ) -> pd.DataFrame:
        """Fetch financial statement data (income, balance sheet, cash flow).

        Returns:
            DataFrame with columns: [stock_code, report_date, revenue, net_profit, roe, ...]
        """
        ...

    @abstractmethod
    def get_stock_list(self, market_date: date | None = None) -> pd.DataFrame:
        """Fetch tradable stock list, excluding ST/suspended.

        Returns:
            DataFrame with columns: [stock_code, name, market, industry, list_date]
        """
        ...

    def get_index_quotes(
        self, index_codes: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Fetch index daily quotes (default: not implemented)."""
        raise NotImplementedError

    def get_industry_classification(self) -> pd.DataFrame:
        """Fetch industry classification (申万/中信).

        Returns:
            DataFrame with columns: [stock_code, industry_l1, industry_l2, industry_l3]
        """
        raise NotImplementedError
