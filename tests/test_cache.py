"""Tests for CachedProvider."""

import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from quant2026.data.base import DataProvider
from quant2026.data.cache import CachedProvider


# ── Mock provider ───────────────────────────────────────────


class MockProvider(DataProvider):
    def __init__(self):
        self.call_log: list[tuple[str, tuple]] = []

    def get_daily_quotes(self, stock_codes, start, end):
        self.call_log.append(("daily_quotes", (stock_codes, start, end)))
        rows = []
        for code in stock_codes:
            d = start
            while d <= end:
                rows.append({
                    "stock_code": code, "date": d.isoformat(),
                    "open": 10, "high": 11, "low": 9, "close": 10.5,
                    "volume": 1000, "amount": 10500,
                })
                d += timedelta(days=1)
        return pd.DataFrame(rows)

    def get_financial_data(self, stock_codes, report_date):
        self.call_log.append(("financial", (stock_codes, report_date)))
        rows = [{"stock_code": c, "report_date": report_date.isoformat(),
                 "revenue": 100, "net_profit": 10, "roe": 0.1} for c in stock_codes]
        return pd.DataFrame(rows)

    def get_stock_list(self, market_date=None):
        self.call_log.append(("stock_list", (market_date,)))
        return pd.DataFrame([
            {"stock_code": "000001", "name": "平安银行", "market": "sz"},
            {"stock_code": "600000", "name": "浦发银行", "market": "sh"},
        ])

    def get_index_quotes(self, index_codes, start, end):
        self.call_log.append(("index_quotes", (index_codes, start, end)))
        rows = [{"index_code": c, "date": start.isoformat(), "close": 3000} for c in index_codes]
        return pd.DataFrame(rows)

    def get_industry_classification(self):
        self.call_log.append(("industry", ()))
        return pd.DataFrame([
            {"stock_code": "000001", "industry_l1": "银行", "industry_l2": "银行", "industry_l3": "银行"},
        ])


@pytest.fixture
def setup():
    with tempfile.TemporaryDirectory() as tmp:
        mock = MockProvider()
        cached = CachedProvider(mock, cache_dir=tmp)
        yield mock, cached


# ── Tests ───────────────────────────────────────────────────


class TestDailyQuotes:
    def test_cache_miss_then_hit(self, setup):
        mock, cached = setup
        s, e = date(2024, 1, 1), date(2024, 1, 5)

        df1 = cached.get_daily_quotes(["000001"], s, e)
        assert len(mock.call_log) == 1
        assert len(df1) == 5

        # Second call should not call upstream
        mock.call_log.clear()
        df2 = cached.get_daily_quotes(["000001"], s, e)
        assert len(mock.call_log) == 0
        assert len(df2) == 5

    def test_incremental_update(self, setup):
        mock, cached = setup
        # First: cache Jan 3-5
        cached.get_daily_quotes(["000001"], date(2024, 1, 3), date(2024, 1, 5))
        assert len(mock.call_log) == 1
        mock.call_log.clear()

        # Extend left to Jan 1 and right to Jan 7
        df = cached.get_daily_quotes(["000001"], date(2024, 1, 1), date(2024, 1, 7))
        # Should have 2 upstream calls (left gap + right gap)
        assert len(mock.call_log) == 2
        assert len(df) == 7


class TestStockList:
    def test_cache_hit(self, setup):
        mock, cached = setup
        d = date(2024, 6, 1)
        cached.get_stock_list(d)
        mock.call_log.clear()
        df = cached.get_stock_list(d)
        assert len(mock.call_log) == 0
        assert len(df) == 2


class TestFinancialData:
    def test_partial_cache(self, setup):
        mock, cached = setup
        rd = date(2024, 3, 31)
        # Cache one stock
        cached.get_financial_data(["000001"], rd)
        mock.call_log.clear()

        # Request two stocks - only 600000 should miss
        df = cached.get_financial_data(["000001", "600000"], rd)
        assert len(mock.call_log) == 1
        assert mock.call_log[0][1][0] == ["600000"]
        assert len(df) == 2


class TestIndustry:
    def test_ttl_hit(self, setup):
        mock, cached = setup
        cached.get_industry_classification()
        mock.call_log.clear()
        cached.get_industry_classification()
        assert len(mock.call_log) == 0


class TestInvalidate:
    def test_invalidate_single(self, setup):
        mock, cached = setup
        cached.get_daily_quotes(["000001"], date(2024, 1, 1), date(2024, 1, 3))
        mock.call_log.clear()

        cached.invalidate("000001")
        cached.get_daily_quotes(["000001"], date(2024, 1, 1), date(2024, 1, 3))
        assert len(mock.call_log) == 1  # had to re-fetch

    def test_invalidate_all(self, setup):
        mock, cached = setup
        cached.get_stock_list(date(2024, 1, 1))
        mock.call_log.clear()

        cached.invalidate()
        cached.get_stock_list(date(2024, 1, 1))
        assert len(mock.call_log) == 1
