"""Tests for get_financial_data and get_industry_classification (mock-based, no network)."""

from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from quant2026.data.akshare_provider import AkShareProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider() -> AkShareProvider:
    return AkShareProvider()


def _mock_financial_indicator(symbol: str, start_year: str) -> pd.DataFrame:
    """Return a fake DataFrame mimicking ak.stock_financial_analysis_indicator."""
    return pd.DataFrame({
        "日期": ["2024-06-30", "2024-03-31"],
        "摊薄每股收益(元)": [34.37, 19.81],
        "每股净资产_调整前(元)": [181.54, 197.85],
        "净资产收益率(%)": [19.08, 10.04],
        "销售毛利率(%)": [75.29, 76.89],
        "资产负债率(%)": [18.32, 12.95],
        "主营业务收入增长率(%)": [17.76, 18.11],
        "净利润增长率(%)": [15.66, 15.60],
        "主营业务利润(元)": [6.17e10, 3.52e10],
    })


def _mock_board_names() -> pd.DataFrame:
    return pd.DataFrame({
        "排名": [1, 2],
        "板块名称": ["白酒", "银行"],
        "板块代码": ["BK001", "BK002"],
    })


def _mock_board_cons(symbol: str) -> pd.DataFrame:
    mapping = {
        "白酒": pd.DataFrame({"代码": ["600519", "000568"], "名称": ["贵州茅台", "泸州老窖"]}),
        "银行": pd.DataFrame({"代码": ["600036", "000001"], "名称": ["招商银行", "平安银行"]}),
    }
    if symbol in mapping:
        return mapping[symbol]
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Tests: get_financial_data
# ---------------------------------------------------------------------------

class TestGetFinancialData:
    @patch("akshare.stock_financial_analysis_indicator", side_effect=_mock_financial_indicator)
    def test_returns_correct_columns(self, mock_ak: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_financial_data(["600519"], date(2024, 6, 30))
        required = {"stock_code", "report_date", "revenue", "net_profit",
                     "roe", "eps", "bps", "gross_margin", "debt_ratio",
                     "dps", "revenue_growth", "profit_growth"}
        assert required.issubset(set(df.columns))

    @patch("akshare.stock_financial_analysis_indicator", side_effect=_mock_financial_indicator)
    def test_values_are_numeric(self, mock_ak: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_financial_data(["600519"], date(2024, 6, 30))
        for col in ["eps", "bps", "roe", "gross_margin", "debt_ratio"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

    @patch("akshare.stock_financial_analysis_indicator", side_effect=_mock_financial_indicator)
    def test_multiple_stocks(self, mock_ak: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_financial_data(["600519", "000858"], date(2024, 6, 30))
        assert len(df) == 2
        assert set(df["stock_code"]) == {"600519", "000858"}

    @patch("akshare.stock_financial_analysis_indicator", side_effect=Exception("network error"))
    def test_failed_stock_skipped(self, mock_ak: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_financial_data(["600519"], date(2024, 6, 30))
        assert df.empty

    @patch("akshare.stock_financial_analysis_indicator", return_value=pd.DataFrame())
    def test_empty_response(self, mock_ak: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_financial_data(["600519"], date(2024, 6, 30))
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: get_industry_classification
# ---------------------------------------------------------------------------

class TestGetIndustryClassification:
    @patch("akshare.stock_board_industry_cons_em", side_effect=_mock_board_cons)
    @patch("akshare.stock_board_industry_name_em", return_value=_mock_board_names())
    def test_returns_correct_columns(self, mock_names: MagicMock, mock_cons: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_industry_classification()
        assert {"stock_code", "industry_l1", "industry_l2", "industry_l3"}.issubset(set(df.columns))

    @patch("akshare.stock_board_industry_cons_em", side_effect=_mock_board_cons)
    @patch("akshare.stock_board_industry_name_em", return_value=_mock_board_names())
    def test_has_expected_stocks(self, mock_names: MagicMock, mock_cons: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_industry_classification()
        assert "600519" in df["stock_code"].values
        assert "600036" in df["stock_code"].values

    @patch("akshare.stock_board_industry_cons_em", side_effect=_mock_board_cons)
    @patch("akshare.stock_board_industry_name_em", return_value=_mock_board_names())
    def test_no_duplicates(self, mock_names: MagicMock, mock_cons: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_industry_classification()
        assert df["stock_code"].is_unique

    @patch("akshare.stock_board_industry_name_em", side_effect=Exception("fail"))
    def test_board_list_fails_returns_empty(self, mock_names: MagicMock, provider: AkShareProvider) -> None:
        df = provider.get_industry_classification()
        assert df.empty
        assert list(df.columns) == ["stock_code", "industry_l1", "industry_l2", "industry_l3"]
