"""AkShare-based data provider for China A-shares."""

from datetime import date

import pandas as pd
from loguru import logger

from .base import DataProvider


class AkShareProvider(DataProvider):
    """Data provider using akshare (free, no API key needed)."""

    def get_daily_quotes(
        self, stock_codes: list[str], start: date, end: date
    ) -> pd.DataFrame:
        import akshare as ak

        frames = []
        for code in stock_codes:
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="qfq",  # 前复权
                )
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                    "成交额": "amount",
                })
                df["stock_code"] = code
                frames.append(df[["stock_code", "date", "open", "high", "low", "close", "volume", "amount"]])
            except Exception as e:
                logger.warning(f"Failed to fetch {code}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_financial_data(
        self, stock_codes: list[str], report_date: date
    ) -> pd.DataFrame:
        # TODO: implement via ak.stock_financial_analysis_indicator
        raise NotImplementedError("Financial data provider not yet implemented")

    def get_stock_list(self, market_date: date | None = None) -> pd.DataFrame:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={"代码": "stock_code", "名称": "name"})

        def _market(code: str) -> str:
            if code.startswith(("600", "601", "603", "688")):
                return "sh"
            return "sz"

        df["market"] = df["stock_code"].apply(_market)
        return df[["stock_code", "name", "market"]]
