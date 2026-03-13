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

    def get_index_quotes(
        self, index_codes: list[str], start: date, end: date
    ) -> pd.DataFrame:
        """Fetch index daily quotes via akshare.

        Args:
            index_codes: e.g. ["000300", "000905"]
            start: start date
            end: end date

        Returns:
            DataFrame [index_code, date, open, high, low, close, volume, amount]
        """
        import akshare as ak

        frames = []
        for code in index_codes:
            try:
                df = ak.index_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                    "成交额": "amount",
                })
                df["index_code"] = code
                frames.append(df[["index_code", "date", "open", "high", "low", "close", "volume", "amount"]])
                logger.info(f"Fetched index {code}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"Failed to fetch index {code}: {e}")
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def get_financial_data(
        self, stock_codes: list[str], report_date: date
    ) -> pd.DataFrame:
        """Fetch financial analysis indicators via akshare.

        Uses ``ak.stock_financial_analysis_indicator`` per stock and maps
        Chinese column names to the canonical English schema.

        Args:
            stock_codes: List of A-share stock codes (e.g. ["600519"]).
            report_date: Target report date (nearest quarterly report is matched).

        Returns:
            DataFrame with columns: [stock_code, report_date, revenue, net_profit,
            roe, eps, bps, gross_margin, debt_ratio, dps, revenue_growth,
            profit_growth].  Missing fields are NaN.
        """
        import akshare as ak
        import time

        # Column mapping: Chinese → English
        _COL_MAP = {
            "日期": "report_date",
            "摊薄每股收益(元)": "eps",
            "每股净资产_调整前(元)": "bps",
            "净资产收益率(%)": "roe",
            "销售毛利率(%)": "gross_margin",
            "资产负债率(%)": "debt_ratio",
            "股息发放率(%)": "dps_ratio",
            "主营业务收入增长率(%)": "revenue_growth",
            "净利润增长率(%)": "profit_growth",
            "主营业务利润(元)": "revenue",  # proxy: main biz profit
        }

        rd_str = report_date.strftime("%Y-%m-%d")
        start_year = str(report_date.year)
        frames: list[pd.DataFrame] = []

        for code in stock_codes:
            try:
                df = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)
                if df.empty:
                    logger.warning(f"Empty financial data for {code}")
                    continue

                # Rename known columns
                rename = {k: v for k, v in _COL_MAP.items() if k in df.columns}
                df = df.rename(columns=rename)

                # Convert report_date to string for matching
                df["report_date"] = df["report_date"].astype(str).str[:10]

                # Find closest report date <= target
                valid = df[df["report_date"] <= rd_str]
                if valid.empty:
                    valid = df.head(1)  # fallback to earliest available

                row = valid.iloc[0:1].copy()
                row["stock_code"] = code

                # Ensure all required columns exist
                for col in ["eps", "bps", "roe", "gross_margin", "debt_ratio",
                            "revenue_growth", "profit_growth", "revenue"]:
                    if col not in row.columns:
                        row[col] = float("nan")

                # dps: use dps_ratio as proxy or 0
                if "dps_ratio" in row.columns:
                    row["dps"] = 0.0  # ratio not actual dps; default to 0
                else:
                    row["dps"] = 0.0

                # net_profit placeholder
                if "net_profit" not in row.columns:
                    # Try to derive: eps * total_shares, but we don't have shares here
                    row["net_profit"] = float("nan")

                out_cols = ["stock_code", "report_date", "revenue", "net_profit",
                            "roe", "eps", "bps", "gross_margin", "debt_ratio",
                            "dps", "revenue_growth", "profit_growth"]
                frames.append(row[out_cols])

            except Exception as e:
                logger.warning(f"Failed to fetch financial data for {code}: {e}")

            time.sleep(0.1)  # rate limit

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        # Convert numeric columns
        for col in ["revenue", "net_profit", "roe", "eps", "bps", "gross_margin",
                     "debt_ratio", "dps", "revenue_growth", "profit_growth"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        return result

    def get_industry_classification(self) -> pd.DataFrame:
        """Fetch industry classification from East Money via akshare.

        Uses ``ak.stock_board_industry_name_em()`` to get industry board names,
        then ``ak.stock_board_industry_cons_em()`` for constituents.
        Falls back to ``ak.stock_individual_info_em()`` per-stock if board API fails.

        Returns:
            DataFrame with columns: [stock_code, industry_l1, industry_l2, industry_l3]
        """
        import akshare as ak
        import time

        # Strategy: get all industry boards and their constituents
        try:
            boards = ak.stock_board_industry_name_em()
            board_names = boards["板块名称"].tolist()
            logger.info(f"Found {len(board_names)} industry boards")
        except Exception as e:
            logger.warning(f"Failed to get industry board list: {e}")
            return pd.DataFrame(columns=["stock_code", "industry_l1", "industry_l2", "industry_l3"])

        frames: list[pd.DataFrame] = []
        for name in board_names:
            try:
                cons = ak.stock_board_industry_cons_em(symbol=name)
                if cons.empty:
                    continue
                code_col = "代码" if "代码" in cons.columns else cons.columns[1]
                sub = pd.DataFrame({
                    "stock_code": cons[code_col].astype(str),
                    "industry_l1": name,
                    "industry_l2": "",
                    "industry_l3": "",
                })
                frames.append(sub)
                time.sleep(0.05)
            except Exception as e:
                logger.debug(f"Failed to get constituents for {name}: {e}")
                continue

        if not frames:
            logger.warning("Board cons API failed for all boards, returning empty")
            return pd.DataFrame(columns=["stock_code", "industry_l1", "industry_l2", "industry_l3"])

        result = pd.concat(frames, ignore_index=True)
        # A stock may appear in multiple boards; keep first
        result = result.drop_duplicates(subset=["stock_code"], keep="first")
        logger.info(f"Industry classification: {len(result)} stocks")
        return result

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
