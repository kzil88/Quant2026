"""A股事件数据采集模块。"""

from __future__ import annotations

from datetime import date

import pandas as pd
from loguru import logger


class EventCollector:
    """A股事件数据采集，基于 akshare。

    所有方法在接口失败时返回空 DataFrame，保证策略不会因数据源问题中断。
    """

    # ── 财报超预期 ──────────────────────────────────────────────

    def get_earnings_surprise(
        self,
        stock_codes: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """获取业绩预告中的超预期/低于预期事件。

        Uses ``ak.stock_yjyg_em()`` (业绩预告).

        Returns:
            DataFrame[stock_code, date, event_type, surprise_pct, description]
        """
        empty = pd.DataFrame(
            columns=["stock_code", "date", "event_type", "surprise_pct", "description"]
        )
        try:
            import akshare as ak

            # 按季度尝试拉取（yjyg 按报告期查询，格式 YYYYMMDD，取季度末）
            quarters = _quarter_dates_between(start, end)
            frames: list[pd.DataFrame] = []
            for q in quarters:
                try:
                    raw = ak.stock_yjyg_em(date=q)
                    frames.append(raw)
                except Exception:
                    continue

            if not frames:
                logger.debug("No earnings forecast data fetched")
                return empty

            raw = pd.concat(frames, ignore_index=True)

            # 过滤股票代码
            code_set = set(stock_codes)
            raw = raw[raw["股票代码"].isin(code_set)].copy()
            if raw.empty:
                return empty

            # 解析公告日期
            raw["date"] = pd.to_datetime(raw["公告日期"], errors="coerce").dt.date
            raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]

            # 业绩变动幅度 → surprise_pct（可能为 NaN）
            raw["surprise_pct"] = pd.to_numeric(raw["业绩变动幅度"], errors="coerce").fillna(0.0)

            result = pd.DataFrame(
                {
                    "stock_code": raw["股票代码"].values,
                    "date": raw["date"].values,
                    "event_type": "earnings_surprise",
                    "surprise_pct": raw["surprise_pct"].values,
                    "description": raw["业绩变动"].values,
                }
            )
            logger.info("Fetched {} earnings surprise events", len(result))
            return result

        except Exception as exc:
            logger.warning("get_earnings_surprise failed: {}", exc)
            return empty

    # ── 大宗交易 ────────────────────────────────────────────────

    def get_block_trades(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """获取大宗交易统计数据。

        Uses ``ak.stock_dzjy_sctj()`` (大宗交易市场统计).

        Returns:
            DataFrame[stock_code, date, volume, premium_pct, buyer]
        """
        empty = pd.DataFrame(
            columns=["stock_code", "date", "volume", "premium_pct", "buyer"]
        )
        try:
            import akshare as ak

            raw = ak.stock_dzjy_sctj()
            if raw is None or raw.empty:
                return empty

            # 列名可能变化，尝试映射
            col_map = _guess_block_trade_columns(raw.columns.tolist())
            if col_map is None:
                logger.warning("Cannot map block trade columns: {}", list(raw.columns))
                return empty

            raw = raw.rename(columns=col_map)

            # 日期过滤
            if "date" in raw.columns:
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date
                raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]

            for c in ["volume", "premium_pct"]:
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)

            # 确保输出列存在
            for c in empty.columns:
                if c not in raw.columns:
                    raw[c] = "" if c == "buyer" else 0.0

            result = raw[list(empty.columns)].copy()
            logger.info("Fetched {} block trade events", len(result))
            return result

        except Exception as exc:
            logger.warning("get_block_trades failed: {}", exc)
            return empty

    # ── 股东增减持 ──────────────────────────────────────────────

    def get_shareholder_changes(
        self,
        stock_codes: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """获取股东增减持数据。

        Uses ``ak.stock_gdfx_free_holding_change_em()``

        Returns:
            DataFrame[stock_code, date, holder_name, change_type, change_pct]
        """
        empty = pd.DataFrame(
            columns=["stock_code", "date", "holder_name", "change_type", "change_pct"]
        )
        try:
            import akshare as ak

            quarters = _quarter_dates_between(start, end)
            frames: list[pd.DataFrame] = []
            for q in quarters:
                try:
                    raw = ak.stock_gdfx_free_holding_change_em(date=q)
                    if raw is not None and not raw.empty:
                        frames.append(raw)
                except Exception:
                    continue

            if not frames:
                logger.debug("No shareholder change data fetched")
                return empty

            raw = pd.concat(frames, ignore_index=True)

            # 尝试映射列
            col_map = _guess_shareholder_columns(raw.columns.tolist())
            if col_map is None:
                logger.warning("Cannot map shareholder columns: {}", list(raw.columns))
                return empty

            raw = raw.rename(columns=col_map)

            # 过滤
            if stock_codes:
                raw = raw[raw["stock_code"].isin(set(stock_codes))]

            if "date" in raw.columns:
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date
                raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]

            for c in empty.columns:
                if c not in raw.columns:
                    raw[c] = "" if c in ("holder_name", "change_type") else 0.0

            if "change_pct" in raw.columns:
                raw["change_pct"] = pd.to_numeric(raw["change_pct"], errors="coerce").fillna(0.0)

            result = raw[list(empty.columns)].copy()
            logger.info("Fetched {} shareholder change events", len(result))
            return result

        except Exception as exc:
            logger.warning("get_shareholder_changes failed: {}", exc)
            return empty


# ── helpers ─────────────────────────────────────────────────────


def _quarter_dates_between(start: date, end: date) -> list[str]:
    """生成 start-end 范围内的季度末日期字符串 (YYYYMMDD)."""
    quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    result: list[str] = []
    for year in range(start.year, end.year + 1):
        for month, day in quarter_ends:
            d = date(year, month, day)
            # 宽松匹配：只要季度末在 start 之前一年到 end 之间
            if date(start.year - 1, 1, 1) <= d <= end:
                result.append(d.strftime("%Y%m%d"))
    return result


def _guess_block_trade_columns(cols: list[str]) -> dict[str, str] | None:
    """尝试将大宗交易原始列名映射到标准列名。"""
    mapping: dict[str, str] = {}
    for c in cols:
        cl = c.lower()
        if "代码" in c or "code" in cl:
            mapping[c] = "stock_code"
        elif "日期" in c or "date" in cl:
            mapping[c] = "date"
        elif "成交量" in c or "volume" in cl:
            mapping[c] = "volume"
        elif "溢价" in c or "折价" in c or "premium" in cl:
            mapping[c] = "premium_pct"
        elif "买方" in c or "buyer" in cl:
            mapping[c] = "buyer"
    return mapping if "stock_code" in mapping.values() else None


def _guess_shareholder_columns(cols: list[str]) -> dict[str, str] | None:
    """尝试将股东变动原始列名映射到标准列名。"""
    mapping: dict[str, str] = {}
    for c in cols:
        cl = c.lower()
        if "代码" in c or "code" in cl:
            mapping[c] = "stock_code"
        elif "日期" in c or "date" in cl:
            mapping[c] = "date"
        elif "股东" in c and "名" in c:
            mapping[c] = "holder_name"
        elif "增减" in c or "变动方向" in c:
            mapping[c] = "change_type"
        elif "变动比例" in c or "比例" in c or "change" in cl:
            mapping[c] = "change_pct"
    return mapping if "stock_code" in mapping.values() else None
