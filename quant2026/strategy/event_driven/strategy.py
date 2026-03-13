"""事件驱动策略：基于财报、大宗交易、股东增减持等事件产生交易信号。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from quant2026.strategy.base import Strategy
from quant2026.strategy.event_driven.events import EventCollector
from quant2026.types import StrategyResult


class EventDrivenStrategy(Strategy):
    """基于公司事件的多因子策略。

    事件类型及权重:
    - 财报超预期 (earnings_weight)
    - 大宗交易溢价 (block_trade_weight)
    - 股东增减持 (shareholder_weight)
    """

    def __init__(
        self,
        earnings_weight: float = 0.5,
        block_trade_weight: float = 0.3,
        shareholder_weight: float = 0.2,
        lookback_days: int = 30,
        event_collector: EventCollector | None = None,
    ) -> None:
        self.earnings_weight = earnings_weight
        self.block_trade_weight = block_trade_weight
        self.shareholder_weight = shareholder_weight
        self.lookback_days = lookback_days
        self.collector = event_collector or EventCollector()

    @property
    def name(self) -> str:
        return "event_driven"

    def generate(
        self,
        data: pd.DataFrame,
        factor_matrix: pd.DataFrame | None,
        target_date: date,
    ) -> StrategyResult:
        """根据事件数据生成股票评分。

        流程:
        1. 收集 lookback 窗口内的所有事件
        2. 按股票汇总事件得分
        3. 返回 StrategyResult
        """
        start = target_date - timedelta(days=self.lookback_days)
        end = target_date

        # 获取 universe
        stock_codes = _extract_stock_codes(data)
        logger.info(
            "EventDriven: generating for {} stocks, window {}-{}",
            len(stock_codes), start, end,
        )

        # ── 收集事件 ──
        earnings = self.collector.get_earnings_surprise(stock_codes, start, end)
        block_trades = self.collector.get_block_trades(start, end)
        shareholder = self.collector.get_shareholder_changes(stock_codes, start, end)

        # ── 计算得分 ──
        scores: dict[str, float] = {}
        metadata: dict[str, int] = {
            "earnings_events": len(earnings),
            "block_trade_events": len(block_trades),
            "shareholder_events": len(shareholder),
        }

        # 财报超预期
        if not earnings.empty:
            for code, group in earnings.groupby("stock_code"):
                s = group["surprise_pct"].sum() * self.earnings_weight
                scores[code] = scores.get(code, 0.0) + s

        # 大宗交易
        if not block_trades.empty:
            # 过滤 universe
            bt = block_trades[block_trades["stock_code"].isin(set(stock_codes))]
            for code, group in bt.groupby("stock_code"):
                s = group["premium_pct"].sum() * self.block_trade_weight
                scores[code] = scores.get(code, 0.0) + s

        # 股东增减持
        if not shareholder.empty:
            for code, group in shareholder.groupby("stock_code"):
                for _, row in group.iterrows():
                    pct = float(row.get("change_pct", 0))
                    ct = str(row.get("change_type", "")).lower()
                    if "decrease" in ct or "减" in ct:
                        scores[code] = scores.get(code, 0.0) - abs(pct) * self.shareholder_weight
                    else:
                        scores[code] = scores.get(code, 0.0) + abs(pct) * self.shareholder_weight

        # 未出现事件的股票得分为 0
        score_series = pd.Series(
            {code: scores.get(code, 0.0) for code in stock_codes},
            dtype=float,
        )

        logger.info(
            "EventDriven: {} stocks with events, score range [{:.4f}, {:.4f}]",
            sum(1 for v in score_series if v != 0),
            score_series.min() if len(score_series) else 0,
            score_series.max() if len(score_series) else 0,
        )

        return StrategyResult(
            name=self.name,
            date=target_date,
            scores=score_series,
            metadata=metadata,
        )


def _extract_stock_codes(data: pd.DataFrame) -> list[str]:
    """从 data 中提取股票代码列表。"""
    if "stock_code" in data.columns:
        return data["stock_code"].unique().tolist()
    if "code" in data.columns:
        return data["code"].unique().tolist()
    # 尝试 index
    if data.index.name in ("stock_code", "code"):
        return data.index.unique().tolist()
    # fallback: 用 columns 里第一个看起来像代码的
    for col in data.columns:
        sample = data[col].dropna().head(5)
        if sample.dtype == object and all(
            isinstance(v, str) and len(v) == 6 and v.isdigit() for v in sample
        ):
            return data[col].unique().tolist()
    return []
