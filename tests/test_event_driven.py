"""Tests for EventDrivenStrategy."""

from datetime import date

import pandas as pd
import pytest

from quant2026.strategy.event_driven.events import EventCollector
from quant2026.strategy.event_driven.strategy import EventDrivenStrategy


# ── Mock EventCollector ────────────────────────────────────────


class MockEventCollector(EventCollector):
    """可控的事件数据源，用于测试。"""

    def __init__(
        self,
        earnings: pd.DataFrame | None = None,
        block_trades: pd.DataFrame | None = None,
        shareholder: pd.DataFrame | None = None,
    ):
        self._earnings = earnings if earnings is not None else pd.DataFrame(
            columns=["stock_code", "date", "event_type", "surprise_pct", "description"]
        )
        self._block_trades = block_trades if block_trades is not None else pd.DataFrame(
            columns=["stock_code", "date", "volume", "premium_pct", "buyer"]
        )
        self._shareholder = shareholder if shareholder is not None else pd.DataFrame(
            columns=["stock_code", "date", "holder_name", "change_type", "change_pct"]
        )

    def get_earnings_surprise(self, stock_codes, start, end):
        return self._earnings

    def get_block_trades(self, start, end):
        return self._block_trades

    def get_shareholder_changes(self, stock_codes, start, end):
        return self._shareholder


# ── Fixtures ───────────────────────────────────────────────────


@pytest.fixture
def market_data() -> pd.DataFrame:
    return pd.DataFrame({"stock_code": ["000001", "000002", "000003", "600000"]})


@pytest.fixture
def target() -> date:
    return date(2025, 3, 10)


# ── Tests ──────────────────────────────────────────────────────


def test_no_events_all_zero(market_data, target):
    """无事件时所有股票得分为 0。"""
    strategy = EventDrivenStrategy(event_collector=MockEventCollector())
    result = strategy.generate(market_data, None, target)

    assert result.name == "event_driven"
    assert result.date == target
    assert (result.scores == 0).all()
    assert len(result.scores) == 4


def test_earnings_surprise_positive(market_data, target):
    """超预期 → 正分。"""
    earnings = pd.DataFrame({
        "stock_code": ["000001"],
        "date": [date(2025, 3, 5)],
        "event_type": ["earnings_surprise"],
        "surprise_pct": [20.0],
        "description": ["净利润大幅增长"],
    })
    strategy = EventDrivenStrategy(
        earnings_weight=0.5,
        event_collector=MockEventCollector(earnings=earnings),
    )
    result = strategy.generate(market_data, None, target)

    assert result.scores["000001"] == pytest.approx(20.0 * 0.5)
    assert result.scores["000002"] == 0.0


def test_earnings_surprise_negative(market_data, target):
    """低于预期（负 surprise_pct） → 负分。"""
    earnings = pd.DataFrame({
        "stock_code": ["000002"],
        "date": [date(2025, 3, 5)],
        "event_type": ["earnings_surprise"],
        "surprise_pct": [-15.0],
        "description": ["业绩下滑"],
    })
    strategy = EventDrivenStrategy(
        earnings_weight=0.5,
        event_collector=MockEventCollector(earnings=earnings),
    )
    result = strategy.generate(market_data, None, target)

    assert result.scores["000002"] < 0
    assert result.scores["000002"] == pytest.approx(-15.0 * 0.5)


def test_block_trade_premium(market_data, target):
    """大宗交易溢价 → 正分。"""
    bt = pd.DataFrame({
        "stock_code": ["000001"],
        "date": [date(2025, 3, 8)],
        "volume": [1000000],
        "premium_pct": [5.0],
        "buyer": ["机构A"],
    })
    strategy = EventDrivenStrategy(
        block_trade_weight=0.3,
        event_collector=MockEventCollector(block_trades=bt),
    )
    result = strategy.generate(market_data, None, target)

    assert result.scores["000001"] == pytest.approx(5.0 * 0.3)


def test_shareholder_increase(market_data, target):
    """股东增持 → 正分。"""
    sh = pd.DataFrame({
        "stock_code": ["000003"],
        "date": [date(2025, 3, 7)],
        "holder_name": ["张三"],
        "change_type": ["increase"],
        "change_pct": [2.5],
    })
    strategy = EventDrivenStrategy(
        shareholder_weight=0.2,
        event_collector=MockEventCollector(shareholder=sh),
    )
    result = strategy.generate(market_data, None, target)

    assert result.scores["000003"] == pytest.approx(2.5 * 0.2)


def test_shareholder_decrease(market_data, target):
    """股东减持 → 负分。"""
    sh = pd.DataFrame({
        "stock_code": ["000003"],
        "date": [date(2025, 3, 7)],
        "holder_name": ["李四"],
        "change_type": ["decrease"],
        "change_pct": [3.0],
    })
    strategy = EventDrivenStrategy(
        shareholder_weight=0.2,
        event_collector=MockEventCollector(shareholder=sh),
    )
    result = strategy.generate(market_data, None, target)

    assert result.scores["000003"] == pytest.approx(-3.0 * 0.2)


def test_multiple_events_stack(market_data, target):
    """多事件叠加。"""
    earnings = pd.DataFrame({
        "stock_code": ["000001"],
        "date": [date(2025, 3, 5)],
        "event_type": ["earnings_surprise"],
        "surprise_pct": [10.0],
        "description": [""],
    })
    bt = pd.DataFrame({
        "stock_code": ["000001"],
        "date": [date(2025, 3, 8)],
        "volume": [500000],
        "premium_pct": [3.0],
        "buyer": [""],
    })
    sh = pd.DataFrame({
        "stock_code": ["000001"],
        "date": [date(2025, 3, 9)],
        "holder_name": ["王五"],
        "change_type": ["increase"],
        "change_pct": [1.0],
    })

    strategy = EventDrivenStrategy(
        earnings_weight=0.5,
        block_trade_weight=0.3,
        shareholder_weight=0.2,
        event_collector=MockEventCollector(
            earnings=earnings, block_trades=bt, shareholder=sh,
        ),
    )
    result = strategy.generate(market_data, None, target)

    expected = 10.0 * 0.5 + 3.0 * 0.3 + 1.0 * 0.2
    assert result.scores["000001"] == pytest.approx(expected)
    # 其他股票仍为 0
    assert result.scores["000002"] == 0.0


def test_metadata_event_counts(market_data, target):
    """metadata 包含事件计数。"""
    earnings = pd.DataFrame({
        "stock_code": ["000001", "000002"],
        "date": [date(2025, 3, 5)] * 2,
        "event_type": ["earnings_surprise"] * 2,
        "surprise_pct": [10.0, -5.0],
        "description": ["", ""],
    })
    strategy = EventDrivenStrategy(
        event_collector=MockEventCollector(earnings=earnings),
    )
    result = strategy.generate(market_data, None, target)

    assert result.metadata["earnings_events"] == 2
    assert result.metadata["block_trade_events"] == 0
    assert result.metadata["shareholder_events"] == 0


def test_shareholder_chinese_decrease(market_data, target):
    """中文 '减持' 也能被识别。"""
    sh = pd.DataFrame({
        "stock_code": ["600000"],
        "date": [date(2025, 3, 7)],
        "holder_name": ["赵六"],
        "change_type": ["减持"],
        "change_pct": [4.0],
    })
    strategy = EventDrivenStrategy(
        shareholder_weight=0.2,
        event_collector=MockEventCollector(shareholder=sh),
    )
    result = strategy.generate(market_data, None, target)
    assert result.scores["600000"] == pytest.approx(-4.0 * 0.2)
