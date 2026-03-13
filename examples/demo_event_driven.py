#!/usr/bin/env python3
"""事件驱动策略 Demo。

尝试从 akshare 拉取真实事件数据；若接口不可用则用模拟数据演示策略逻辑。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from loguru import logger

from quant2026.strategy.event_driven.events import EventCollector
from quant2026.strategy.event_driven.strategy import EventDrivenStrategy


def _make_mock_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """构造模拟事件数据。"""
    today = date.today()

    earnings = pd.DataFrame({
        "stock_code": ["000001", "000002", "600036", "601318"],
        "date": [today - timedelta(days=d) for d in [5, 10, 3, 7]],
        "event_type": ["earnings_surprise"] * 4,
        "surprise_pct": [25.0, -10.0, 40.0, 5.0],
        "description": ["净利润大增", "营收下滑", "业绩翻倍", "稳健增长"],
    })

    block_trades = pd.DataFrame({
        "stock_code": ["000001", "600036", "601318"],
        "date": [today - timedelta(days=d) for d in [2, 8, 1]],
        "volume": [5000000, 3000000, 8000000],
        "premium_pct": [3.5, -1.2, 6.0],
        "buyer": ["机构A", "个人", "机构B"],
    })

    shareholder = pd.DataFrame({
        "stock_code": ["000002", "600036", "601318"],
        "date": [today - timedelta(days=d) for d in [4, 6, 9]],
        "holder_name": ["大股东A", "管理层B", "战略投资者C"],
        "change_type": ["decrease", "increase", "increase"],
        "change_pct": [2.0, 5.0, 3.0],
    })

    return earnings, block_trades, shareholder


class MockCollector(EventCollector):
    """用模拟数据的 EventCollector。"""

    def __init__(self, earnings, block_trades, shareholder):
        self._e = earnings
        self._b = block_trades
        self._s = shareholder

    def get_earnings_surprise(self, stock_codes, start, end):
        return self._e[self._e["stock_code"].isin(stock_codes)]

    def get_block_trades(self, start, end):
        return self._b

    def get_shareholder_changes(self, stock_codes, start, end):
        return self._s[self._s["stock_code"].isin(stock_codes)]


def try_real_data() -> bool:
    """尝试用真实 akshare 数据跑策略。"""
    print("=" * 60)
    print("📡 尝试从 akshare 获取真实事件数据...")
    print("=" * 60)

    target = date.today()
    stocks = ["000001", "000002", "600036", "601318", "600519"]
    data = pd.DataFrame({"stock_code": stocks})

    collector = EventCollector()
    start = target - timedelta(days=90)

    # 测试各接口
    earnings = collector.get_earnings_surprise(stocks, start, target)
    block_trades = collector.get_block_trades(start, target)
    shareholder = collector.get_shareholder_changes(stocks, start, target)

    total = len(earnings) + len(block_trades) + len(shareholder)
    print(f"\n📊 数据统计:")
    print(f"  财报事件: {len(earnings)} 条")
    print(f"  大宗交易: {len(block_trades)} 条")
    print(f"  股东变动: {len(shareholder)} 条")

    if total == 0:
        print("  ⚠️  未获取到真实数据，将使用模拟数据")
        return False

    # 跑策略
    strategy = EventDrivenStrategy(event_collector=collector)
    result = strategy.generate(data, None, target)

    print(f"\n🎯 策略结果 (target={target}):")
    scores = result.scores.sort_values(ascending=False)
    for code, score in scores.items():
        bar = "█" * int(abs(score))
        sign = "+" if score >= 0 else ""
        print(f"  {code}: {sign}{score:.2f}  {bar}")

    return True


def run_mock_demo():
    """用模拟数据演示策略。"""
    print("\n" + "=" * 60)
    print("🎭 使用模拟事件数据演示策略逻辑")
    print("=" * 60)

    earnings, block_trades, shareholder = _make_mock_data()
    target = date.today()
    stocks = ["000001", "000002", "600036", "601318"]
    data = pd.DataFrame({"stock_code": stocks})

    print("\n📋 模拟事件数据:")
    print("\n  [财报超预期]")
    for _, r in earnings.iterrows():
        sign = "+" if r["surprise_pct"] > 0 else ""
        print(f"    {r['stock_code']} | {sign}{r['surprise_pct']:.0f}% | {r['description']}")

    print("\n  [大宗交易]")
    for _, r in block_trades.iterrows():
        sign = "+" if r["premium_pct"] > 0 else ""
        print(f"    {r['stock_code']} | 溢价{sign}{r['premium_pct']:.1f}% | {r['buyer']}")

    print("\n  [股东增减持]")
    for _, r in shareholder.iterrows():
        direction = "📈增持" if r["change_type"] == "increase" else "📉减持"
        print(f"    {r['stock_code']} | {direction} {r['change_pct']:.1f}% | {r['holder_name']}")

    # 跑策略
    collector = MockCollector(earnings, block_trades, shareholder)
    strategy = EventDrivenStrategy(
        earnings_weight=0.5,
        block_trade_weight=0.3,
        shareholder_weight=0.2,
        lookback_days=30,
        event_collector=collector,
    )
    result = strategy.generate(data, None, target)

    print(f"\n🎯 策略评分结果 (权重: 财报0.5 / 大宗0.3 / 股东0.2)")
    print("-" * 50)
    scores = result.scores.sort_values(ascending=False)
    max_abs = max(abs(scores.max()), abs(scores.min()), 1)
    for code, score in scores.items():
        bar_len = int(abs(score) / max_abs * 20)
        bar = "█" * bar_len
        sign = "+" if score >= 0 else ""
        emoji = "🟢" if score > 0 else ("🔴" if score < 0 else "⚪")
        print(f"  {emoji} {code}: {sign}{score:>8.2f}  {bar}")

    print(f"\n📈 事件统计: {result.metadata}")
    print("\n✅ Demo 完成！")


def main():
    """主入口。"""
    logger.remove()
    logger.add(lambda msg: None)  # 静默 loguru 输出

    real_ok = try_real_data()
    run_mock_demo()


if __name__ == "__main__":
    main()
