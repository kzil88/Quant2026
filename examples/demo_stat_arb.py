#!/usr/bin/env python3
"""
Quant2026 Statistical Arbitrage / Pairs Trading Demo
=====================================================
找协整对 → 配对交易 → 回测 → HTML 报告

用法: python3 examples/demo_stat_arb.py
"""

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant2026.data.akshare_provider import AkShareProvider
from quant2026.data.cache import CachedProvider
from quant2026.data.cleaner import DataCleaner
from quant2026.strategy.stat_arb.cointegration import CointegrationAnalyzer
from quant2026.strategy.stat_arb.strategy import StatArbStrategy
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.risk.manager import RiskManager
from quant2026.backtest.engine import BacktestEngine, BacktestConfig
from quant2026.backtest.report import BacktestReporter
from quant2026.types import PortfolioTarget

# ── 配置 ───────────────────────────────────────────────────────
START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 12, 31)
REBALANCE_INTERVAL = 20

STOCK_POOL: list[str] = [
    "600519", "000858", "601318", "600036", "000333",
    "600276", "601166", "000001", "600900", "601888",
    "000651", "600030", "601398", "600809", "002714",
    "600887", "000568", "601012", "600585", "002415",
    "601668", "600031", "000002", "600048", "601899",
    "002304", "600309", "601601", "000725", "002475",
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")


def main() -> None:
    # ── 1. 数据获取 ──────────────────────────────────────────
    logger.info("Step 1: 获取行情数据")
    provider = CachedProvider(AkShareProvider())

    try:
        data = provider.get_daily_quotes(STOCK_POOL, START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"数据获取失败: {e}")
        return

    if data.empty:
        logger.error("无数据"); return

    logger.info(f"获取到 {data['stock_code'].nunique()} 只股票, {len(data)} 条记录")

    # ── 2. 清洗 ──────────────────────────────────────────────
    logger.info("Step 2: 数据清洗")
    cleaner = DataCleaner()
    data = cleaner.remove_new_stocks(data, min_days=30)
    data = cleaner.handle_limit_up_down(data)
    data = cleaner.handle_suspended(data)
    data["date"] = data["date"].astype(str)
    valid_stocks = data["stock_code"].unique()
    logger.info(f"清洗后 {len(valid_stocks)} 只股票")

    # ── 3. 协整配对分析 ──────────────────────────────────────
    logger.info("Step 3: 协整配对分析")
    pivot = data.pivot_table(index="date", columns="stock_code", values="close").sort_index()
    analyzer = CointegrationAnalyzer(significance=0.05)
    all_pairs = analyzer.find_pairs(pivot, min_obs=120)

    print("\n" + "=" * 70)
    print("🔗 协整配对 (p < 0.05)")
    print("=" * 70)
    print(f"{'Stock A':<10} {'Stock B':<10} {'p-value':<10} {'Hedge Ratio':<14} {'Half-Life':<12} {'Corr':<8}")
    print("-" * 70)
    for p in all_pairs[:20]:
        print(f"{p['stock_a']:<10} {p['stock_b']:<10} {p['p_value']:<10.4f} {p['hedge_ratio']:<14.4f} {p['half_life']:<12.1f} {p['correlation']:<8.3f}")
    if len(all_pairs) > 20:
        print(f"  ... and {len(all_pairs) - 20} more pairs")
    print(f"\nTotal: {len(all_pairs)} cointegrated pairs found")
    print("=" * 70)

    # ── 4. 策略回测 ──────────────────────────────────────────
    logger.info("Step 4: 配对交易策略回测")
    strategy = StatArbStrategy(
        lookback=120, entry_zscore=2.0, exit_zscore=0.5,
        max_pairs=10, recalc_interval=60,
    )
    optimizer = PortfolioOptimizer(
        max_stocks=15, max_single_weight=0.15,
        strategy_weights={strategy.name: 1.0},
    )

    all_dates = sorted(data["date"].unique())
    rebalance_indices = list(range(119, len(all_dates), REBALANCE_INTERVAL))
    rebalance_dates_str = [all_dates[i] for i in rebalance_indices]

    targets: dict[date, PortfolioTarget] = {}
    for idx, dt_str in enumerate(rebalance_dates_str):
        target_date = date.fromisoformat(dt_str)
        result = strategy.generate(data, None, target_date)

        if result.scores.empty:
            continue

        target = optimizer.combine([result], target_date)
        targets[target_date] = target

        if (idx + 1) % 3 == 0 or idx == len(rebalance_dates_str) - 1:
            n_pairs = len(result.metadata.get("active_pairs", []))
            logger.info(f"  调仓 {idx+1}/{len(rebalance_dates_str)}: {dt_str}, "
                        f"{n_pairs} active pairs, {len(target.weights)} stocks")

    if not targets:
        logger.error("未生成调仓目标"); return

    # ── 5. 风控 ──────────────────────────────────────────────
    risk_mgr = RiskManager(max_single_weight=0.15, min_stocks=3)
    for dt, tgt in targets.items():
        targets[dt] = risk_mgr.check_pre_trade(tgt)

    # ── 6. 回测执行 ──────────────────────────────────────────
    logger.info("Step 5: 回测")
    config = BacktestConfig(
        start_date=START_DATE, end_date=END_DATE,
        initial_capital=1_000_000,
        commission_rate=0.0003, stamp_tax_rate=0.0005, slippage_pct=0.001,
    )
    engine = BacktestEngine(config)
    bt_data = data.copy()
    bt_data["date"] = pd.to_datetime(bt_data["date"]).dt.date
    bt_result = engine.run(bt_data, targets)

    if bt_result.equity_curve.empty:
        logger.error("回测无结果"); return

    # ── 7. 报告 ──────────────────────────────────────────────
    logger.info("Step 6: 生成报告")
    benchmark_df = pd.DataFrame()
    try:
        benchmark_df = provider.get_index_quotes(["000300"], START_DATE, END_DATE)
        if not benchmark_df.empty:
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
    except Exception as e:
        logger.warning(f"基准获取失败: {e}")

    try:
        reporter = BacktestReporter(bt_result, benchmark_df, benchmark_name="CSI 300")
        report_path = reporter.generate_html(OUTPUT_DIR / "stat_arb_report.html")
        logger.info(f"HTML report: {report_path}")
    except Exception as e:
        logger.warning(f"报告生成失败: {e}")

    # ── 8. 打印绩效 ──────────────────────────────────────────
    metrics = bt_result.metrics
    print("\n" + "=" * 50)
    print("📊 统计套利回测绩效")
    print("=" * 50)
    print(f"  回测区间:   {START_DATE} ~ {END_DATE}")
    print(f"  股票池:     {len(STOCK_POOL)} 只")
    print(f"  调仓次数:   {len(targets)}")
    print(f"  初始资金:   ¥1,000,000")
    print("-" * 50)
    labels = {
        "total_return": "总收益率", "annual_return": "年化收益率",
        "volatility": "年化波动率", "sharpe_ratio": "夏普比率",
        "max_drawdown": "最大回撤", "trade_count": "交易次数",
        "avg_turnover": "平均换手率",
    }
    for k, v in metrics.items():
        print(f"  {labels.get(k, k):<12} {v}")
    print("-" * 50)
    print(f"  终值:       ¥{bt_result.equity_curve.iloc[-1]:,.0f}")
    print("=" * 50)
    print(f"\n📋 HTML报告: {OUTPUT_DIR / 'stat_arb_report.html'}")


if __name__ == "__main__":
    main()
