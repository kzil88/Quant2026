#!/usr/bin/env python3
"""
Quant2026 Demo Pipeline
=======================
End-to-end: 数据获取 → 因子计算 → 策略打分 → 组合优化 → 风控 → 回测 → 可视化

用法: python3 examples/demo_pipeline.py
"""

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

# ── 项目根目录加入 path ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant2026.data.akshare_provider import AkShareProvider
from quant2026.data.cache import CachedProvider
from quant2026.data.cleaner import DataCleaner
from quant2026.factor.library import (
    MomentumFactor, VolatilityFactor, TurnoverFactor,
    RSIFactor, MACDFactor, BollingerFactor,
    PEFactor, PBFactor, DividendYieldFactor,
    ROEFactor, GrossMarginFactor, DebtRatioFactor,
    RevenueGrowthFactor, ProfitGrowthFactor,
)
from quant2026.factor.preprocessing import FactorPreprocessor
from quant2026.strategy.multi_factor.strategy import MultiFactorStrategy
from quant2026.strategy.mean_reversion.strategy import MeanReversionStrategy
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.portfolio.turnover import TurnoverConstraint
from quant2026.risk.manager import RiskManager
from quant2026.backtest.engine import BacktestEngine, BacktestConfig
from quant2026.backtest.report import BacktestReporter
from quant2026.types import PortfolioTarget

# ── 配置 ───────────────────────────────────────────────────────
START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 12, 31)
REBALANCE_INTERVAL = 20  # 每20个交易日调仓

# 硬编码沪深300代表性子集 (30只大盘蓝筹)
STOCK_POOL: list[str] = [
    "600519",  # 贵州茅台
    "000858",  # 五粮液
    "601318",  # 中国平安
    "600036",  # 招商银行
    "000333",  # 美的集团
    "600276",  # 恒瑞医药
    "601166",  # 兴业银行
    "000001",  # 平安银行
    "600900",  # 长江电力
    "601888",  # 中国中免
    "000651",  # 格力电器
    "600030",  # 中信证券
    "601398",  # 工商银行
    "600809",  # 山西汾酒
    "002714",  # 牧原股份
    "600887",  # 伊利股份
    "000568",  # 泸州老窖
    "601012",  # 隆基绿能
    "600585",  # 海螺水泥
    "002415",  # 海康威视
    "601668",  # 中国建筑
    "600031",  # 三一重工
    "000002",  # 万科A
    "600048",  # 保利发展
    "601899",  # 紫金矿业
    "002304",  # 洋河股份
    "600309",  # 万华化学
    "601601",  # 中国太保
    "000725",  # 京东方A
    "002475",  # 立讯精密
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── loguru 配置 ─────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")


def main() -> None:
    """Run the full demo pipeline."""

    # ================================================================
    # Step 1: 数据获取
    # ================================================================
    logger.info("=" * 60)
    logger.info("Step 1/7: 获取行情数据")
    provider = CachedProvider(AkShareProvider())

    try:
        data = provider.get_daily_quotes(STOCK_POOL, START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"数据获取失败: {e}")
        logger.info("提示: 检查网络连接或 akshare 版本")
        return

    if data.empty:
        logger.error("未获取到任何数据，退出")
        return

    logger.info(f"获取到 {data['stock_code'].nunique()} 只股票, {len(data)} 条记录")

    # ================================================================
    # Step 2: 数据清洗
    # ================================================================
    logger.info("Step 2/7: 数据清洗")
    cleaner = DataCleaner()

    # 获取股票列表用于 ST 过滤
    try:
        stock_list = provider.get_stock_list()
        data = cleaner.remove_st_stocks(data, stock_list)
    except Exception as e:
        logger.warning(f"获取股票列表失败，跳过 ST 过滤: {e}")

    data = cleaner.remove_new_stocks(data, min_days=30)
    data = cleaner.handle_limit_up_down(data)
    data = cleaner.handle_suspended(data)

    # 确保 date 列为字符串 (因子计算需要)
    data["date"] = data["date"].astype(str)

    valid_stocks = data["stock_code"].unique()
    logger.info(f"清洗后剩余 {len(valid_stocks)} 只股票")

    if len(valid_stocks) < 5:
        logger.error("有效股票太少，无法继续")
        return

    # ================================================================
    # Step 2b: 获取财务数据
    # ================================================================
    logger.info("Step 2b/7: 获取财务数据")
    financial_data: pd.DataFrame | None = None
    try:
        financial_data = provider.get_financial_data(list(valid_stocks), END_DATE)
        if financial_data is not None and not financial_data.empty:
            logger.info(f"获取到 {len(financial_data)} 只股票的财务数据")
        else:
            logger.warning("财务数据为空，将仅使用技术因子")
            financial_data = None
    except Exception as e:
        logger.warning(f"财务数据获取失败，将仅使用技术因子: {e}")
        financial_data = None

    # ================================================================
    # Step 3: 因子计算 & 预处理
    # ================================================================
    logger.info("Step 3/7: 因子计算")
    # Technical factors (always available)
    factors = [
        MomentumFactor(window=20),
        VolatilityFactor(window=20),
        TurnoverFactor(window=20),
        RSIFactor(window=14),
        MACDFactor(),
        BollingerFactor(window=20),
    ]
    # Financial factors (need financial_data)
    if financial_data is not None:
        factors.extend([
            PEFactor(),
            PBFactor(),
            DividendYieldFactor(),
            ROEFactor(),
            GrossMarginFactor(),
            DebtRatioFactor(),
            RevenueGrowthFactor(),
            ProfitGrowthFactor(),
        ])
        logger.info(f"使用 {len(factors)} 个因子 (技术 + 财务)")
    else:
        logger.info(f"使用 {len(factors)} 个技术因子")

    preprocessor = FactorPreprocessor()

    # 获取所有交易日
    all_dates = sorted(data["date"].unique())
    logger.info(f"共 {len(all_dates)} 个交易日")

    # 确定调仓日
    rebalance_indices = list(range(59, len(all_dates), REBALANCE_INTERVAL))  # 从第60天开始确保因子有足够数据
    rebalance_dates_str = [all_dates[i] for i in rebalance_indices]
    logger.info(f"调仓 {len(rebalance_dates_str)} 次")

    # ================================================================
    # Step 4 & 5: 策略打分 + 组合优化
    # ================================================================
    logger.info("Step 4/7: 策略打分 & 组合优化")
    multi_factor_strategy = MultiFactorStrategy()  # 等权
    mean_reversion_strategy = MeanReversionStrategy(window=20, use_bollinger=True, use_rsi=True)
    turnover_constraint = TurnoverConstraint(max_turnover=0.3, penalty_weight=0.01)
    optimizer = PortfolioOptimizer(
        max_stocks=15, max_single_weight=0.10,
        method="markowitz",           # 使用 Markowitz max_sharpe 优化
        markowitz_method="max_sharpe",
        strategy_weights={
            multi_factor_strategy.name: 0.6,
            mean_reversion_strategy.name: 0.4,
        },
        turnover_constraint=turnover_constraint,
    )

    targets: dict[date, PortfolioTarget] = {}
    current_weights: pd.Series | None = None  # 维护当前持仓权重

    for idx, dt_str in enumerate(rebalance_dates_str):
        target_date = date.fromisoformat(dt_str)

        # 计算因子矩阵
        factor_values: dict[str, pd.Series] = {}
        for f in factors:
            try:
                vals = f.compute(data, target_date, financial_data=financial_data)
                if not vals.empty:
                    factor_values[f.name] = vals
            except Exception as e:
                logger.debug(f"因子 {f.name} 在 {dt_str} 计算失败: {e}")

        if not factor_values:
            continue

        factor_matrix = pd.DataFrame(factor_values).dropna(how="all")
        if factor_matrix.empty:
            continue

        # 预处理 (去极值 + 标准化，不做行业中性化因为没有行业数据)
        factor_matrix = preprocessor.full_pipeline(factor_matrix, industry=None)

        # 策略打分
        mf_result = multi_factor_strategy.generate(data, factor_matrix, target_date)
        mr_result = mean_reversion_strategy.generate(data, None, target_date)

        # 组合优化 (multi_factor: 0.6, mean_reversion: 0.4) — 传入行情数据和当前权重
        target = optimizer.combine(
            [mf_result, mr_result], target_date, price_data=data,
            current_weights=current_weights,
        )
        targets[target_date] = target

        # 打印换手率和预估成本
        if current_weights is not None:
            cost_info = turnover_constraint.estimate_cost(target.weights, current_weights)
            logger.info(
                f"  调仓 {dt_str}: 换手率={cost_info['turnover']:.2%}, "
                f"预估成本={cost_info['total_cost_pct']:.4%}"
            )

        # 更新当前权重
        current_weights = target.weights

        if (idx + 1) % 3 == 0 or idx == len(rebalance_dates_str) - 1:
            logger.info(f"  调仓 {idx+1}/{len(rebalance_dates_str)}: {dt_str}, 持仓 {len(target.weights)} 只")

    if not targets:
        logger.error("未生成任何调仓目标")
        return

    # ================================================================
    # Step 6: 风控检查
    # ================================================================
    logger.info("Step 5/7: 风控检查")
    risk_mgr = RiskManager(max_single_weight=0.10, min_stocks=5)

    for dt, tgt in targets.items():
        targets[dt] = risk_mgr.check_pre_trade(tgt)

    logger.info(f"风控检查完成，{len(targets)} 个调仓目标")

    # ================================================================
    # Step 7: 回测
    # ================================================================
    logger.info("Step 6/7: 回测")
    config = BacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=1_000_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        slippage_pct=0.001,
    )
    engine = BacktestEngine(config)

    # BacktestEngine 需要 date 列为可比较类型
    bt_data = data.copy()
    bt_data["date"] = pd.to_datetime(bt_data["date"]).dt.date
    result = engine.run(bt_data, targets)

    if result.equity_curve.empty:
        logger.error("回测未产生结果")
        return

    # 风控 post-trade
    risk_metrics = risk_mgr.check_post_trade(result.equity_curve)

    # ================================================================
    # Step 7b: 获取基准数据 & 生成HTML报告
    # ================================================================
    logger.info("Step 7/7: 生成报告")

    # Fetch CSI 300 benchmark
    benchmark_df = pd.DataFrame()
    try:
        logger.info("Fetching CSI 300 benchmark data...")
        benchmark_df = provider.get_index_quotes(["000300"], START_DATE, END_DATE)
        if not benchmark_df.empty:
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
            logger.info(f"Benchmark: {len(benchmark_df)} rows")
    except Exception as e:
        logger.warning(f"Failed to fetch benchmark: {e}")

    # Generate HTML report
    try:
        reporter = BacktestReporter(result, benchmark_df, benchmark_name="CSI 300")
        report_path = reporter.generate_html(OUTPUT_DIR / "report.html")
        logger.info(f"HTML report: {report_path}")
    except Exception as e:
        logger.warning(f"HTML report generation failed: {e}")

    # ================================================================
    # Step 8: 输出结果
    # ================================================================

    # 打印绩效指标
    metrics = result.metrics
    print("\n" + "=" * 50)
    print("📊 回测绩效报告")
    print("=" * 50)
    print(f"  回测区间:   {START_DATE} ~ {END_DATE}")
    print(f"  股票池:     {len(STOCK_POOL)} 只 → 有效 {len(valid_stocks)} 只")
    print(f"  调仓次数:   {len(targets)}")
    print(f"  初始资金:   ¥1,000,000")
    print("-" * 50)
    for k, v in metrics.items():
        label = {
            "total_return": "总收益率",
            "annual_return": "年化收益率",
            "volatility": "年化波动率",
            "sharpe_ratio": "夏普比率",
            "max_drawdown": "最大回撤",
            "trade_count": "交易次数",
            "avg_turnover": "平均换手率",
        }.get(k, k)
        print(f"  {label:<12} {v}")
    print("-" * 50)
    print(f"  终值:       ¥{result.equity_curve.iloc[-1]:,.0f}")

    if risk_metrics.warnings:
        print("\n⚠️  风险警告:")
        for w in risk_metrics.warnings:
            print(f"    - {w}")
    print("=" * 50)

    # 绘制净值曲线
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # 净值曲线
    ax1 = axes[0]
    equity = result.equity_curve / result.equity_curve.iloc[0]
    ax1.plot(equity.index, equity.values, "b-", linewidth=1.5, label="Strategy NAV")
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_title("Quant2026 Multi-Factor + Mean Reversion Strategy Backtest (2024)", fontsize=14)
    ax1.set_ylabel("NAV")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 回撤曲线
    ax2 = axes[1]
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "equity_curve.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"净值曲线已保存: {out_path}")
    print(f"\n📈 净值曲线: {out_path}")
    if not benchmark_df.empty:
        print(f"📋 HTML报告: {OUTPUT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
