#!/usr/bin/env python3
"""
Quant2026 ML Strategy Demo Pipeline
====================================
滚动训练 LightGBM/XGBoost → 因子预测 → 回测 → HTML 报告

用法: python3 examples/demo_ml_pipeline.py
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
from quant2026.factor.library import (
    MomentumFactor, VolatilityFactor, TurnoverFactor,
    RSIFactor, MACDFactor, BollingerFactor,
)
from quant2026.factor.preprocessing import FactorPreprocessor
from quant2026.strategy.ml_model import MLStrategy, MLTrainer
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.risk.manager import RiskManager
from quant2026.backtest.engine import BacktestEngine, BacktestConfig
from quant2026.backtest.report import BacktestReporter
from quant2026.types import PortfolioTarget

# ── 配置 ───────────────────────────────────────────────────────
START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 12, 31)
REBALANCE_INTERVAL = 20
MODEL_TYPE = "lightgbm"   # "lightgbm" | "xgboost"
FORWARD_DAYS = 20
TRAIN_WINDOW = 120        # 训练用的历史调仓期数 (取最近的N个采样点)
TOP_N = 15

STOCK_POOL: list[str] = [
    "600519", "000858", "601318", "600036", "000333",
    "600276", "601166", "000001", "600900", "601888",
    "000651", "600030", "601398", "600809", "002714",
    "600887", "000568", "601012", "600585", "002415",
    "601668", "600031", "000002", "600048", "601899",
    "002304", "600309", "601601", "000725", "002475",
]

FACTORS = [
    MomentumFactor(window=20),
    VolatilityFactor(window=20),
    TurnoverFactor(window=20),
    RSIFactor(window=14),
    MACDFactor(),
    BollingerFactor(window=20),
]

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")


def compute_factors(data: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Compute and preprocess factor matrix for a given date."""
    preprocessor = FactorPreprocessor()
    factor_values: dict[str, pd.Series] = {}
    for f in FACTORS:
        try:
            vals = f.compute(data, target_date)
            if not vals.empty:
                factor_values[f.name] = vals
        except Exception:
            pass
    if not factor_values:
        return pd.DataFrame()
    fm = pd.DataFrame(factor_values).dropna(how="all")
    if fm.empty:
        return fm
    return preprocessor.full_pipeline(fm, industry=None)


def main() -> None:
    # ── Step 1: 数据获取 & 清洗 ──
    logger.info("Step 1: 获取行情数据")
    provider = CachedProvider(AkShareProvider())
    try:
        data = provider.get_daily_quotes(STOCK_POOL, START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"数据获取失败: {e}")
        return
    if data.empty:
        logger.error("无数据"); return

    logger.info(f"获取 {data['stock_code'].nunique()} 只股票, {len(data)} 条记录")

    cleaner = DataCleaner()
    try:
        stock_list = provider.get_stock_list()
        data = cleaner.remove_st_stocks(data, stock_list)
    except Exception:
        pass
    data = cleaner.remove_new_stocks(data, min_days=30)
    data = cleaner.handle_limit_up_down(data)
    data = cleaner.handle_suspended(data)
    data["date"] = data["date"].astype(str)
    logger.info(f"清洗后 {data['stock_code'].nunique()} 只股票")

    # ── Step 2: 确定调仓日 ──
    all_dates = sorted(data["date"].unique())
    rebalance_indices = list(range(59, len(all_dates), REBALANCE_INTERVAL))
    rebalance_dates = [date.fromisoformat(all_dates[i]) for i in rebalance_indices]
    logger.info(f"共 {len(all_dates)} 交易日, {len(rebalance_dates)} 次调仓")

    # ── Step 3: 构建训练数据 (用 MLTrainer) ──
    logger.info("Step 2: 构建 ML 训练数据")
    trainer = MLTrainer(forward_days=FORWARD_DAYS)
    # 采样日期：调仓日之间每10天取一个点，确保充分覆盖
    sample_dates = [date.fromisoformat(all_dates[i]) for i in range(59, len(all_dates), 10)]
    all_factor_matrices, all_returns = trainer.build_dataset(data, compute_factors, sample_dates)

    if len(all_factor_matrices) < 5:
        logger.error(f"训练数据不足 ({len(all_factor_matrices)} 期)")
        return

    # ── Step 4: 滚动训练 & 回测 ──
    logger.info("Step 3: 滚动训练 + 策略打分")
    strategy = MLStrategy(model_type=MODEL_TYPE, forward_days=FORWARD_DAYS, top_n=TOP_N)
    optimizer = PortfolioOptimizer(max_stocks=TOP_N, max_single_weight=0.10)
    risk_mgr = RiskManager(max_single_weight=0.10, min_stocks=5)
    targets: dict[date, PortfolioTarget] = {}

    sorted_fm_dates = sorted(all_factor_matrices.keys())

    for idx, rb_date in enumerate(rebalance_dates):
        # 用 rb_date 之前的数据训练
        train_dates = [d for d in sorted_fm_dates if d < rb_date][-TRAIN_WINDOW:]
        if len(train_dates) < 3:
            logger.debug(f"跳过 {rb_date}: 训练数据不足")
            continue

        train_fm = {d: all_factor_matrices[d] for d in train_dates}
        train_ret = {d: all_returns[d] for d in train_dates}

        try:
            strategy.fit(train_fm, train_ret)
        except ValueError as e:
            logger.warning(f"训练失败 {rb_date}: {e}")
            continue

        # 预测
        fm = compute_factors(data, rb_date)
        if fm.empty:
            continue
        result = strategy.generate(data, fm, rb_date)
        target = optimizer.combine([result], rb_date)
        targets[rb_date] = risk_mgr.check_pre_trade(target)

        if (idx + 1) % 3 == 0 or idx == len(rebalance_dates) - 1:
            logger.info(f"  调仓 {idx+1}/{len(rebalance_dates)}: {rb_date}, 持仓 {len(target.weights)} 只")

    if not targets:
        logger.error("未生成调仓目标"); return

    # ── Step 5: 回测 ──
    logger.info("Step 4: 回测")
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

    # ── Step 6: 特征重要性 ──
    if strategy._feature_importances is not None:
        print("\n📊 特征重要性 (最后一次训练):")
        for feat, imp in strategy._feature_importances.items():
            bar = "█" * int(imp / strategy._feature_importances.max() * 30)
            print(f"  {feat:<25} {imp:>8.1f}  {bar}")

    # ── Step 7: 报告 ──
    logger.info("Step 5: 生成报告")
    benchmark_df = pd.DataFrame()
    try:
        benchmark_df = provider.get_index_quotes(["000300"], START_DATE, END_DATE)
        if not benchmark_df.empty:
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
    except Exception as e:
        logger.warning(f"基准获取失败: {e}")

    try:
        reporter = BacktestReporter(bt_result, benchmark_df, benchmark_name="CSI 300")
        report_path = reporter.generate_html(OUTPUT_DIR / "ml_report.html")
        logger.info(f"HTML 报告: {report_path}")
    except Exception as e:
        logger.warning(f"报告生成失败: {e}")

    # 打印绩效
    metrics = bt_result.metrics
    print("\n" + "=" * 50)
    print(f"📊 ML Strategy ({MODEL_TYPE}) 回测报告")
    print("=" * 50)
    print(f"  回测区间:   {START_DATE} ~ {END_DATE}")
    print(f"  调仓次数:   {len(targets)}")
    for k, v in metrics.items():
        label = {
            "total_return": "总收益率", "annual_return": "年化收益率",
            "volatility": "年化波动率", "sharpe_ratio": "夏普比率",
            "max_drawdown": "最大回撤", "trade_count": "交易次数",
            "avg_turnover": "平均换手率",
        }.get(k, k)
        print(f"  {label:<12} {v}")
    print(f"  终值:       ¥{bt_result.equity_curve.iloc[-1]:,.0f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
