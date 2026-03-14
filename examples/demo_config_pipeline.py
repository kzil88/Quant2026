#!/usr/bin/env python3
"""
Quant2026 Config-Driven Demo Pipeline
======================================
用 YAML 配置文件驱动全流程：数据获取 → 因子计算 → 策略 → 优化 → 风控 → 回测

用法:
    python3 examples/demo_config_pipeline.py                       # 默认配置
    python3 examples/demo_config_pipeline.py config/aggressive.yaml  # 激进配置
"""

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

# ── 项目根目录 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant2026.logging import setup_logging
setup_logging(level="INFO", log_dir="logs")

from quant2026.config import ConfigLoader
from quant2026.factory import ComponentFactory
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
from quant2026.backtest.engine import BacktestEngine
from quant2026.backtest.report import BacktestReporter
from quant2026.types import PortfolioTarget

# ── loguru ──────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

# ── rebalance frequency → interval ─────────────────────────────
_FREQ_MAP = {"daily": 1, "weekly": 5, "monthly": 20}


def main(config_path: str | None = None) -> None:
    """Run config-driven pipeline."""

    # ── 加载配置 ────────────────────────────────────────────────
    path = Path(config_path or PROJECT_ROOT / "config" / "default.yaml")
    cfg = ConfigLoader.load(path)
    errors = ConfigLoader.validate(cfg)
    if errors:
        for e in errors:
            logger.error(f"配置错误: {e}")
        return
    logger.info(f"配置已加载: {path.name}  策略={[s.name for s in cfg.strategies]}")

    # ── 从配置创建组件 ──────────────────────────────────────────
    strategies_with_weights = ComponentFactory.create_strategies(cfg)
    optimizer = ComponentFactory.create_optimizer(cfg)
    risk_mgr = ComponentFactory.create_risk_manager(cfg)
    bt_config = ComponentFactory.create_backtest_config(cfg)

    START_DATE = date.fromisoformat(cfg.data.start_date)
    END_DATE = date.fromisoformat(cfg.data.end_date)
    STOCK_POOL = cfg.data.stock_pool
    REBALANCE_INTERVAL = _FREQ_MAP.get(cfg.backtest.rebalance_frequency, 20)

    OUTPUT_DIR = Path(cfg.output.dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 数据获取 ────────────────────────────────────────
    logger.info("Step 1/7: 获取行情数据")
    provider = CachedProvider(AkShareProvider())
    try:
        data = provider.get_daily_quotes(STOCK_POOL, START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"数据获取失败: {e}")
        return
    if data.empty:
        logger.error("未获取到任何数据")
        return
    logger.info(f"获取到 {data['stock_code'].nunique()} 只股票, {len(data)} 条记录")

    # ── Step 2: 数据清洗 ────────────────────────────────────────
    logger.info("Step 2/7: 数据清洗")
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
    valid_stocks = data["stock_code"].unique()
    logger.info(f"清洗后 {len(valid_stocks)} 只股票")
    if len(valid_stocks) < 5:
        logger.error("有效股票太少")
        return

    # ── Step 2b: 财务数据 ───────────────────────────────────────
    logger.info("Step 2b/7: 获取财务数据")
    financial_data = None
    try:
        financial_data = provider.get_financial_data(list(valid_stocks), END_DATE)
        if financial_data is None or financial_data.empty:
            financial_data = None
    except Exception:
        financial_data = None

    # ── Step 3: 因子 ────────────────────────────────────────────
    logger.info("Step 3/7: 因子计算")
    factors = [
        MomentumFactor(window=20), VolatilityFactor(window=20),
        TurnoverFactor(window=20), RSIFactor(window=14),
        MACDFactor(), BollingerFactor(window=20),
    ]
    if financial_data is not None:
        factors.extend([
            PEFactor(), PBFactor(), DividendYieldFactor(),
            ROEFactor(), GrossMarginFactor(), DebtRatioFactor(),
            RevenueGrowthFactor(), ProfitGrowthFactor(),
        ])

    preprocessor = FactorPreprocessor()
    all_dates = sorted(data["date"].unique())
    rebalance_indices = list(range(59, len(all_dates), REBALANCE_INTERVAL))
    rebalance_dates_str = [all_dates[i] for i in rebalance_indices]
    logger.info(f"调仓 {len(rebalance_dates_str)} 次")

    # ── Step 4 & 5: 策略打分 + 组合优化 ────────────────────────
    logger.info("Step 4/7: 策略打分 & 组合优化")
    targets: dict[date, PortfolioTarget] = {}
    current_weights: pd.Series | None = None

    for idx, dt_str in enumerate(rebalance_dates_str):
        target_date = date.fromisoformat(dt_str)

        # 因子矩阵
        factor_values: dict[str, pd.Series] = {}
        for f in factors:
            try:
                vals = f.compute(data, target_date, financial_data=financial_data)
                if not vals.empty:
                    factor_values[f.name] = vals
            except Exception:
                pass
        if not factor_values:
            continue
        factor_matrix = pd.DataFrame(factor_values).dropna(how="all")
        if factor_matrix.empty:
            continue
        factor_matrix = preprocessor.full_pipeline(factor_matrix, industry=None)

        # 各策略打分
        results = []
        for strategy, _w in strategies_with_weights:
            try:
                r = strategy.generate(data, factor_matrix, target_date)
                results.append(r)
            except Exception:
                pass

        if not results:
            continue

        target = optimizer.combine(
            results, target_date, price_data=data,
            current_weights=current_weights,
        )
        targets[target_date] = target
        current_weights = target.weights

        if (idx + 1) % 3 == 0 or idx == len(rebalance_dates_str) - 1:
            logger.info(f"  调仓 {idx+1}/{len(rebalance_dates_str)}: {dt_str}, 持仓 {len(target.weights)} 只")

    if not targets:
        logger.error("未生成任何调仓目标")
        return

    # ── Step 6: 风控 ────────────────────────────────────────────
    logger.info("Step 5/7: 风控检查")
    for dt, tgt in targets.items():
        targets[dt] = risk_mgr.check_pre_trade(tgt)

    # ── Step 7: 回测 ────────────────────────────────────────────
    logger.info("Step 6/7: 回测")
    engine = BacktestEngine(bt_config)
    bt_data = data.copy()
    bt_data["date"] = pd.to_datetime(bt_data["date"]).dt.date
    result = engine.run(bt_data, targets)

    if result.equity_curve.empty:
        logger.error("回测未产生结果")
        return

    risk_metrics = risk_mgr.check_post_trade(result.equity_curve)

    # ── Step 7b: 报告 ──────────────────────────────────────────
    logger.info("Step 7/7: 生成报告")
    benchmark_df = pd.DataFrame()
    try:
        benchmark_df = provider.get_index_quotes([cfg.backtest.benchmark], START_DATE, END_DATE)
        if not benchmark_df.empty:
            benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
    except Exception:
        pass

    if cfg.output.report:
        try:
            reporter = BacktestReporter(result, benchmark_df, benchmark_name="CSI 300")
            reporter.generate_html(OUTPUT_DIR / "report.html")
        except Exception as e:
            logger.warning(f"HTML 报告失败: {e}")

    # ── 打印结果 ────────────────────────────────────────────────
    metrics = result.metrics
    print("\n" + "=" * 50)
    print(f"📊 回测绩效报告 ({path.name})")
    print("=" * 50)
    print(f"  配置文件:   {path}")
    print(f"  回测区间:   {START_DATE} ~ {END_DATE}")
    print(f"  股票池:     {len(STOCK_POOL)} 只")
    print(f"  调仓次数:   {len(targets)}")
    print("-" * 50)
    for k, v in metrics.items():
        label = {
            "total_return": "总收益率", "annual_return": "年化收益率",
            "volatility": "年化波动率", "sharpe_ratio": "夏普比率",
            "max_drawdown": "最大回撤", "trade_count": "交易次数",
            "avg_turnover": "平均换手率",
        }.get(k, k)
        print(f"  {label:<12} {v}")
    print(f"  终值:       ¥{result.equity_curve.iloc[-1]:,.0f}")
    print("=" * 50)

    if cfg.output.equity_curve:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
        equity = result.equity_curve / result.equity_curve.iloc[0]
        axes[0].plot(equity.index, equity.values, "b-", linewidth=1.5, label="Strategy NAV")
        axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title(f"Quant2026 Backtest ({path.stem})", fontsize=14)
        axes[0].set_ylabel("NAV")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150)
        plt.close()
        logger.info(f"净值曲线: {OUTPUT_DIR / 'equity_curve.png'}")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_file)
