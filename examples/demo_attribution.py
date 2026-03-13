#!/usr/bin/env python3
"""
Quant2026 Attribution Demo
==========================
Runs performance attribution analysis on backtest results.

Usage: python3 examples/demo_attribution.py
"""

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
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
from quant2026.strategy.multi_factor.strategy import MultiFactorStrategy
from quant2026.strategy.mean_reversion.strategy import MeanReversionStrategy
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.risk.manager import RiskManager
from quant2026.backtest.engine import BacktestEngine, BacktestConfig
from quant2026.backtest.attribution import PerformanceAttribution
from quant2026.types import PortfolioTarget

# ── Config ─────────────────────────────────────────────────────
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
    """Run attribution analysis demo."""

    # ── Step 1: Data ───────────────────────────────────────────
    logger.info("Step 1: Fetching data")
    provider = CachedProvider(AkShareProvider())

    try:
        data = provider.get_daily_quotes(STOCK_POOL, START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return

    if data.empty:
        logger.error("No data fetched")
        return

    # Clean
    cleaner = DataCleaner()
    try:
        stock_list = provider.get_stock_list()
        data = cleaner.remove_st_stocks(data, stock_list)
    except Exception:
        pass
    data = cleaner.remove_new_stocks(data, min_days=30)
    data["date"] = data["date"].astype(str)
    valid_stocks = data["stock_code"].unique()
    logger.info(f"Valid stocks: {len(valid_stocks)}")

    # ── Step 2: Factor calculation & backtest (same as demo_pipeline) ─
    logger.info("Step 2: Factor calculation & backtest")
    factors = [
        MomentumFactor(window=20), VolatilityFactor(window=20),
        TurnoverFactor(window=20), RSIFactor(window=14),
        MACDFactor(), BollingerFactor(window=20),
    ]
    preprocessor = FactorPreprocessor()
    multi_factor_strategy = MultiFactorStrategy()
    mean_reversion_strategy = MeanReversionStrategy(window=20, use_bollinger=True, use_rsi=True)
    optimizer = PortfolioOptimizer(
        max_stocks=15, max_single_weight=0.10,
        method="markowitz", markowitz_method="max_sharpe",
        strategy_weights={
            multi_factor_strategy.name: 0.6,
            mean_reversion_strategy.name: 0.4,
        },
    )

    all_dates = sorted(data["date"].unique())
    rebalance_indices = list(range(59, len(all_dates), REBALANCE_INTERVAL))
    rebalance_dates_str = [all_dates[i] for i in rebalance_indices]

    targets: dict[date, PortfolioTarget] = {}
    for dt_str in rebalance_dates_str:
        target_date = date.fromisoformat(dt_str)
        factor_values: dict[str, pd.Series] = {}
        for f in factors:
            try:
                vals = f.compute(data, target_date)
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
        mf_result = multi_factor_strategy.generate(data, factor_matrix, target_date)
        mr_result = mean_reversion_strategy.generate(data, None, target_date)
        target = optimizer.combine([mf_result, mr_result], target_date, price_data=data)
        targets[target_date] = target

    if not targets:
        logger.error("No rebalance targets generated")
        return

    risk_mgr = RiskManager(max_single_weight=0.10, min_stocks=5)
    for dt, tgt in targets.items():
        targets[dt] = risk_mgr.check_pre_trade(tgt)

    config = BacktestConfig(
        start_date=START_DATE, end_date=END_DATE,
        initial_capital=1_000_000,
    )
    engine = BacktestEngine(config)
    bt_data = data.copy()
    bt_data["date"] = pd.to_datetime(bt_data["date"]).dt.date
    result = engine.run(bt_data, targets)

    if result.equity_curve.empty:
        logger.error("Backtest produced no results")
        return

    # ── Step 3: Prepare attribution inputs ─────────────────────
    logger.info("Step 3: Preparing attribution inputs")

    # Build returns matrix (daily returns per stock)
    pivot_close = data.pivot_table(index="date", columns="stock_code", values="close")
    pivot_close = pivot_close.sort_index()
    returns_matrix = pivot_close.pct_change().dropna(how="all")

    # Portfolio weights dict
    portfolio_weights: dict[date, pd.Series] = {}
    for dt, tgt in targets.items():
        portfolio_weights[dt] = tgt.weights

    # Benchmark weights: equal-weight across stock pool
    bm_stocks = list(valid_stocks)
    benchmark_weights = pd.Series(1.0 / len(bm_stocks), index=bm_stocks)

    # Industry classification
    logger.info("Fetching industry classification...")
    try:
        industry_df = provider.get_industry_classification()
        if not industry_df.empty:
            industry_map = industry_df.set_index("stock_code")["industry_l1"]
            logger.info(f"Industry map: {len(industry_map)} stocks")
        else:
            raise ValueError("Empty industry data")
    except Exception as e:
        logger.warning(f"Industry fetch failed ({e}), using mock classification")
        # Fallback: assign industries based on stock code ranges
        industry_map_dict = {}
        sectors = ["Banking", "Consumer", "Tech", "Industrial", "Energy", "Healthcare"]
        for i, s in enumerate(valid_stocks):
            industry_map_dict[s] = sectors[i % len(sectors)]
        industry_map = pd.Series(industry_map_dict)

    # ── Step 4: Run attribution ────────────────────────────────
    logger.info("Step 4: Running attribution analysis")
    attr = PerformanceAttribution()

    # Sector attribution
    sector_result = attr.sector_attribution(
        portfolio_weights=portfolio_weights,
        benchmark_weights=benchmark_weights,
        returns=returns_matrix,
        industry_map=industry_map,
    )
    print("\n📊 Sector Attribution:")
    print(sector_result.to_string())

    # Factor attribution — construct simple factor returns
    # Use momentum, volatility as proxy factor returns
    logger.info("Computing factor proxy returns...")
    factor_rets = pd.DataFrame(index=returns_matrix.index)
    # Market factor: equal-weight portfolio return
    factor_rets["market"] = returns_matrix.mean(axis=1)
    # Size proxy: small vs large (by code, rough)
    sorted_stocks = sorted(returns_matrix.columns)
    mid = len(sorted_stocks) // 2
    factor_rets["size"] = returns_matrix[sorted_stocks[:mid]].mean(axis=1) - returns_matrix[sorted_stocks[mid:]].mean(axis=1)
    # Momentum proxy: top vs bottom momentum stocks
    mom20 = returns_matrix.rolling(20).sum()
    factor_rets["momentum"] = mom20.apply(lambda row: row.nlargest(10).mean() - row.nsmallest(10).mean(), axis=1).fillna(0)

    portfolio_daily_returns = result.daily_returns
    factor_result = attr.factor_attribution(
        portfolio_returns=portfolio_daily_returns,
        factor_returns=factor_rets,
    )
    print("\n📊 Factor Attribution:")
    print(f"  Alpha (daily): {factor_result['alpha']:.6f}")
    print(f"  R²: {factor_result['r_squared']:.4f}")
    for f, beta in factor_result["factor_exposures"].items():
        print(f"  {f}: β={beta:.4f}, contribution={factor_result['factor_contributions'][f]:.4%}")

    # Monthly attribution
    # Benchmark returns
    benchmark_df = pd.DataFrame()
    try:
        benchmark_df = provider.get_index_quotes(["000300"], START_DATE, END_DATE)
    except Exception:
        pass

    if not benchmark_df.empty:
        bm = benchmark_df.copy()
        bm["date"] = pd.to_datetime(bm["date"])
        bm = bm.sort_values("date").set_index("date")["close"]
        bm_daily_ret = bm.pct_change().dropna()
        # Align index types
        bm_daily_ret.index = bm_daily_ret.index.date
    else:
        logger.warning("No benchmark data, using equal-weight as benchmark")
        bm_daily_ret = returns_matrix.mean(axis=1)

    monthly_result = attr.monthly_attribution(
        portfolio_returns=portfolio_daily_returns,
        benchmark_returns=bm_daily_ret,
    )
    print("\n📊 Monthly Attribution:")
    print(monthly_result.to_string())

    # ── Step 5: Generate HTML report ───────────────────────────
    logger.info("Step 5: Generating HTML attribution report")
    report_path = attr.generate_report(
        sector_attr=sector_result,
        factor_attr=factor_result,
        monthly_attr=monthly_result,
        output_path=str(OUTPUT_DIR / "attribution_report.html"),
    )
    print(f"\n📋 Attribution report: {report_path}")


if __name__ == "__main__":
    main()
