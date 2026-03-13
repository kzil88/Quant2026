#!/usr/bin/env python3
"""
Factor Evaluation Demo
======================
Compute IC/IR, IC decay, factor correlation for all available factors,
and generate a self-contained HTML report.

Usage: python3 examples/demo_factor_evaluation.py
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
    PEFactor, PBFactor, DividendYieldFactor,
    ROEFactor, GrossMarginFactor, DebtRatioFactor,
    RevenueGrowthFactor, ProfitGrowthFactor,
)
from quant2026.factor.preprocessing import FactorPreprocessor
from quant2026.factor.evaluation import FactorEvaluator

# ── Config ──────────────────────────────────────────────────────
START_DATE = date(2024, 1, 1)
END_DATE = date(2024, 12, 31)
FORWARD_PERIOD = 10  # default forward return period for IC
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
    """Run factor evaluation pipeline."""

    # ── Data ──
    logger.info("Step 1: Fetching data")
    provider = CachedProvider(AkShareProvider())
    try:
        data = provider.get_daily_quotes(STOCK_POOL, START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return
    if data.empty:
        logger.error("No data, exiting")
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
    logger.info(f"Data: {data['stock_code'].nunique()} stocks, {len(data)} rows")

    # Financial data (optional)
    financial_data = None
    try:
        financial_data = provider.get_financial_data(list(data["stock_code"].unique()), END_DATE)
        if financial_data is not None and financial_data.empty:
            financial_data = None
    except Exception:
        pass

    # ── Factors ──
    logger.info("Step 2: Computing factors")
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

    all_dates = sorted(data["date"].unique())
    eval_indices = list(range(59, len(all_dates) - FORWARD_PERIOD, REBALANCE_INTERVAL))
    eval_dates_str = [all_dates[i] for i in eval_indices]
    logger.info(f"Evaluation dates: {len(eval_dates_str)}")

    # Compute factor values per date
    factor_values_by_name: dict[str, dict[date, pd.Series]] = {f.name: {} for f in factors}
    preprocessor = FactorPreprocessor()

    for dt_str in eval_dates_str:
        target_date = date.fromisoformat(dt_str)
        for f in factors:
            try:
                vals = f.compute(data, target_date, financial_data=financial_data)
                if not vals.empty:
                    vals = preprocessor.winsorize(vals)
                    vals = preprocessor.standardize(vals)
                    factor_values_by_name[f.name][target_date] = vals
            except Exception:
                pass

    # ── Evaluate ──
    logger.info("Step 3: IC/IR evaluation")
    evaluator = FactorEvaluator()

    # Forward returns
    fwd_returns = evaluator._compute_forward_returns(data, FORWARD_PERIOD)

    ic_summaries: dict[str, dict] = {}
    ic_series_dict: dict[str, pd.Series] = {}
    decay_dict: dict[str, pd.DataFrame] = {}

    for f in factors:
        name = f.name
        fv = factor_values_by_name[name]
        if not fv:
            continue
        ic_s = evaluator.compute_ic_series(fv, fwd_returns)
        ic_series_dict[name] = ic_s
        ic_summaries[name] = evaluator.ic_summary(ic_s)

        # IC decay
        logger.info(f"  IC decay: {name}")
        decay_dict[name] = evaluator.ic_decay(data, fv, periods=[5, 10, 20, 40, 60])

    # ── Correlation ──
    logger.info("Step 4: Factor correlation")
    # Use last evaluation date for cross-sectional correlation
    last_date = date.fromisoformat(eval_dates_str[-1]) if eval_dates_str else None
    corr_df = pd.DataFrame()
    if last_date:
        factor_matrix = {}
        for f in factors:
            if last_date in factor_values_by_name[f.name]:
                factor_matrix[f.name] = factor_values_by_name[f.name][last_date]
        if factor_matrix:
            fm = pd.DataFrame(factor_matrix).dropna(how="all")
            corr_df = evaluator.factor_correlation(fm)

    # ── Report ──
    logger.info("Step 5: Generating report")
    report_path = evaluator.generate_report(
        ic_summaries, ic_series_dict, decay_dict, corr_df, str(OUTPUT_DIR)
    )

    # ── Terminal Summary ──
    print("\n" + "=" * 90)
    print("📊 Factor IC/IR Summary")
    print("=" * 90)
    print(f"{'Factor':<22} {'IC Mean':>8} {'IC Std':>8} {'IR':>8} {'IC>0%':>8} {'|IC|>.02':>8} {'t-stat':>8} {'N':>5}")
    print("-" * 90)
    for name, s in ic_summaries.items():
        print(
            f"{name:<22} {s['ic_mean']:>8.4f} {s['ic_std']:>8.4f} {s['ir']:>8.4f} "
            f"{s['ic_positive_ratio']:>7.1%} {s['ic_abs_gt_002_ratio']:>7.1%} "
            f"{s['t_stat']:>8.2f} {s['n_periods']:>5}"
        )
    print("=" * 90)
    print(f"\n📋 HTML Report: {report_path}")


if __name__ == "__main__":
    main()
