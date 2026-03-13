#!/usr/bin/env python3
"""
Walk-Forward Analysis Demo
===========================
Demonstrates rolling out-of-sample backtesting using WalkForwardAnalyzer.

Usage: python3 examples/demo_walk_forward.py
"""

import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from loguru import logger

# ── Project root ───────────────────────────────────────────────
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
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.backtest.engine import BacktestConfig
from quant2026.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig
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


def make_strategy_factory(full_data: pd.DataFrame):
    """Create a strategy_factory closure that captures full_data for reference.

    Returns a factory function: (train_data, train_dates) -> dict[date, PortfolioTarget]
    """
    factors = [
        MomentumFactor(window=20),
        VolatilityFactor(window=20),
        TurnoverFactor(window=20),
        RSIFactor(window=14),
        MACDFactor(),
        BollingerFactor(window=20),
    ]
    preprocessor = FactorPreprocessor()
    strategy = MultiFactorStrategy()
    optimizer = PortfolioOptimizer(
        max_stocks=15, max_single_weight=0.10,
        method="equal_weight",
    )

    def factory(train_data: pd.DataFrame, train_dates: list[date]) -> dict[date, PortfolioTarget]:
        """Given training data, produce rebalance targets."""
        targets: dict[date, PortfolioTarget] = {}

        # Ensure date column is string for factor computation
        td = train_data.copy()
        td["date"] = td["date"].astype(str)
        all_dates = sorted(td["date"].unique())

        # Pick rebalance dates from later part of training (ensures enough history)
        rebal_indices = list(range(59, len(all_dates), REBALANCE_INTERVAL))
        if not rebal_indices:
            # At least use last date
            rebal_indices = [len(all_dates) - 1] if len(all_dates) > 60 else []

        for idx in rebal_indices:
            dt_str = all_dates[idx]
            target_date = date.fromisoformat(dt_str)

            factor_values: dict[str, pd.Series] = {}
            for f in factors:
                try:
                    vals = f.compute(td, target_date)
                    if not vals.empty:
                        factor_values[f.name] = vals
                except Exception:
                    continue

            if not factor_values:
                continue

            factor_matrix = pd.DataFrame(factor_values).dropna(how="all")
            if factor_matrix.empty:
                continue

            factor_matrix = preprocessor.full_pipeline(factor_matrix, industry=None)
            result = strategy.generate(td, factor_matrix, target_date)
            target = optimizer.combine([result], target_date, price_data=td)
            targets[target_date] = target

        # Also produce a target at the very end of training (to carry into test period)
        if all_dates and len(all_dates) > 60:
            last_dt_str = all_dates[-1]
            last_date = date.fromisoformat(last_dt_str)
            if last_date not in targets:
                factor_values = {}
                for f in factors:
                    try:
                        vals = f.compute(td, last_date)
                        if not vals.empty:
                            factor_values[f.name] = vals
                    except Exception:
                        continue
                if factor_values:
                    fm = pd.DataFrame(factor_values).dropna(how="all")
                    if not fm.empty:
                        fm = preprocessor.full_pipeline(fm, industry=None)
                        res = strategy.generate(td, fm, last_date)
                        tgt = optimizer.combine([res], last_date, price_data=td)
                        targets[last_date] = tgt

        return targets

    return factory


def main() -> None:
    """Run walk-forward analysis demo."""

    # Step 1: Fetch data
    logger.info("=" * 60)
    logger.info("Walk-Forward Demo: Fetching data")
    provider = CachedProvider(AkShareProvider())

    try:
        data = provider.get_daily_quotes(STOCK_POOL, START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return

    if data.empty:
        logger.error("No data, exit")
        return

    logger.info(f"Got {data['stock_code'].nunique()} stocks, {len(data)} rows")

    # Step 2: Clean
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

    logger.info(f"After cleaning: {data['stock_code'].nunique()} stocks")

    # Step 3: Walk-forward config
    wf_config = WalkForwardConfig(
        train_months=6,
        test_months=2,
        step_months=2,
        min_train_days=100,
    )
    bt_config = BacktestConfig(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=1_000_000,
        commission_rate=0.0003,
        stamp_tax_rate=0.0005,
        slippage_pct=0.001,
    )

    # Step 4: Run
    analyzer = WalkForwardAnalyzer(wf_config)
    factory = make_strategy_factory(data)
    result = analyzer.run(data, factory, bt_config)

    if not result.windows:
        logger.error("No walk-forward windows completed")
        return

    # Step 5: Print results
    print("\n" + "=" * 60)
    print("📊 Walk-Forward Analysis Results")
    print("=" * 60)
    print(f"  Windows completed: {len(result.windows)}")
    print(f"  Efficiency ratio:  {result.efficiency_ratio:.3f}")

    if result.efficiency_ratio < 0.5:
        print("  ⚠️  OVERFITTING WARNING: Efficiency ratio < 0.5")
    elif result.efficiency_ratio < 0.8:
        print("  ⚡ Moderate efficiency — some overfitting possible")
    else:
        print("  ✅ Good efficiency — strategy generalises well")

    print("\n  Per-window comparison:")
    print(f"  {'Window':<8} {'Train Period':<25} {'Test Period':<25} {'IS Sharpe':>10} {'OOS Sharpe':>11}")
    print("  " + "-" * 80)
    for w in result.windows:
        is_s = w.in_sample_metrics.get("sharpe_ratio", "N/A")
        oos_s = w.out_sample_metrics.get("sharpe_ratio", "N/A")
        print(f"  W{w.window_id:<6} {str(w.train_start)}~{str(w.train_end):<13} "
              f"{str(w.test_start)}~{str(w.test_end):<13} {is_s:>10} {oos_s:>11}")

    print("\n  Combined OOS metrics:")
    for k, v in result.combined_metrics.items():
        print(f"    {k:<16} {v}")
    print("=" * 60)

    # Step 6: Generate HTML report
    report_path = OUTPUT_DIR / "walk_forward_report.html"
    analyzer.generate_report(result, str(report_path))
    print(f"\n📋 HTML Report: {report_path}")


if __name__ == "__main__":
    main()
