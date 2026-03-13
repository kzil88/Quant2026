"""Demo: optimize MeanReversionStrategy parameters via grid search."""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant2026.backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from quant2026.optimization import ParamSpace, StrategyOptimizer
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.strategy.mean_reversion.strategy import MeanReversionStrategy


def make_synthetic_data(n_stocks: int = 20, n_days: int = 300) -> pd.DataFrame:
    """Generate synthetic A-share-like OHLCV data."""
    rng = np.random.RandomState(123)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    rows = []
    for i in range(n_stocks):
        code = f"{600000 + i:06d}"
        price = 10.0 + rng.randn() * 3
        for d in dates:
            ret = rng.normal(0.0003, 0.025)
            price *= 1 + ret
            price = max(price, 1.0)
            rows.append({
                "stock_code": code,
                "date": d.strftime("%Y-%m-%d"),
                "open": round(price * (1 + rng.uniform(-0.005, 0.005)), 2),
                "high": round(price * (1 + rng.uniform(0, 0.02)), 2),
                "low": round(price * (1 - rng.uniform(0, 0.02)), 2),
                "close": round(price, 2),
                "volume": int(rng.uniform(1e6, 1e7)),
            })
    return pd.DataFrame(rows)


def pipeline_fn(
    strategy: MeanReversionStrategy,
    data: pd.DataFrame,
    config: BacktestConfig,
) -> BacktestResult:
    """Full pipeline: strategy → portfolio → backtest."""
    # Generate signals at each rebalance date
    dates = sorted(data["date"].unique())
    rebalance_dates = dates[config.rebalance_days::config.rebalance_days]

    port_opt = PortfolioOptimizer(method="equal", top_n=5)
    targets = {}

    for dt_str in rebalance_dates:
        dt = pd.Timestamp(dt_str).date()
        if dt < config.start_date or dt > config.end_date:
            continue
        result = strategy.generate(data, None, dt)
        if result.scores.empty:
            continue
        target = port_opt.optimize(
            strategy_results=[result],
            price_data=data[data["date"] <= dt_str],
            target_date=dt,
        )
        if target is not None:
            targets[dt] = target

    engine = BacktestEngine(config)
    return engine.run(data, targets)


def main() -> None:
    logger.info("Generating synthetic data...")
    data = make_synthetic_data()

    config = BacktestConfig(
        start_date=date(2024, 3, 1),
        end_date=date(2025, 1, 31),
        initial_capital=1_000_000,
        rebalance_days=5,
    )

    # Define parameter spaces
    param_spaces = [
        ParamSpace(name="window", type="choice", choices=[10, 15, 20, 30, 40]),
        ParamSpace(
            name="zscore_threshold", type="choice",
            choices=[-2.0, -1.5, -1.0, -0.5],
        ),
    ]

    def strategy_factory(params: dict) -> MeanReversionStrategy:
        return MeanReversionStrategy(
            window=params["window"],
            zscore_threshold=params["zscore_threshold"],
        )

    optimizer = StrategyOptimizer(objective="sharpe")

    # Grid search (5 * 4 = 20 combos)
    logger.info("Running grid search...")
    result = optimizer.grid_search(
        param_spaces=param_spaces,
        strategy_factory=strategy_factory,
        data=data,
        backtest_config=config,
        pipeline_fn=pipeline_fn,
    )

    print("\n" + "=" * 60)
    print("GRID SEARCH RESULTS")
    print("=" * 60)
    print(f"Best params:  {result.best_params}")
    print(f"Best score:   {result.best_score:.4f}")
    print(f"Best metrics: {result.best_metrics}")
    print(f"Total time:   {result.optimization_time:.1f}s")
    print(f"Evaluations:  {len(result.all_results)}")

    # Generate HTML report
    output_dir = Path(__file__).parent / "output"
    report_path = str(output_dir / "optimization_report.html")
    optimizer.generate_report(result, param_spaces, report_path)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
