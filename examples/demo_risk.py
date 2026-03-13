#!/usr/bin/env python3
"""
Demo: VaR/CVaR 风控 + 止损策略 + 黑名单
=========================================
独立展示风控模块功能，不依赖完整回测流程。

用法: python3 examples/demo_risk.py
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quant2026.risk.var import VaRCalculator
from quant2026.risk.stop_loss import StopLossManager
from quant2026.risk.manager import RiskManager
from quant2026.types import PortfolioTarget

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")


def main() -> None:
    np.random.seed(42)

    # ── 模拟数据 ──────────────────────────────────────────────
    n_days = 250
    stocks = ["600519", "000858", "601318", "600036", "000333"]
    returns_matrix = pd.DataFrame(
        np.random.normal(0.0005, 0.02, (n_days, len(stocks))),
        columns=stocks,
        index=pd.date_range("2024-01-01", periods=n_days, freq="B"),
    )
    equity_curve = pd.Series(
        1_000_000 * np.cumprod(1 + returns_matrix.mean(axis=1)),
        index=returns_matrix.index,
    )

    # ================================================================
    # 1. VaR / CVaR
    # ================================================================
    print("=" * 55)
    print("📊 VaR / CVaR 风险报告")
    print("=" * 55)

    var_calc = VaRCalculator(confidence=0.95)
    port_returns = returns_matrix.mean(axis=1)

    print(f"  历史法 VaR(95%):  {var_calc.historical_var(port_returns):.4f}")
    print(f"  参数法 VaR(95%):  {var_calc.parametric_var(port_returns):.4f}")
    print(f"  CVaR(95%):        {var_calc.cvar(port_returns):.4f}")

    weights = pd.Series([0.3, 0.25, 0.2, 0.15, 0.1], index=stocks)
    pvar = var_calc.portfolio_var(weights, returns_matrix)
    print(f"\n  组合 VaR(hist):   {pvar['var_hist']:.4f}")
    print(f"  组合 CVaR:        {pvar['cvar']:.4f}")
    print("  Component VaR:")
    for s, v in pvar["component_var"].items():
        print(f"    {s}: {v:.6f}")

    # ================================================================
    # 2. 止损策略
    # ================================================================
    print("\n" + "=" * 55)
    print("🛑 止损策略检查")
    print("=" * 55)

    sl = StopLossManager(
        stock_stop_loss=-0.10,
        portfolio_stop_loss=-0.15,
        trailing_stop=-0.08,
    )

    # 模拟持仓收益率
    holdings = {
        "600519": 0.05,    # 茅台 +5%
        "000858": -0.12,   # 五粮液 -12% → 触发止损
        "601318": -0.03,   # 平安 -3%
        "600036": -0.11,   # 招商 -11% → 触发止损
        "000333": 0.08,    # 美的 +8%
    }

    stock_stops = sl.check_stock_stop_loss(holdings)
    print(f"  个股止损(-10%): {stock_stops or '无'}")

    port_stop = sl.check_portfolio_stop_loss(equity_curve)
    print(f"  组合止损(-15%): {'触发!' if port_stop else '正常'}")

    # ================================================================
    # 3. 黑名单
    # ================================================================
    print("\n" + "=" * 55)
    print("🚫 黑名单过滤")
    print("=" * 55)

    target = PortfolioTarget(
        date=date(2024, 6, 1),
        weights=pd.Series({s: 0.2 for s in stocks}),
    )
    print(f"  原始权重: {dict(target.weights)}")

    blacklist = {"000858", "601318"}
    result = sl.apply_blacklist(target, blacklist=blacklist)
    print(f"  黑名单:   {blacklist}")
    print(f"  过滤后:   {dict(result.weights.round(4))}")
    print(f"  权重之和: {result.weights.sum():.4f}")

    # ================================================================
    # 4. RiskManager 综合报告
    # ================================================================
    print("\n" + "=" * 55)
    print("📋 RiskManager 综合风控报告")
    print("=" * 55)

    rm = RiskManager()
    metrics = rm.check_post_trade(equity_curve)

    print(f"  最大回撤:     {metrics.max_drawdown:.2%}")
    print(f"  年化波动率:   {metrics.volatility:.2%}")
    print(f"  夏普比率:     {metrics.sharpe_ratio:.2f}")
    print(f"  VaR(95%):     {metrics.var_95:.4f}")
    print(f"  CVaR(95%):    {metrics.cvar_95:.4f}")
    if metrics.warnings:
        print("  ⚠️ 警告:")
        for w in metrics.warnings:
            print(f"    - {w}")
    print("=" * 55)


if __name__ == "__main__":
    main()
