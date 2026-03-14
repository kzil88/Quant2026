"""Quant2026 CLI 命令行入口。

用 click 提供回测、优化、walk-forward、配置验证等命令。
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger


# ── 版本 ────────────────────────────────────────────────────────

def _get_version() -> str:
    """Read version from package metadata or fallback."""
    try:
        from importlib.metadata import version
        return version("quant2026")
    except Exception:
        return "0.1.0"


# ── 工具函数 ────────────────────────────────────────────────────

_FREQ_MAP = {"daily": 1, "weekly": 5, "monthly": 20}


def _setup_logging(verbose: bool) -> None:
    """Configure loguru logging level via unified setup."""
    from quant2026.logging import setup_logging
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, log_dir="logs")


def _parse_overrides(overrides: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``-o key=value`` pairs into a dict.

    Attempts to cast numeric values automatically.
    """
    result: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise click.BadParameter(f"override 格式错误: {item!r}，应为 key=value")
        key, val_str = item.split("=", 1)
        # Try numeric cast
        try:
            value: Any = int(val_str)
        except ValueError:
            try:
                value = float(val_str)
            except ValueError:
                # bool
                if val_str.lower() in ("true", "false"):
                    value = val_str.lower() == "true"
                else:
                    value = val_str
        result[key] = value
    return result


def _echo_success(msg: str) -> None:
    click.echo(click.style(msg, fg="green"))


def _echo_error(msg: str) -> None:
    click.echo(click.style(f"✗ {msg}", fg="red"), err=True)


def _echo_info(msg: str) -> None:
    click.echo(click.style(msg, fg="cyan"))


# ── CLI Group ───────────────────────────────────────────────────


@click.group()
@click.version_option(version=_get_version(), prog_name="quant2026")
def cli() -> None:
    """Quant2026 - A股量化投资框架"""
    pass


# ── backtest ────────────────────────────────────────────────────


@cli.command()
@click.option("--config", "-c", default="config/default.yaml", help="配置文件路径")
@click.option("--override", "-o", multiple=True, help="覆盖参数 key=value")
@click.option("--output", default=None, help="输出目录")
@click.option("--verbose", "-v", is_flag=True, help="详细日志")
def backtest(config: str, override: tuple[str, ...], output: str | None, verbose: bool) -> None:
    """运行回测"""
    _setup_logging(verbose)

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

    # 1. 加载配置
    try:
        overrides = _parse_overrides(override)
        if overrides:
            cfg = ConfigLoader.load_with_overrides(config, overrides)
        else:
            cfg = ConfigLoader.load(config)
    except Exception as e:
        _echo_error(f"配置加载失败: {e}")
        raise SystemExit(1)

    errors = ConfigLoader.validate(cfg)
    if errors:
        _echo_error("配置验证失败:")
        for e in errors:
            _echo_error(f"  • {e}")
        raise SystemExit(1)

    # Override output dir
    if output:
        cfg.output.dir = output

    _echo_info(f"📋 配置: {config}  策略={[s.name for s in cfg.strategies]}")

    # 2. ComponentFactory 创建组件
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

    # 3. 拉数据
    _echo_info("📥 Step 1/7: 获取行情数据")
    provider = CachedProvider(AkShareProvider())
    try:
        data = provider.get_daily_quotes(STOCK_POOL, START_DATE, END_DATE)
    except Exception as e:
        _echo_error(f"数据获取失败: {e}")
        raise SystemExit(1)
    if data.empty:
        _echo_error("未获取到任何数据")
        raise SystemExit(1)
    logger.info(f"获取到 {data['stock_code'].nunique()} 只股票, {len(data)} 条记录")

    # 数据清洗
    _echo_info("🧹 Step 2/7: 数据清洗")
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
        _echo_error("有效股票太少（< 5）")
        raise SystemExit(1)

    # 财务数据
    _echo_info("💰 Step 2b/7: 获取财务数据")
    financial_data = None
    try:
        financial_data = provider.get_financial_data(list(valid_stocks), END_DATE)
        if financial_data is None or financial_data.empty:
            financial_data = None
    except Exception:
        financial_data = None

    # 4. 因子计算
    _echo_info("📊 Step 3/7: 因子计算")
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

    # 5 & 6. 策略打分 + 组合优化
    _echo_info("🎯 Step 4-5/7: 策略打分 & 组合优化")
    targets: dict[date, PortfolioTarget] = {}
    current_weights: pd.Series | None = None

    with click.progressbar(
        rebalance_dates_str,
        label="调仓进度",
        show_pos=True,
    ) as bar:
        for dt_str in bar:
            target_date = date.fromisoformat(dt_str)

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

    if not targets:
        _echo_error("未生成任何调仓目标")
        raise SystemExit(1)

    # 风控
    _echo_info("🛡️ Step 5/7: 风控检查")
    for dt, tgt in targets.items():
        targets[dt] = risk_mgr.check_pre_trade(tgt)

    # 7. 回测
    _echo_info("⚡ Step 6/7: 回测")
    engine = BacktestEngine(bt_config)
    bt_data = data.copy()
    bt_data["date"] = pd.to_datetime(bt_data["date"]).dt.date
    result = engine.run(bt_data, targets)

    if result.equity_curve.empty:
        _echo_error("回测未产生结果")
        raise SystemExit(1)

    risk_mgr.check_post_trade(result.equity_curve)

    # 8. 生成报告
    _echo_info("📝 Step 7/7: 生成报告")
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
            logger.info(f"HTML 报告: {OUTPUT_DIR / 'report.html'}")
        except Exception as e:
            logger.warning(f"HTML 报告失败: {e}")

    if cfg.output.equity_curve:
        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
            equity = result.equity_curve / result.equity_curve.iloc[0]
            axes[0].plot(equity.index, equity.values, "b-", linewidth=1.5, label="Strategy NAV")
            axes[0].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
            axes[0].set_title("Quant2026 Backtest", fontsize=14)
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
        except Exception as e:
            logger.warning(f"净值曲线生成失败: {e}")

    # 9. 打印摘要
    metrics = result.metrics
    click.echo()
    click.echo(click.style("=" * 50, fg="yellow"))
    click.echo(click.style(f"📊 回测绩效报告", fg="yellow", bold=True))
    click.echo(click.style("=" * 50, fg="yellow"))
    click.echo(f"  配置文件:   {config}")
    click.echo(f"  回测区间:   {START_DATE} ~ {END_DATE}")
    click.echo(f"  股票池:     {len(STOCK_POOL)} 只")
    click.echo(f"  调仓次数:   {len(targets)}")
    click.echo(click.style("-" * 50, fg="yellow"))
    _LABELS = {
        "total_return": "总收益率",
        "annual_return": "年化收益率",
        "volatility": "年化波动率",
        "sharpe_ratio": "夏普比率",
        "max_drawdown": "最大回撤",
        "trade_count": "交易次数",
        "avg_turnover": "平均换手率",
    }
    for k, v in metrics.items():
        label = _LABELS.get(k, k)
        click.echo(f"  {label:<12} {v}")
    click.echo(f"  终值:       ¥{result.equity_curve.iloc[-1]:,.0f}")
    click.echo(click.style("=" * 50, fg="yellow"))
    _echo_success(f"✓ 报告输出目录: {OUTPUT_DIR}")


# ── optimize ────────────────────────────────────────────────────


@cli.command()
@click.option("--config", "-c", default="config/default.yaml", help="配置文件路径")
@click.option("--strategy", "-s", required=True, help="要优化的策略名称")
@click.option("--method", default="grid", type=click.Choice(["grid", "random", "bayesian"]))
@click.option("--n-iter", default=50, help="搜索次数（random/bayesian）")
@click.option("--output", default=None, help="输出目录")
@click.option("--verbose", "-v", is_flag=True, help="详细日志")
def optimize(config: str, strategy: str, method: str, n_iter: int, output: str | None, verbose: bool) -> None:
    """策略参数优化"""
    _setup_logging(verbose)

    from quant2026.config import ConfigLoader

    try:
        cfg = ConfigLoader.load(config)
    except Exception as e:
        _echo_error(f"配置加载失败: {e}")
        raise SystemExit(1)

    errors = ConfigLoader.validate(cfg)
    if errors:
        _echo_error("配置验证失败:")
        for e in errors:
            _echo_error(f"  • {e}")
        raise SystemExit(1)

    # Find the strategy config
    strat_cfg = None
    for s in cfg.strategies:
        if s.name == strategy:
            strat_cfg = s
            break
    if strat_cfg is None:
        _echo_error(f"策略 {strategy!r} 不在配置中。可用: {[s.name for s in cfg.strategies]}")
        raise SystemExit(1)

    _echo_info(f"🔍 优化策略: {strategy} method={method} n_iter={n_iter}")
    _echo_info("⚠️  参数优化需要定义 ParamSpace，请参考 quant2026.optimization.param_optimizer")
    _echo_success("✓ 优化框架就绪（需要用户代码定义搜索空间和目标函数）")


# ── walkforward ─────────────────────────────────────────────────


@cli.command()
@click.option("--config", "-c", default="config/default.yaml", help="配置文件路径")
@click.option("--train-months", default=6, help="训练窗口（月）")
@click.option("--test-months", default=2, help="测试窗口（月）")
@click.option("--output", default=None, help="输出目录")
@click.option("--verbose", "-v", is_flag=True, help="详细日志")
def walkforward(config: str, train_months: int, test_months: int, output: str | None, verbose: bool) -> None:
    """Walk-Forward 滚动回测"""
    _setup_logging(verbose)

    from quant2026.config import ConfigLoader
    from quant2026.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig

    try:
        cfg = ConfigLoader.load(config)
    except Exception as e:
        _echo_error(f"配置加载失败: {e}")
        raise SystemExit(1)

    errors = ConfigLoader.validate(cfg)
    if errors:
        _echo_error("配置验证失败:")
        for e in errors:
            _echo_error(f"  • {e}")
        raise SystemExit(1)

    if output:
        cfg.output.dir = output

    _echo_info(f"🔄 Walk-Forward: train={train_months}m test={test_months}m")
    _echo_info("⚠️  Walk-Forward 需要定义 pipeline_fn，请参考 quant2026.backtest.walk_forward")
    _echo_success("✓ Walk-Forward 框架就绪")


# ── validate ────────────────────────────────────────────────────


@cli.command()
@click.option("--config", "-c", default="config/default.yaml", help="配置文件路径")
@click.option("--verbose", "-v", is_flag=True, help="详细日志")
def validate(config: str, verbose: bool) -> None:
    """验证配置文件"""
    _setup_logging(verbose)

    from quant2026.config import ConfigLoader

    try:
        cfg = ConfigLoader.load(config)
    except Exception as e:
        _echo_error(f"配置加载失败: {e}")
        raise SystemExit(1)

    errors = ConfigLoader.validate(cfg)
    if errors:
        _echo_error(f"配置验证发现 {len(errors)} 个错误:")
        for e in errors:
            _echo_error(f"  • {e}")
        raise SystemExit(1)
    else:
        _echo_success("✓ 配置有效")
        click.echo(f"  策略: {[s.name for s in cfg.strategies]}")
        click.echo(f"  区间: {cfg.data.start_date} ~ {cfg.data.end_date}")
        click.echo(f"  股票池: {len(cfg.data.stock_pool)} 只")


# ── init ────────────────────────────────────────────────────────


@cli.command()
@click.option("--output", "-o", default="config/custom.yaml", help="输出文件路径")
def init(output: str) -> None:
    """生成默认配置文件"""
    from quant2026.config import ConfigLoader, Quant2026Config

    cfg = Quant2026Config()
    out_path = Path(output)

    if out_path.exists():
        if not click.confirm(f"文件 {out_path} 已存在，是否覆盖?"):
            _echo_info("已取消")
            return

    ConfigLoader.to_yaml(cfg, out_path)
    _echo_success(f"✓ 默认配置已生成: {out_path}")
    click.echo("  编辑此文件后运行: quant2026 backtest -c " + str(out_path))


# ── main ────────────────────────────────────────────────────────


def main() -> None:
    """CLI 入口函数。"""
    cli()


if __name__ == "__main__":
    main()
