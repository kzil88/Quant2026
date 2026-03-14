"""Quant2026 YAML 配置加载器。

提供 dataclass 配置模型和 ConfigLoader 工具类，
支持从 YAML 加载、环境变量替换、点号路径覆盖、验证和导出。
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


# ── 配置 dataclasses ────────────────────────────────────────────


@dataclass
class DataConfig:
    """数据源配置。"""

    provider: str = "akshare"
    cache_enabled: bool = True
    cache_dir: str = ".cache"
    cache_ttl_hours: int = 24
    stock_pool: list[str] = field(default_factory=list)
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"


@dataclass
class BacktestConfigYaml:
    """回测参数配置（YAML 层，与 engine.BacktestConfig 区分）。"""

    initial_capital: float = 1_000_000
    commission: float = 0.0003
    stamp_tax: float = 0.001
    slippage: float = 0.001
    rebalance_frequency: str = "monthly"
    benchmark: str = "000300"


@dataclass
class StrategyConfig:
    """单个策略配置。"""

    name: str = ""
    type: str = ""
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioConfig:
    """组合优化配置。"""

    method: str = "markowitz"
    markowitz_risk_free_rate: float = 0.025
    markowitz_max_single_weight: float = 0.10
    max_turnover: float = 0.3
    turnover_penalty_weight: float = 0.01


@dataclass
class RiskConfig:
    """风控配置。"""

    max_position_size: float = 0.10
    max_sector_exposure: float = 0.30
    stock_stop_loss: float = -0.10
    portfolio_stop_loss: float = -0.15
    trailing_stop: float = -0.08
    blacklist: list[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """日志配置。"""

    level: str = "INFO"
    log_dir: str = "logs"
    rotation: str = "10 MB"
    retention: str = "30 days"


@dataclass
class OutputConfig:
    """输出配置。"""

    dir: str = "examples/output"
    report: bool = True
    equity_curve: bool = True


@dataclass
class Quant2026Config:
    """顶层配置，聚合所有子配置。"""

    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfigYaml = field(default_factory=BacktestConfigYaml)
    strategies: list[StrategyConfig] = field(default_factory=list)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    factors: list[str] = field(default_factory=list)
    output: OutputConfig = field(default_factory=OutputConfig)


# ── YAML ↔ dataclass 映射 ──────────────────────────────────────

_VALID_STRATEGY_TYPES = {
    "MultiFactorStrategy",
    "MeanReversionStrategy",
    "StatArbStrategy",
    "EventDrivenStrategy",
    "MLStrategy",
}

_REBALANCE_FREQUENCIES = {"daily", "weekly", "monthly"}

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _parse_data(raw: dict) -> DataConfig:
    cache = raw.get("cache", {})
    return DataConfig(
        provider=raw.get("provider", "akshare"),
        cache_enabled=cache.get("enabled", True),
        cache_dir=cache.get("cache_dir", ".cache"),
        cache_ttl_hours=cache.get("ttl_hours", 24),
        stock_pool=raw.get("stock_pool", []),
        start_date=raw.get("start_date", "2024-01-01"),
        end_date=raw.get("end_date", "2024-12-31"),
    )


def _parse_backtest(raw: dict) -> BacktestConfigYaml:
    return BacktestConfigYaml(
        initial_capital=raw.get("initial_capital", 1_000_000),
        commission=raw.get("commission", 0.0003),
        stamp_tax=raw.get("stamp_tax", 0.001),
        slippage=raw.get("slippage", 0.001),
        rebalance_frequency=raw.get("rebalance_frequency", "monthly"),
        benchmark=raw.get("benchmark", "000300"),
    )


def _parse_strategies(raw: list) -> list[StrategyConfig]:
    result = []
    for item in raw:
        result.append(
            StrategyConfig(
                name=item.get("name", ""),
                type=item.get("type", ""),
                weight=item.get("weight", 1.0),
                params=item.get("params", {}),
            )
        )
    return result


def _parse_portfolio(raw: dict) -> PortfolioConfig:
    mk = raw.get("markowitz", {})
    to = raw.get("turnover", {})
    return PortfolioConfig(
        method=raw.get("method", "markowitz"),
        markowitz_risk_free_rate=mk.get("risk_free_rate", 0.025),
        markowitz_max_single_weight=mk.get("max_single_weight", 0.10),
        max_turnover=to.get("max_turnover", 0.3),
        turnover_penalty_weight=to.get("penalty_weight", 0.01),
    )


def _parse_risk(raw: dict) -> RiskConfig:
    sl = raw.get("stop_loss", {})
    return RiskConfig(
        max_position_size=raw.get("max_position_size", 0.10),
        max_sector_exposure=raw.get("max_sector_exposure", 0.30),
        stock_stop_loss=sl.get("stock_stop_loss", -0.10),
        portfolio_stop_loss=sl.get("portfolio_stop_loss", -0.15),
        trailing_stop=sl.get("trailing_stop", -0.08),
        blacklist=raw.get("blacklist", []),
    )


def _parse_output(raw: dict) -> OutputConfig:
    return OutputConfig(
        dir=raw.get("dir", "examples/output"),
        report=raw.get("report", True),
        equity_curve=raw.get("equity_curve", True),
    )


def _parse_logging(raw: dict) -> LoggingConfig:
    return LoggingConfig(
        level=raw.get("level", "INFO"),
        log_dir=raw.get("log_dir", "logs"),
        rotation=raw.get("rotation", "10 MB"),
        retention=raw.get("retention", "30 days"),
    )


def _raw_to_config(raw: dict) -> Quant2026Config:
    """将原始 YAML dict 转为 Quant2026Config。"""
    return Quant2026Config(
        data=_parse_data(raw.get("data", {})),
        backtest=_parse_backtest(raw.get("backtest", {})),
        strategies=_parse_strategies(raw.get("strategies", [])),
        portfolio=_parse_portfolio(raw.get("portfolio", {})),
        risk=_parse_risk(raw.get("risk", {})),
        logging=_parse_logging(raw.get("logging", {})),
        factors=raw.get("factors", []),
        output=_parse_output(raw.get("output", {})),
    )


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """按点号路径设置嵌套 dict 的值。"""
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _env_substitute(text: str) -> str:
    """替换 ${ENV_VAR} 为环境变量值。"""

    def _replacer(m: re.Match) -> str:
        var = m.group(1)
        val = os.environ.get(var, "")
        if not val:
            logger.warning(f"环境变量 ${{{var}}} 未设置，替换为空字符串")
        return val

    return _ENV_PATTERN.sub(_replacer, text)


# ── ConfigLoader ────────────────────────────────────────────────


class ConfigLoader:
    """YAML 配置加载器。"""

    @staticmethod
    def load(path: str | Path) -> Quant2026Config:
        """从 YAML 文件加载配置。

        Args:
            path: YAML 文件路径。

        Returns:
            Quant2026Config 实例。
        """
        path = Path(path)
        logger.info(f"加载配置: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return _raw_to_config(raw)

    @staticmethod
    def load_with_overrides(
        path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> Quant2026Config:
        """加载配置并用 overrides 字典覆盖。

        Args:
            path: YAML 文件路径。
            overrides: 点号路径覆盖，如 ``{"backtest.initial_capital": 2000000}``。

        Returns:
            Quant2026Config 实例。
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if overrides:
            for dotted_key, value in overrides.items():
                _set_nested(raw, dotted_key, value)
                logger.debug(f"覆盖 {dotted_key} = {value}")
        return _raw_to_config(raw)

    @staticmethod
    def from_env(path: str | Path) -> Quant2026Config:
        """加载配置，支持环境变量替换。

        YAML 中 ``${ENV_VAR}`` 会被替换为对应环境变量值。

        Args:
            path: YAML 文件路径。

        Returns:
            Quant2026Config 实例。
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        text = _env_substitute(text)
        raw = yaml.safe_load(text) or {}
        return _raw_to_config(raw)

    @staticmethod
    def to_yaml(config: Quant2026Config, path: str | Path) -> None:
        """将配置导出为 YAML 文件。

        Args:
            config: 配置实例。
            path: 输出文件路径。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 重构为与 YAML 结构一致的嵌套 dict
        d = _config_to_raw(config)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        logger.info(f"配置已导出: {path}")

    @staticmethod
    def validate(config: Quant2026Config) -> list[str]:
        """验证配置，返回错误列表（空列表 = 有效）。

        检查项：
        - 日期格式合法
        - 策略类型存在
        - 策略权重和为正
        - 数值范围合理
        """
        errors: list[str] = []

        # 日期格式
        for label, ds in [("start_date", config.data.start_date), ("end_date", config.data.end_date)]:
            try:
                datetime.strptime(ds, "%Y-%m-%d")
            except ValueError:
                errors.append(f"data.{label} 格式无效: {ds!r}，应为 YYYY-MM-DD")

        if config.data.start_date >= config.data.end_date:
            errors.append("data.start_date 必须早于 data.end_date")

        # stock_pool
        if not config.data.stock_pool:
            errors.append("data.stock_pool 不能为空")

        # 策略
        if not config.strategies:
            errors.append("strategies 不能为空")
        else:
            total_weight = sum(s.weight for s in config.strategies)
            if total_weight <= 0:
                errors.append(f"strategies 权重之和必须为正，当前为 {total_weight}")
            for s in config.strategies:
                if s.type not in _VALID_STRATEGY_TYPES:
                    errors.append(f"策略 {s.name!r} 的 type {s.type!r} 不在支持列表: {_VALID_STRATEGY_TYPES}")
                if not s.name:
                    errors.append("策略 name 不能为空")

        # rebalance_frequency
        if config.backtest.rebalance_frequency not in _REBALANCE_FREQUENCIES:
            errors.append(
                f"backtest.rebalance_frequency {config.backtest.rebalance_frequency!r} "
                f"不在 {_REBALANCE_FREQUENCIES}"
            )

        # 数值范围
        if config.backtest.initial_capital <= 0:
            errors.append("backtest.initial_capital 必须 > 0")
        if not (0 <= config.backtest.commission <= 0.01):
            errors.append(f"backtest.commission 超出合理范围 [0, 0.01]: {config.backtest.commission}")
        if not (0 < config.portfolio.max_turnover <= 1):
            errors.append(f"portfolio.max_turnover 应在 (0, 1]: {config.portfolio.max_turnover}")
        if not (0 < config.risk.max_position_size <= 1):
            errors.append(f"risk.max_position_size 应在 (0, 1]: {config.risk.max_position_size}")

        # portfolio method
        if config.portfolio.method not in {"equal", "markowitz", "risk_parity"}:
            errors.append(f"portfolio.method {config.portfolio.method!r} 不支持")

        return errors


def _config_to_raw(config: Quant2026Config) -> dict:
    """将 Quant2026Config 转回与 YAML 结构一致的嵌套 dict。"""
    return {
        "data": {
            "provider": config.data.provider,
            "cache": {
                "enabled": config.data.cache_enabled,
                "cache_dir": config.data.cache_dir,
                "ttl_hours": config.data.cache_ttl_hours,
            },
            "stock_pool": config.data.stock_pool,
            "start_date": config.data.start_date,
            "end_date": config.data.end_date,
        },
        "backtest": asdict(config.backtest),
        "strategies": [asdict(s) for s in config.strategies],
        "portfolio": {
            "method": config.portfolio.method,
            "markowitz": {
                "risk_free_rate": config.portfolio.markowitz_risk_free_rate,
                "max_single_weight": config.portfolio.markowitz_max_single_weight,
            },
            "turnover": {
                "max_turnover": config.portfolio.max_turnover,
                "penalty_weight": config.portfolio.turnover_penalty_weight,
            },
        },
        "risk": {
            "max_position_size": config.risk.max_position_size,
            "max_sector_exposure": config.risk.max_sector_exposure,
            "stop_loss": {
                "stock_stop_loss": config.risk.stock_stop_loss,
                "portfolio_stop_loss": config.risk.portfolio_stop_loss,
                "trailing_stop": config.risk.trailing_stop,
            },
            "blacklist": config.risk.blacklist,
        },
        "factors": config.factors,
        "logging": asdict(config.logging),
        "output": asdict(config.output),
    }
