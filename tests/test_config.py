"""Tests for quant2026.config and quant2026.factory."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

# Ensure project root on path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant2026.config import (
    ConfigLoader,
    Quant2026Config,
    DataConfig,
    BacktestConfigYaml,
    StrategyConfig,
    PortfolioConfig,
    RiskConfig,
    OutputConfig,
)
from quant2026.factory import ComponentFactory
from quant2026.strategy.multi_factor.strategy import MultiFactorStrategy
from quant2026.strategy.mean_reversion.strategy import MeanReversionStrategy
from quant2026.portfolio.optimizer import PortfolioOptimizer
from quant2026.risk.manager import RiskManager
from quant2026.backtest.engine import BacktestConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


# ── 加载默认配置 ───────────────────────────────────────────────


class TestConfigLoader:
    """ConfigLoader 基本功能测试。"""

    def test_load_default(self):
        """加载 default.yaml 成功。"""
        cfg = ConfigLoader.load(DEFAULT_CONFIG)
        assert isinstance(cfg, Quant2026Config)
        assert len(cfg.data.stock_pool) == 30
        assert cfg.backtest.initial_capital == 1_000_000
        assert len(cfg.strategies) == 2
        assert cfg.portfolio.method == "markowitz"

    def test_load_validates_ok(self):
        """默认配置通过验证。"""
        cfg = ConfigLoader.load(DEFAULT_CONFIG)
        errors = ConfigLoader.validate(cfg)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_load_with_overrides(self):
        """覆盖参数生效。"""
        cfg = ConfigLoader.load_with_overrides(
            DEFAULT_CONFIG,
            overrides={
                "backtest.initial_capital": 2_000_000,
                "backtest.rebalance_frequency": "weekly",
            },
        )
        assert cfg.backtest.initial_capital == 2_000_000
        assert cfg.backtest.rebalance_frequency == "weekly"

    def test_env_substitution(self, tmp_path: Path):
        """环境变量替换。"""
        os.environ["Q2026_START"] = "2023-06-01"
        os.environ["Q2026_END"] = "2023-12-31"
        try:
            yaml_content = {
                "data": {
                    "provider": "akshare",
                    "stock_pool": ["600519"],
                    "start_date": "${Q2026_START}",
                    "end_date": "${Q2026_END}",
                },
                "strategies": [
                    {"name": "mf", "type": "MultiFactorStrategy", "weight": 1.0, "params": {}},
                ],
            }
            p = tmp_path / "env_test.yaml"
            with open(p, "w") as f:
                yaml.dump(yaml_content, f)
            # Write raw text with ${} placeholders
            p.write_text(
                "data:\n"
                "  provider: akshare\n"
                "  stock_pool: ['600519']\n"
                "  start_date: '${Q2026_START}'\n"
                "  end_date: '${Q2026_END}'\n"
                "strategies:\n"
                "  - name: mf\n"
                "    type: MultiFactorStrategy\n"
                "    weight: 1.0\n"
                "    params: {}\n"
            )
            cfg = ConfigLoader.from_env(p)
            assert cfg.data.start_date == "2023-06-01"
            assert cfg.data.end_date == "2023-12-31"
        finally:
            del os.environ["Q2026_START"]
            del os.environ["Q2026_END"]

    def test_to_yaml_roundtrip(self, tmp_path: Path):
        """导出再加载，配置一致。"""
        cfg1 = ConfigLoader.load(DEFAULT_CONFIG)
        out = tmp_path / "roundtrip.yaml"
        ConfigLoader.to_yaml(cfg1, out)
        cfg2 = ConfigLoader.load(out)
        assert cfg2.backtest.initial_capital == cfg1.backtest.initial_capital
        assert len(cfg2.strategies) == len(cfg1.strategies)
        assert cfg2.portfolio.method == cfg1.portfolio.method


# ── 验证测试 ───────────────────────────────────────────────────


class TestValidation:
    """配置验证测试。"""

    def test_empty_stock_pool(self):
        """空 stock_pool 报错。"""
        cfg = Quant2026Config(
            strategies=[StrategyConfig(name="mf", type="MultiFactorStrategy", weight=1.0)],
        )
        errors = ConfigLoader.validate(cfg)
        assert any("stock_pool" in e for e in errors)

    def test_empty_strategies(self):
        """空 strategies 报错。"""
        cfg = Quant2026Config(
            data=DataConfig(stock_pool=["600519"]),
        )
        errors = ConfigLoader.validate(cfg)
        assert any("strategies" in e for e in errors)

    def test_invalid_strategy_type(self):
        """无效策略类型报错。"""
        cfg = Quant2026Config(
            data=DataConfig(stock_pool=["600519"]),
            strategies=[StrategyConfig(name="bad", type="NonExistentStrategy", weight=1.0)],
        )
        errors = ConfigLoader.validate(cfg)
        assert any("NonExistentStrategy" in e for e in errors)

    def test_bad_date_format(self):
        """日期格式错误。"""
        cfg = Quant2026Config(
            data=DataConfig(stock_pool=["600519"], start_date="2024/01/01"),
            strategies=[StrategyConfig(name="mf", type="MultiFactorStrategy", weight=1.0)],
        )
        errors = ConfigLoader.validate(cfg)
        assert any("start_date" in e for e in errors)

    def test_start_after_end(self):
        """start_date >= end_date 报错。"""
        cfg = Quant2026Config(
            data=DataConfig(stock_pool=["600519"], start_date="2025-01-01", end_date="2024-01-01"),
            strategies=[StrategyConfig(name="mf", type="MultiFactorStrategy", weight=1.0)],
        )
        errors = ConfigLoader.validate(cfg)
        assert any("start_date" in e for e in errors)

    def test_negative_weight_sum(self):
        """权重和 <= 0 报错。"""
        cfg = Quant2026Config(
            data=DataConfig(stock_pool=["600519"]),
            strategies=[StrategyConfig(name="mf", type="MultiFactorStrategy", weight=-1.0)],
        )
        errors = ConfigLoader.validate(cfg)
        assert any("权重" in e for e in errors)

    def test_invalid_rebalance_frequency(self):
        """无效 rebalance_frequency。"""
        cfg = Quant2026Config(
            data=DataConfig(stock_pool=["600519"]),
            strategies=[StrategyConfig(name="mf", type="MultiFactorStrategy", weight=1.0)],
            backtest=BacktestConfigYaml(rebalance_frequency="hourly"),
        )
        errors = ConfigLoader.validate(cfg)
        assert any("rebalance_frequency" in e for e in errors)


# ── 工厂测试 ───────────────────────────────────────────────────


class TestComponentFactory:
    """ComponentFactory 测试。"""

    @pytest.fixture
    def cfg(self) -> Quant2026Config:
        return ConfigLoader.load(DEFAULT_CONFIG)

    def test_create_strategies(self, cfg: Quant2026Config):
        """工厂创建正确类型的策略。"""
        results = ComponentFactory.create_strategies(cfg)
        assert len(results) == 2
        types = [type(s) for s, _w in results]
        assert MultiFactorStrategy in types
        assert MeanReversionStrategy in types
        # 检查权重
        weights = {type(s).__name__: w for s, w in results}
        assert weights["MultiFactorStrategy"] == 0.6
        assert weights["MeanReversionStrategy"] == 0.4

    def test_create_optimizer(self, cfg: Quant2026Config):
        """工厂创建 PortfolioOptimizer。"""
        opt = ComponentFactory.create_optimizer(cfg)
        assert isinstance(opt, PortfolioOptimizer)

    def test_create_risk_manager(self, cfg: Quant2026Config):
        """工厂创建 RiskManager。"""
        rm = ComponentFactory.create_risk_manager(cfg)
        assert isinstance(rm, RiskManager)
        assert rm.max_single_weight == cfg.risk.max_position_size

    def test_create_backtest_config(self, cfg: Quant2026Config):
        """工厂创建 BacktestConfig。"""
        bc = ComponentFactory.create_backtest_config(cfg)
        assert isinstance(bc, BacktestConfig)
        assert bc.initial_capital == cfg.backtest.initial_capital
        assert bc.commission_rate == cfg.backtest.commission


# ── 多配置文件测试 ─────────────────────────────────────────────


class TestMultipleConfigs:
    """加载 aggressive / conservative 配置。"""

    def test_load_aggressive(self):
        cfg = ConfigLoader.load(PROJECT_ROOT / "config" / "aggressive.yaml")
        assert cfg.risk.max_position_size == 0.20
        assert cfg.risk.stock_stop_loss == -0.15
        assert ConfigLoader.validate(cfg) == []

    def test_load_conservative(self):
        cfg = ConfigLoader.load(PROJECT_ROOT / "config" / "conservative.yaml")
        assert cfg.risk.max_position_size == 0.05
        assert cfg.portfolio.method == "risk_parity"
        assert ConfigLoader.validate(cfg) == []
