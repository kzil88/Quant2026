"""Backtest layer: historical simulation with A-share constraints."""

from .engine import BacktestEngine, BacktestConfig, BacktestResult
from .attribution import PerformanceAttribution
from .report import BacktestReporter, ExtendedMetrics
from .walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
)

__all__ = [
    "BacktestEngine", "BacktestConfig", "BacktestResult",
    "BacktestReporter", "ExtendedMetrics",
    "PerformanceAttribution",
    "WalkForwardAnalyzer", "WalkForwardConfig", "WalkForwardResult", "WalkForwardWindow",
]
