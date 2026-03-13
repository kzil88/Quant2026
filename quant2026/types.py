"""Core types shared across all modules."""

from dataclasses import dataclass, field
from datetime import date
from enum import Enum, auto
from typing import Any

import pandas as pd


class Market(Enum):
    """Supported markets."""
    SH = "sh"   # 上交所
    SZ = "sz"   # 深交所
    BJ = "bj"   # 北交所


class Signal(Enum):
    """Trading signal."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()


@dataclass
class StrategyResult:
    """Standardized output from any strategy."""
    name: str
    date: date
    scores: pd.Series          # stock_code -> score (higher = more bullish)
    signals: pd.Series | None = None  # stock_code -> Signal
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioTarget:
    """Target portfolio from portfolio optimizer."""
    date: date
    weights: pd.Series         # stock_code -> target weight [0, 1]
    strategy_weights: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeOrder:
    """A single trade order."""
    stock_code: str
    signal: Signal
    target_weight: float
    current_weight: float = 0.0
    reason: str = ""


@dataclass
class RiskMetrics:
    """Risk assessment snapshot."""
    date: date
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    exposure: dict[str, float] = field(default_factory=dict)  # sector/factor exposure
    warnings: list[str] = field(default_factory=list)
    var_95: float | None = None
    cvar_95: float | None = None
    stop_loss_triggers: list[dict[str, Any]] = field(default_factory=list)
