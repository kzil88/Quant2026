"""Statistical Arbitrage / Pairs Trading strategy."""

from .cointegration import CointegrationAnalyzer
from .strategy import StatArbStrategy

__all__ = ["CointegrationAnalyzer", "StatArbStrategy"]
