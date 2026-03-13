"""Strategy layer: pluggable strategies with unified interface."""

from .ml_model import MLStrategy, MLTrainer
from .mean_reversion import MeanReversionStrategy
from .stat_arb import CointegrationAnalyzer, StatArbStrategy
from .event_driven import EventCollector, EventDrivenStrategy

__all__ = [
    "MLStrategy", "MLTrainer",
    "MeanReversionStrategy",
    "CointegrationAnalyzer", "StatArbStrategy",
    "EventCollector", "EventDrivenStrategy",
]
