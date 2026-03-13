"""Basic sanity tests."""
from quant2026.types import Signal, StrategyResult
from datetime import date
import pandas as pd


def test_signal_enum():
    assert Signal.BUY != Signal.SELL


def test_strategy_result():
    r = StrategyResult(
        name="test",
        date=date.today(),
        scores=pd.Series({"000001": 0.8, "600519": 0.5}),
    )
    assert r.scores["000001"] > r.scores["600519"]
