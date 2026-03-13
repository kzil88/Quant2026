"""Momentum / trend-following strategy (动量/趋势跟踪)."""

from datetime import date

import numpy as np
import pandas as pd

from quant2026.types import StrategyResult, Signal
from quant2026.strategy.base import Strategy


class MomentumStrategy(Strategy):
    """Dual moving average crossover + momentum scoring."""

    def __init__(self, fast_window: int = 5, slow_window: int = 20):
        self._fast = fast_window
        self._slow = slow_window

    @property
    def name(self) -> str:
        return f"momentum_{self._fast}_{self._slow}"

    def generate(
        self,
        data: pd.DataFrame,
        factor_matrix: pd.DataFrame | None,
        target_date: date,
    ) -> StrategyResult:
        df = data[data["date"] <= str(target_date)].copy()

        scores = {}
        signals = {}

        for code, group in df.groupby("stock_code"):
            if len(group) < self._slow + 1:
                continue
            close = group.set_index("date")["close"].sort_index()
            ma_fast = close.rolling(self._fast).mean()
            ma_slow = close.rolling(self._slow).mean()

            latest_fast = ma_fast.iloc[-1]
            latest_slow = ma_slow.iloc[-1]

            # Score: distance between fast and slow MA, normalized by price
            score = (latest_fast - latest_slow) / latest_slow if latest_slow > 0 else 0
            scores[code] = score

            # Signal
            prev_fast = ma_fast.iloc[-2] if len(ma_fast) > 1 else latest_fast
            prev_slow = ma_slow.iloc[-2] if len(ma_slow) > 1 else latest_slow

            if prev_fast <= prev_slow and latest_fast > latest_slow:
                signals[code] = Signal.BUY
            elif prev_fast >= prev_slow and latest_fast < latest_slow:
                signals[code] = Signal.SELL
            else:
                signals[code] = Signal.HOLD

        return StrategyResult(
            name=self.name,
            date=target_date,
            scores=pd.Series(scores).sort_values(ascending=False),
            signals=pd.Series(signals),
        )
