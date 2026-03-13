"""Mean Reversion Strategy (均值回归策略).

Scores stocks by how oversold they are relative to their moving average,
optionally incorporating RSI and Bollinger Band signals.
"""

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from quant2026.strategy.base import Strategy
from quant2026.types import Signal, StrategyResult


class MeanReversionStrategy(Strategy):
    """Mean reversion: buy oversold, avoid overbought.

    Core idea: stocks that deviate far below their moving average
    tend to revert. The further below, the higher the score.

    Args:
        window: Moving average lookback window.
        zscore_threshold: Z-score below which a stock is considered oversold.
        use_bollinger: Incorporate Bollinger Band position into scoring.
        use_rsi: Incorporate RSI oversold signal into scoring.
        rsi_period: RSI calculation period.
    """

    def __init__(
        self,
        window: int = 20,
        zscore_threshold: float = -1.0,
        use_bollinger: bool = True,
        use_rsi: bool = True,
        rsi_period: int = 14,
    ) -> None:
        self._window = window
        self._zscore_threshold = zscore_threshold
        self._use_bollinger = use_bollinger
        self._use_rsi = use_rsi
        self._rsi_period = rsi_period

    @property
    def name(self) -> str:
        return f"mean_reversion_{self._window}"

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> float:
        """Return the latest RSI value (0-100)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        latest = rsi.iloc[-1]
        return float(latest) if np.isfinite(latest) else 50.0

    # ── main entry point ─────────────────────────────────────────

    def generate(
        self,
        data: pd.DataFrame,
        factor_matrix: pd.DataFrame | None,
        target_date: date,
    ) -> StrategyResult:
        """Generate mean-reversion scores purely from OHLCV data.

        Higher score = more oversold = stronger buy candidate.
        """
        df = data[data["date"] <= str(target_date)].copy()
        min_rows = max(self._window, self._rsi_period) + 5

        scores: dict[str, float] = {}
        signals: dict[str, Signal] = {}

        for code, group in df.groupby("stock_code"):
            if len(group) < min_rows:
                continue

            close = group.set_index("date")["close"].sort_index().astype(float)

            # 1. Deviation from MA
            ma = close.rolling(self._window).mean()
            deviation = (close - ma) / ma  # series

            # 2. Z-score of deviation
            dev_mean = deviation.rolling(self._window, min_periods=self._window).mean()
            dev_std = deviation.rolling(self._window, min_periods=self._window).std()
            zscore = (deviation - dev_mean) / dev_std.replace(0, np.nan)

            latest_zscore = zscore.iloc[-1]
            if not np.isfinite(latest_zscore):
                continue

            # Base score: negative zscore → higher score (more oversold = better)
            score = -float(latest_zscore)

            # 3. Optional: Bollinger Band position bonus
            if self._use_bollinger:
                bb_mid = ma.iloc[-1]
                bb_std = close.rolling(self._window).std().iloc[-1]
                if np.isfinite(bb_std) and bb_std > 0:
                    bb_lower = bb_mid - 2 * bb_std
                    bb_upper = bb_mid + 2 * bb_std
                    bb_pos = (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower)
                    # Near lower band (bb_pos close to 0) → bonus up to +1
                    bb_bonus = max(0.0, 1.0 - float(bb_pos)) if np.isfinite(bb_pos) else 0.0
                    score += bb_bonus

            # 4. Optional: RSI oversold bonus
            if self._use_rsi:
                rsi = self._compute_rsi(close, self._rsi_period)
                if rsi < 30:
                    score += (30 - rsi) / 30  # max +1 when RSI=0

            scores[code] = score

            # Signal
            if latest_zscore < self._zscore_threshold:
                signals[code] = Signal.BUY
            elif latest_zscore > -self._zscore_threshold:
                signals[code] = Signal.SELL
            else:
                signals[code] = Signal.HOLD

        logger.info(
            f"MeanReversion({self._window}): scored {len(scores)} stocks on {target_date}"
        )

        return StrategyResult(
            name=self.name,
            date=target_date,
            scores=pd.Series(scores, dtype=float).sort_values(ascending=False),
            signals=pd.Series(signals) if signals else None,
        )
