"""Statistical Arbitrage / Pairs Trading Strategy (统计套利/配对交易策略).

Long-only variant for A-share market: buy the underperforming stock
in a cointegrated pair when the spread deviates significantly.
"""

from datetime import date

import numpy as np
import pandas as pd
from loguru import logger

from quant2026.strategy.base import Strategy
from quant2026.strategy.stat_arb.cointegration import CointegrationAnalyzer
from quant2026.types import Signal, StrategyResult


class StatArbStrategy(Strategy):
    """Pairs trading strategy based on cointegration.

    A-share constraint: long-only, so we only buy the *lagging* stock
    when spread z-score exceeds the entry threshold.

    Args:
        lookback: Window for cointegration testing.
        entry_zscore: Open position when |z| exceeds this.
        exit_zscore: Close signal when |z| drops below this.
        max_pairs: Maximum number of pairs to hold.
        recalc_interval: Re-estimate pairs every N trading days.
    """

    def __init__(
        self,
        lookback: int = 120,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        max_pairs: int = 10,
        recalc_interval: int = 60,
    ) -> None:
        self._lookback = lookback
        self._entry_zscore = entry_zscore
        self._exit_zscore = exit_zscore
        self._max_pairs = max_pairs
        self._recalc_interval = recalc_interval

        # Cache
        self._pairs: list[dict] = []
        self._last_calc_date: date | None = None
        self._days_since_calc: int = 0
        self._analyzer = CointegrationAnalyzer()

    @property
    def name(self) -> str:
        return "stat_arb"

    # ── helpers ──────────────────────────────────────────────────

    def _build_price_matrix(
        self, data: pd.DataFrame, target_date: date
    ) -> pd.DataFrame:
        """Build price matrix from raw data up to target_date."""
        df = data[data["date"] <= str(target_date)].copy()
        pivot = df.pivot_table(index="date", columns="stock_code", values="close")
        pivot = pivot.sort_index()
        # Keep only last `lookback` rows
        return pivot.iloc[-self._lookback:]

    def _need_recalc(self, target_date: date) -> bool:
        if self._last_calc_date is None:
            return True
        delta = (target_date - self._last_calc_date).days
        return delta >= self._recalc_interval

    # ── main entry point ─────────────────────────────────────────

    def generate(
        self,
        data: pd.DataFrame,
        factor_matrix: pd.DataFrame | None,
        target_date: date,
    ) -> StrategyResult:
        """Generate pair-trading scores.

        Higher score = stronger buy signal (more oversold in its pair).
        """
        price_matrix = self._build_price_matrix(data, target_date)

        # Recalculate pairs if needed
        if self._need_recalc(target_date):
            self._pairs = self._analyzer.find_pairs(
                price_matrix, min_obs=min(self._lookback, len(price_matrix))
            )
            self._pairs = self._pairs[: self._max_pairs]
            self._last_calc_date = target_date
            logger.info(
                f"StatArb: recalculated pairs on {target_date}, "
                f"found {len(self._pairs)} pairs"
            )

        scores: dict[str, float] = {}
        signals: dict[str, Signal] = {}
        active_pairs: list[dict] = []

        for pair in self._pairs:
            sa_code = pair["stock_a"]
            sb_code = pair["stock_b"]
            hr = pair["hedge_ratio"]

            if sa_code not in price_matrix.columns or sb_code not in price_matrix.columns:
                continue

            sa = price_matrix[sa_code].dropna()
            sb = price_matrix[sb_code].dropna()
            common = sa.index.intersection(sb.index)
            if len(common) < 20:
                continue

            sa, sb = sa.loc[common], sb.loc[common]
            z_series = self._analyzer.compute_spread(sa, sb, hr)
            z = float(z_series.iloc[-1])

            if z < -self._entry_zscore:
                # stock_a is undervalued relative to stock_b → buy stock_a
                scores[sa_code] = scores.get(sa_code, 0.0) + abs(z)
                signals[sa_code] = Signal.BUY
            elif z > self._entry_zscore:
                # stock_b is undervalued relative to stock_a → buy stock_b
                scores[sb_code] = scores.get(sb_code, 0.0) + abs(z)
                signals[sb_code] = Signal.BUY
            elif abs(z) < self._exit_zscore:
                # No signal
                pass

            active_pairs.append({
                "stock_a": sa_code,
                "stock_b": sb_code,
                "hedge_ratio": round(hr, 4),
                "z_score": round(z, 4),
                "half_life": round(pair.get("half_life", 0), 1),
            })

        logger.info(
            f"StatArb: {target_date} — {len(active_pairs)} active pairs, "
            f"{len(scores)} stocks scored"
        )

        return StrategyResult(
            name=self.name,
            date=target_date,
            scores=pd.Series(scores, dtype=float).sort_values(ascending=False),
            signals=pd.Series(signals) if signals else None,
            metadata={"active_pairs": active_pairs, "total_pairs": len(self._pairs)},
        )
