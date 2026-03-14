"""Tests for signal-driven rebalance (Phase 1)."""

from datetime import date

import pandas as pd
import pytest

from quant2026.backtest.engine import BacktestConfig, BacktestEngine
from quant2026.types import PortfolioTarget, Signal


def _make_data(stocks, dates_prices):
    """Helper: build OHLCV DataFrame from {date_str: {stock: price}}."""
    rows = []
    for dt_str, prices in dates_prices.items():
        for stock, price in prices.items():
            rows.append({
                "stock_code": stock, "date": pd.Timestamp(dt_str),
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price, "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


class TestMultiFactorSignals:
    """MultiFactorStrategy now emits BUY/SELL/HOLD signals."""

    def test_signals_generated(self):
        from quant2026.strategy.multi_factor.strategy import MultiFactorStrategy
        strategy = MultiFactorStrategy(top_n=2)
        factor_matrix = pd.DataFrame(
            {"f1": [0.8, 0.5, 0.2, -0.1, -0.5]},
            index=["A", "B", "C", "D", "E"],
        )
        result = strategy.generate(pd.DataFrame(), factor_matrix, date(2024, 6, 1))
        assert result.signals is not None
        assert result.signals["A"] == Signal.BUY
        assert result.signals["B"] == Signal.BUY
        assert result.signals["D"] == Signal.SELL
        assert result.signals["E"] == Signal.SELL
        assert result.signals["C"] == Signal.HOLD


class TestEventDrivenSignals:
    """EventDrivenStrategy now emits BUY/SELL/HOLD signals."""

    def test_signals_generated(self):
        from unittest.mock import MagicMock
        from quant2026.strategy.event_driven.strategy import EventDrivenStrategy

        # Mock the EventCollector to avoid network calls
        mock_collector = MagicMock()
        mock_collector.get_earnings_surprise.return_value = pd.DataFrame(
            columns=["stock_code", "date", "event_type", "surprise_pct", "description"]
        )
        mock_collector.get_block_trades.return_value = pd.DataFrame(
            columns=["stock_code", "date", "premium_pct"]
        )
        mock_collector.get_shareholder_changes.return_value = pd.DataFrame(
            columns=["stock_code", "date", "change_type", "change_pct"]
        )

        strategy = EventDrivenStrategy(event_collector=mock_collector)
        data = pd.DataFrame({"stock_code": ["A", "B", "C"], "date": ["2024-06-01"] * 3})
        result = strategy.generate(data, None, date(2024, 6, 1))
        assert result.signals is not None
        assert len(result.signals) == 3


class TestHybridRebalance:
    """BacktestEngine hybrid rebalance mode."""

    def _build_scenario(self):
        """5 trading days, 2 stocks. Rebalance on day 1, signal SELL on day 3."""
        dates_prices = {
            "2024-01-01": {"A": 10.0, "B": 20.0},
            "2024-01-02": {"A": 10.5, "B": 19.5},
            "2024-01-03": {"A": 8.0, "B": 21.0},   # A drops, signal SELL
            "2024-01-04": {"A": 7.5, "B": 21.5},
            "2024-01-05": {"A": 7.0, "B": 22.0},
        }
        data = _make_data(["A", "B"], dates_prices)

        targets = {
            date(2024, 1, 1): PortfolioTarget(
                date=date(2024, 1, 1),
                weights=pd.Series({"A": 0.5, "B": 0.5}),
            ),
        }

        # Signal: SELL A on day 3
        daily_signals = {
            date(2024, 1, 3): pd.Series({"A": Signal.SELL, "B": Signal.HOLD}),
        }
        return data, targets, daily_signals

    def test_periodic_ignores_signals(self):
        """In periodic mode, daily_signals are ignored."""
        data, targets, daily_signals = self._build_scenario()
        cfg = BacktestConfig(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 5),
            rebalance_mode="periodic", rebalance_days=20,
            t_plus_one=False, t_plus_1=False,
        )
        engine = BacktestEngine(cfg)
        result = engine.run(data, targets, daily_signals=daily_signals)
        # A is still held at end (no signal sell in periodic mode)
        # Result should have no signal trades
        signal_trades = [t for t in result.trades if "signal" in t.get("action", "")]
        assert len(signal_trades) == 0

    def test_hybrid_sells_on_signal(self):
        """In hybrid mode, SELL signal triggers immediate sell."""
        data, targets, daily_signals = self._build_scenario()
        cfg = BacktestConfig(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 5),
            rebalance_mode="hybrid", rebalance_days=20,
            t_plus_one=False, t_plus_1=False,
        )
        engine = BacktestEngine(cfg)
        result = engine.run(data, targets, daily_signals=daily_signals)
        # Should have a signal_sell trade for A
        signal_sells = [t for t in result.trades if t.get("action") == "signal_sell"]
        assert len(signal_sells) == 1
        assert signal_sells[0]["stock"] == "A"

    def test_hybrid_buys_on_signal(self):
        """In hybrid mode, BUY signal adds small position."""
        data, targets, daily_signals = self._build_scenario()
        # Override: no initial position, BUY signal for A on day 3
        targets = {
            date(2024, 1, 1): PortfolioTarget(
                date=date(2024, 1, 1),
                weights=pd.Series({"B": 0.5}),
            ),
        }
        daily_signals = {
            date(2024, 1, 3): pd.Series({"A": Signal.BUY}),
        }
        cfg = BacktestConfig(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 5),
            rebalance_mode="hybrid", rebalance_days=20,
            signal_buy_max_weight=0.10,
            t_plus_one=False, t_plus_1=False,
        )
        engine = BacktestEngine(cfg)
        result = engine.run(data, targets, daily_signals=daily_signals)
        signal_buys = [t for t in result.trades if t.get("action") == "signal_buy"]
        assert len(signal_buys) == 1
        assert signal_buys[0]["stock"] == "A"
        assert signal_buys[0]["new_weight"] <= 0.10

    def test_signal_mode_only_signals(self):
        """In signal mode, SELL works but BUY does not add positions."""
        data, targets, daily_signals = self._build_scenario()
        cfg = BacktestConfig(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 5),
            rebalance_mode="signal", rebalance_days=20,
            t_plus_one=False, t_plus_1=False,
        )
        engine = BacktestEngine(cfg)
        result = engine.run(data, targets, daily_signals=daily_signals)
        signal_sells = [t for t in result.trades if t.get("action") == "signal_sell"]
        signal_buys = [t for t in result.trades if t.get("action") == "signal_buy"]
        assert len(signal_sells) == 1
        assert len(signal_buys) == 0  # signal mode doesn't allow inter-rebalance buys

    def test_t_plus_one_blocks_same_day_sell(self):
        """T+1 should block selling a stock bought on the same day."""
        dates_prices = {
            "2024-01-01": {"A": 10.0, "B": 20.0},
            "2024-01-02": {"A": 10.5, "B": 19.5},
        }
        data = _make_data(["A", "B"], dates_prices)
        targets = {
            date(2024, 1, 1): PortfolioTarget(
                date=date(2024, 1, 1),
                weights=pd.Series({"A": 0.5, "B": 0.5}),
            ),
        }
        # Try to SELL A on same day as buy
        daily_signals = {
            date(2024, 1, 1): pd.Series({"A": Signal.SELL}),
        }
        cfg = BacktestConfig(
            start_date=date(2024, 1, 1), end_date=date(2024, 1, 2),
            rebalance_mode="hybrid", rebalance_days=20,
            t_plus_one=True, t_plus_1=True,
        )
        engine = BacktestEngine(cfg)
        result = engine.run(data, targets, daily_signals=daily_signals)
        # T+1 should block the sell
        signal_sells = [t for t in result.trades if t.get("action") == "signal_sell"]
        assert len(signal_sells) == 0
