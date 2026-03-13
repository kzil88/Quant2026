"""Walk-Forward Analysis engine for rolling out-of-sample backtesting."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from .engine import BacktestConfig, BacktestEngine, BacktestResult


# ── Data classes ───────────────────────────────────────────────


@dataclass
class WalkForwardConfig:
    """Walk-forward window parameters."""
    train_months: int = 6
    test_months: int = 2
    step_months: int = 2
    min_train_days: int = 100


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    in_sample_metrics: dict = field(default_factory=dict)
    out_sample_metrics: dict = field(default_factory=dict)
    out_sample_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward results."""
    windows: list[WalkForwardWindow] = field(default_factory=list)
    combined_equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    combined_metrics: dict = field(default_factory=dict)
    efficiency_ratio: float = 0.0

    def summary(self) -> dict:
        """Return summary dict."""
        return {
            "num_windows": len(self.windows),
            "efficiency_ratio": f"{self.efficiency_ratio:.3f}",
            "overfitting_warning": self.efficiency_ratio < 0.5,
            **self.combined_metrics,
        }


# ── Helper ─────────────────────────────────────────────────────


def _extract_sharpe(metrics: dict) -> float:
    """Extract numeric sharpe from metrics dict."""
    val = metrics.get("sharpe_ratio", 0)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return 0.0
    return float(val)


# ── Analyzer ───────────────────────────────────────────────────


class WalkForwardAnalyzer:
    """Walk-Forward rolling backtest analyzer."""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    # ── Window generation ──────────────────────────────────────

    def _generate_windows(self, data_start: date, data_end: date) -> list[tuple[date, date, date, date]]:
        """Generate (train_start, train_end, test_start, test_end) tuples.

        train_end is the day before test_start (no overlap).
        """
        cfg = self.config
        windows: list[tuple[date, date, date, date]] = []
        cursor = data_start

        while True:
            train_start = cursor
            train_end_dt = train_start + relativedelta(months=cfg.train_months) - relativedelta(days=1)
            test_start_dt = train_end_dt + relativedelta(days=1)
            test_end_dt = test_start_dt + relativedelta(months=cfg.test_months) - relativedelta(days=1)

            if test_end_dt > data_end:
                # Clip last window
                test_end_dt = data_end
                if test_start_dt > data_end:
                    break

            windows.append((train_start, train_end_dt, test_start_dt, test_end_dt))
            cursor += relativedelta(months=cfg.step_months)

            if test_end_dt >= data_end:
                break

        return windows

    # ── Run ────────────────────────────────────────────────────

    def run(
        self,
        data: pd.DataFrame,
        strategy_factory: Callable,
        backtest_config: BacktestConfig,
    ) -> WalkForwardResult:
        """Execute walk-forward analysis.

        Args:
            data: daily OHLCV with columns [stock_code, date, open, high, low, close, volume].
                  ``date`` can be str, datetime, or date.
            strategy_factory: ``(train_data, train_dates) -> dict[date, PortfolioTarget]``
                Returns rebalance targets for the *test* period, trained on train_data.
            backtest_config: template config (start/end will be overridden per window).

        Returns:
            WalkForwardResult with all window results combined.
        """
        # Normalise dates
        df = data.copy()
        df["_date"] = pd.to_datetime(df["date"]).dt.date
        data_start = df["_date"].min()
        data_end = df["_date"].max()

        raw_windows = self._generate_windows(data_start, data_end)
        if not raw_windows:
            logger.warning("No walk-forward windows could be generated")
            return WalkForwardResult()

        logger.info(f"Walk-forward: {len(raw_windows)} windows, "
                     f"train={self.config.train_months}m test={self.config.test_months}m step={self.config.step_months}m")

        results: list[WalkForwardWindow] = []
        oos_equities: list[pd.Series] = []

        for i, (tr_s, tr_e, te_s, te_e) in enumerate(raw_windows):
            logger.info(f"Window {i}: train {tr_s}~{tr_e}, test {te_s}~{te_e}")

            train_mask = (df["_date"] >= tr_s) & (df["_date"] <= tr_e)
            train_data = df.loc[train_mask].drop(columns=["_date"])
            train_dates = sorted(df.loc[train_mask, "_date"].unique())

            if len(train_dates) < self.config.min_train_days:
                logger.warning(f"Window {i}: only {len(train_dates)} train days (< {self.config.min_train_days}), skip")
                continue

            # Strategy factory produces targets for the test period
            try:
                targets = strategy_factory(train_data, train_dates)
            except Exception as e:
                logger.error(f"Window {i} strategy_factory failed: {e}")
                continue

            if not targets:
                logger.warning(f"Window {i}: no targets generated, skip")
                continue

            # --- In-sample backtest ---
            is_cfg = BacktestConfig(
                start_date=tr_s, end_date=tr_e,
                initial_capital=backtest_config.initial_capital,
                commission_rate=backtest_config.commission_rate,
                stamp_tax_rate=backtest_config.stamp_tax_rate,
                slippage_pct=backtest_config.slippage_pct,
            )
            # Use same targets filtered to train period for IS
            is_targets = {d: t for d, t in targets.items() if tr_s <= d <= tr_e}
            # If factory only returns test-period targets, run IS with factory again on train
            # For simplicity: IS uses whatever targets fall in train range
            is_engine = BacktestEngine(is_cfg)
            bt_train = train_data.copy()
            bt_train["date"] = pd.to_datetime(bt_train["date"]).dt.date
            is_result = is_engine.run(bt_train, is_targets) if is_targets else BacktestResult()

            # --- Out-of-sample backtest ---
            test_mask = (df["_date"] >= te_s) & (df["_date"] <= te_e)
            test_data = df.loc[test_mask].drop(columns=["_date"])
            oos_targets = {d: t for d, t in targets.items() if te_s <= d <= te_e}

            if not oos_targets:
                # Use last available target before test start
                before = {d: t for d, t in targets.items() if d < te_s}
                if before:
                    last_d = max(before)
                    oos_targets = {te_s: before[last_d]}

            oos_cfg = BacktestConfig(
                start_date=te_s, end_date=te_e,
                initial_capital=backtest_config.initial_capital,
                commission_rate=backtest_config.commission_rate,
                stamp_tax_rate=backtest_config.stamp_tax_rate,
                slippage_pct=backtest_config.slippage_pct,
            )
            oos_engine = BacktestEngine(oos_cfg)
            bt_test = test_data.copy()
            bt_test["date"] = pd.to_datetime(bt_test["date"]).dt.date
            oos_result = oos_engine.run(bt_test, oos_targets)

            is_metrics = is_result.metrics if is_result.equity_curve is not None and not is_result.equity_curve.empty else {}
            oos_metrics = oos_result.metrics if oos_result.equity_curve is not None and not oos_result.equity_curve.empty else {}

            wf_window = WalkForwardWindow(
                window_id=i,
                train_start=tr_s,
                train_end=tr_e,
                test_start=te_s,
                test_end=te_e,
                in_sample_metrics=is_metrics,
                out_sample_metrics=oos_metrics,
                out_sample_equity=oos_result.equity_curve if not oos_result.equity_curve.empty else pd.Series(dtype=float),
            )
            results.append(wf_window)

            if not oos_result.equity_curve.empty:
                oos_equities.append(oos_result.equity_curve)

        if not results:
            logger.warning("No valid walk-forward windows")
            return WalkForwardResult()

        # Combine OOS equity curves (chain them)
        combined = self._chain_equity(oos_equities)
        combined_metrics = self._compute_combined_metrics(combined)

        # Efficiency ratio
        is_sharpes = [_extract_sharpe(w.in_sample_metrics) for w in results if w.in_sample_metrics]
        oos_sharpes = [_extract_sharpe(w.out_sample_metrics) for w in results if w.out_sample_metrics]
        avg_is = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_oos = np.mean(oos_sharpes) if oos_sharpes else 0.0
        eff = avg_oos / avg_is if abs(avg_is) > 1e-9 else 0.0

        wf_result = WalkForwardResult(
            windows=results,
            combined_equity=combined,
            combined_metrics=combined_metrics,
            efficiency_ratio=eff,
        )
        logger.info(f"Walk-forward complete: {len(results)} windows, efficiency={eff:.3f}")
        return wf_result

    # ── Equity chaining ────────────────────────────────────────

    @staticmethod
    def _chain_equity(equities: list[pd.Series]) -> pd.Series:
        """Chain OOS equity curves into continuous NAV starting at 1.0."""
        if not equities:
            return pd.Series(dtype=float)

        parts: list[pd.Series] = []
        current_nav = 1.0

        for eq in equities:
            if eq.empty:
                continue
            nav = eq / eq.iloc[0] * current_nav
            parts.append(nav)
            current_nav = nav.iloc[-1]

        if not parts:
            return pd.Series(dtype=float)
        return pd.concat(parts)

    @staticmethod
    def _compute_combined_metrics(equity: pd.Series) -> dict:
        """Compute summary metrics for combined OOS equity."""
        if equity.empty or len(equity) < 2:
            return {}
        dr = equity.pct_change().dropna()
        total_ret = equity.iloc[-1] / equity.iloc[0] - 1
        n_days = (pd.Timestamp(equity.index[-1]) - pd.Timestamp(equity.index[0])).days
        years = max(n_days / 365.25, 0.01)
        annual_ret = (1 + total_ret) ** (1 / years) - 1
        vol = dr.std() * np.sqrt(252)
        sharpe = (dr.mean() - 0.025 / 252) / dr.std() * np.sqrt(252) if dr.std() > 0 else 0.0
        cummax = equity.cummax()
        max_dd = ((equity - cummax) / cummax).min()
        return {
            "total_return": f"{total_ret:.2%}",
            "annual_return": f"{annual_ret:.2%}",
            "volatility": f"{vol:.2%}",
            "sharpe_ratio": f"{sharpe:.2f}",
            "max_drawdown": f"{max_dd:.2%}",
        }

    # ── Report ─────────────────────────────────────────────────

    def generate_report(self, result: WalkForwardResult, output_path: str) -> str:
        """Generate self-contained HTML walk-forward report.

        Args:
            result: WalkForwardResult from run()
            output_path: file path for HTML output

        Returns:
            Path string of generated report.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        timeline_img = self._chart_timeline(result)
        equity_img = self._chart_combined_equity(result)
        window_imgs = self._chart_window_equities(result)
        table_html = self._metrics_table(result)

        eff = result.efficiency_ratio
        eff_color = "#c62828" if eff < 0.5 else ("#f57c00" if eff < 0.8 else "#2e7d32")
        eff_label = "⚠️ OVERFITTING WARNING" if eff < 0.5 else ("Moderate" if eff < 0.8 else "Good")

        window_sections = ""
        for img in window_imgs:
            window_sections += f'<img src="data:image/png;base64,{img}" style="max-width:48%;display:inline-block;margin:4px;" />\n'

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Walk-Forward Analysis Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 3px solid #1565c0; padding-bottom: 10px; }}
  h2 {{ color: #1565c0; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
  th {{ background: #1565c0; color: #fff; padding: 8px 12px; text-align: center; }}
  td {{ padding: 6px 12px; border: 1px solid #ddd; text-align: right; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  img {{ max-width: 100%; margin: 10px 0; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .eff-box {{ font-size: 1.3em; padding: 15px; border-radius: 8px; background: #fff;
              border: 2px solid {eff_color}; margin: 15px 0; text-align: center; }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #999; font-size: 0.85em; }}
</style>
</head><body>
<h1>Walk-Forward Analysis Report</h1>
<p>Windows: {len(result.windows)} | Train: {self.config.train_months}m | Test: {self.config.test_months}m | Step: {self.config.step_months}m</p>

<div class="eff-box" style="color: {eff_color};">
  Efficiency Ratio (OOS Sharpe / IS Sharpe): <b>{eff:.3f}</b> — {eff_label}
</div>

<h2>Window Timeline</h2>
<img src="data:image/png;base64,{timeline_img}" />

<h2>IS vs OOS Metrics by Window</h2>
{table_html}

<h2>Combined OOS Equity Curve</h2>
<img src="data:image/png;base64,{equity_img}" />

<h2>Per-Window OOS Equity</h2>
{window_sections}

<h2>Combined OOS Performance</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{"".join(f"<tr><td style='text-align:left'>{k}</td><td>{v}</td></tr>" for k, v in result.combined_metrics.items())}
</table>

<div class="footer">Generated by Quant2026 WalkForwardAnalyzer</div>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info(f"Walk-forward report saved to {path}")
        return str(path)

    # ── Chart helpers ──────────────────────────────────────────

    @staticmethod
    def _fig_to_b64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _chart_timeline(self, result: WalkForwardResult) -> str:
        """Timeline chart with train (blue) and test (green) bars."""
        fig, ax = plt.subplots(figsize=(12, max(2, len(result.windows) * 0.5 + 1)))
        for w in result.windows:
            y = w.window_id
            train_days = (w.train_end - w.train_start).days
            test_days = (w.test_end - w.test_start).days
            ax.barh(y, train_days, left=(w.train_start - result.windows[0].train_start).days,
                    color="#1565c0", alpha=0.7, height=0.6)
            ax.barh(y, test_days, left=(w.test_start - result.windows[0].train_start).days,
                    color="#2e7d32", alpha=0.7, height=0.6)
        ax.set_yticks([w.window_id for w in result.windows])
        ax.set_yticklabels([f"W{w.window_id}" for w in result.windows])
        ax.set_xlabel("Days from start")
        ax.set_title("Walk-Forward Window Timeline (Blue=Train, Green=Test)")
        ax.legend(handles=[
            mpatches.Patch(color="#1565c0", alpha=0.7, label="Train"),
            mpatches.Patch(color="#2e7d32", alpha=0.7, label="Test"),
        ])
        ax.grid(True, alpha=0.3)
        return self._fig_to_b64(fig)

    def _chart_combined_equity(self, result: WalkForwardResult) -> str:
        """Combined OOS equity curve."""
        if result.combined_equity.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return self._fig_to_b64(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        eq = result.combined_equity
        ax.plot(eq.index, eq.values, linewidth=1.5, color="#1565c0")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
        ax.set_title("Combined Out-of-Sample Equity Curve")
        ax.set_ylabel("NAV")
        ax.grid(True, alpha=0.3)
        return self._fig_to_b64(fig)

    def _chart_window_equities(self, result: WalkForwardResult) -> list[str]:
        """Small equity charts per window."""
        imgs = []
        for w in result.windows:
            if w.out_sample_equity.empty:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            nav = w.out_sample_equity / w.out_sample_equity.iloc[0]
            ax.plot(nav.index, nav.values, linewidth=1.2, color="#2e7d32")
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
            ax.set_title(f"W{w.window_id}: {w.test_start} ~ {w.test_end}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            imgs.append(self._fig_to_b64(fig))
        return imgs

    def _metrics_table(self, result: WalkForwardResult) -> str:
        """HTML table of IS vs OOS metrics per window."""
        rows = ""
        for w in result.windows:
            is_sharpe = w.in_sample_metrics.get("sharpe_ratio", "N/A")
            oos_sharpe = w.out_sample_metrics.get("sharpe_ratio", "N/A")
            is_ret = w.in_sample_metrics.get("total_return", "N/A")
            oos_ret = w.out_sample_metrics.get("total_return", "N/A")
            is_dd = w.in_sample_metrics.get("max_drawdown", "N/A")
            oos_dd = w.out_sample_metrics.get("max_drawdown", "N/A")
            rows += f"""<tr>
                <td style="text-align:left">W{w.window_id}</td>
                <td style="text-align:left">{w.train_start}~{w.train_end}</td>
                <td style="text-align:left">{w.test_start}~{w.test_end}</td>
                <td>{is_ret}</td><td>{oos_ret}</td>
                <td>{is_sharpe}</td><td>{oos_sharpe}</td>
                <td>{is_dd}</td><td>{oos_dd}</td>
            </tr>"""
        return f"""<table>
            <tr><th>Window</th><th>Train Period</th><th>Test Period</th>
            <th>IS Return</th><th>OOS Return</th>
            <th>IS Sharpe</th><th>OOS Sharpe</th>
            <th>IS MaxDD</th><th>OOS MaxDD</th></tr>
            {rows}</table>"""
