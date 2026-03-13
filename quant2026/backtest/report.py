"""Backtest report generator — self-contained HTML with embedded charts."""

import base64
import io
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from .engine import BacktestResult

sns.set_theme(style="whitegrid", palette="muted")


@dataclass
class ExtendedMetrics:
    """Extended performance metrics."""
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_days: int = 0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    # Benchmark-relative
    benchmark_total_return: float = 0.0
    benchmark_annual_return: float = 0.0
    excess_return: float = 0.0
    excess_annual_return: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0


class BacktestReporter:
    """Generate self-contained HTML backtest reports.

    Args:
        result: BacktestResult from engine
        benchmark: DataFrame with [date, close] for benchmark index
        benchmark_name: display name for the benchmark
    """

    def __init__(
        self,
        result: BacktestResult,
        benchmark: pd.DataFrame | None = None,
        benchmark_name: str = "CSI 300",
    ):
        self.result = result
        self.benchmark_name = benchmark_name
        self._prepare_benchmark(benchmark)
        self.metrics = self._compute_metrics()

    def _prepare_benchmark(self, benchmark: pd.DataFrame | None) -> None:
        """Align benchmark to strategy dates and compute NAV."""
        if benchmark is None or benchmark.empty:
            self.bench_nav = pd.Series(dtype=float)
            return

        bm = benchmark.copy()
        bm["date"] = pd.to_datetime(bm["date"])
        bm = bm.sort_values("date").set_index("date")["close"]

        # Align to strategy equity curve dates
        eq_idx = pd.to_datetime(self.result.equity_curve.index)
        bm = bm.reindex(eq_idx, method="ffill").dropna()
        self.bench_nav = bm / bm.iloc[0] if len(bm) > 0 else pd.Series(dtype=float)

    def _compute_metrics(self) -> ExtendedMetrics:
        """Compute all extended metrics."""
        eq = self.result.equity_curve
        dr = self.result.daily_returns
        if eq.empty:
            return ExtendedMetrics()

        nav = eq / eq.iloc[0]
        total_ret = nav.iloc[-1] - 1
        n_days = (pd.Timestamp(eq.index[-1]) - pd.Timestamp(eq.index[0])).days
        years = max(n_days / 365.25, 0.01)
        annual_ret = (1 + total_ret) ** (1 / years) - 1
        vol = dr.std() * np.sqrt(252)
        sharpe = ((dr.mean() - 0.025 / 252) / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

        # Max drawdown & duration
        cummax = nav.cummax()
        dd = (nav - cummax) / cummax
        max_dd = dd.min()

        # Drawdown duration
        in_dd = dd < 0
        max_dd_days = 0
        current_dd_days = 0
        for v in in_dd:
            if v:
                current_dd_days += 1
                max_dd_days = max(max_dd_days, current_dd_days)
            else:
                current_dd_days = 0

        calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0
        win_rate = (dr > 0).sum() / max(len(dr[dr != 0]), 1)

        m = ExtendedMetrics(
            total_return=total_ret,
            annual_return=annual_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_days=max_dd_days,
            calmar_ratio=calmar,
            win_rate=win_rate,
        )

        # Benchmark metrics
        if not self.bench_nav.empty and len(self.bench_nav) > 1:
            bm_ret = self.bench_nav.iloc[-1] - 1
            bm_annual = (1 + bm_ret) ** (1 / years) - 1
            m.benchmark_total_return = bm_ret
            m.benchmark_annual_return = bm_annual
            m.excess_return = total_ret - bm_ret
            m.excess_annual_return = annual_ret - bm_annual

            # Information ratio
            bench_dr = self.bench_nav.pct_change().dropna()
            common = dr.index.intersection(bench_dr.index)
            if len(common) > 10:
                excess_dr = dr.loc[common] - bench_dr.loc[common]
                te = excess_dr.std() * np.sqrt(252)
                m.tracking_error = te
                m.information_ratio = m.excess_annual_return / te if te > 1e-9 else 0.0

        return m

    # ── Chart helpers ──────────────────────────────────────────

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _chart_equity(self) -> str:
        """Strategy vs benchmark NAV chart."""
        fig, ax = plt.subplots(figsize=(10, 5))
        nav = self.result.equity_curve / self.result.equity_curve.iloc[0]
        ax.plot(nav.index, nav.values, linewidth=1.5, label="Strategy")
        if not self.bench_nav.empty:
            ax.plot(self.bench_nav.index, self.bench_nav.values, linewidth=1.5, label=self.benchmark_name, alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
        ax.set_title("NAV: Strategy vs Benchmark")
        ax.set_ylabel("NAV")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._fig_to_base64(fig)

    def _chart_excess(self) -> str:
        """Excess return curve."""
        if self.bench_nav.empty:
            return ""
        fig, ax = plt.subplots(figsize=(10, 3.5))
        nav = self.result.equity_curve / self.result.equity_curve.iloc[0]
        # align
        common = nav.index.intersection(self.bench_nav.index)
        excess = nav.loc[common] - self.bench_nav.loc[common]
        ax.plot(common, excess.values, color="green", linewidth=1.2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)
        ax.fill_between(common, excess.values, 0, where=excess.values >= 0, color="green", alpha=0.15)
        ax.fill_between(common, excess.values, 0, where=excess.values < 0, color="red", alpha=0.15)
        ax.set_title("Excess Return (Strategy - Benchmark)")
        ax.set_ylabel("Excess NAV")
        ax.grid(True, alpha=0.3)
        return self._fig_to_base64(fig)

    def _chart_drawdown(self) -> str:
        """Drawdown curve."""
        fig, ax = plt.subplots(figsize=(10, 3))
        nav = self.result.equity_curve / self.result.equity_curve.iloc[0]
        dd = (nav - nav.cummax()) / nav.cummax()
        ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.35)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown")
        ax.grid(True, alpha=0.3)
        return self._fig_to_base64(fig)

    def _chart_monthly_heatmap(self) -> str:
        """Monthly returns heatmap (year × month)."""
        dr = self.result.daily_returns.copy()
        idx = pd.to_datetime(dr.index)
        dr.index = idx
        monthly = dr.groupby([idx.year, idx.month]).apply(lambda x: (1 + x).prod() - 1)
        monthly.index.names = ["Year", "Month"]
        table = monthly.unstack("Month")
        table.columns = [f"{m:02d}" for m in table.columns]

        fig, ax = plt.subplots(figsize=(10, max(2.5, len(table) * 0.8 + 1)))
        sns.heatmap(
            table.astype(float) * 100,
            annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Return (%)"},
        )
        ax.set_title("Monthly Returns (%)")
        ax.set_ylabel("Year")
        ax.set_xlabel("Month")
        return self._fig_to_base64(fig)

    # ── Monthly stats table ────────────────────────────────────

    def _monthly_stats_html(self) -> str:
        """Build monthly return stats as HTML table."""
        dr = self.result.daily_returns.copy()
        idx = pd.to_datetime(dr.index)
        dr.index = idx
        monthly = dr.groupby([idx.year, idx.month]).apply(lambda x: (1 + x).prod() - 1)
        monthly.index.names = ["Year", "Month"]
        table = monthly.unstack("Month")
        table.columns = [f"{m:02d}" for m in table.columns]
        # Add yearly total
        table["Year Total"] = table.apply(lambda row: (1 + row.dropna()).prod() - 1, axis=1)

        rows_html = ""
        for year, row in table.iterrows():
            cells = f"<td><b>{year}</b></td>"
            for val in row:
                if pd.isna(val):
                    cells += "<td>-</td>"
                else:
                    color = "#2e7d32" if val >= 0 else "#c62828"
                    cells += f'<td style="color:{color}">{val:.2%}</td>'
            rows_html += f"<tr>{cells}</tr>\n"

        months = list(table.columns)
        header = "<th>Year</th>" + "".join(f"<th>{m}</th>" for m in months)
        return f"<table class='stats'><tr>{header}</tr>{rows_html}</table>"

    # ── HTML generation ────────────────────────────────────────

    def generate_html(self, output_path: str | Path) -> Path:
        """Generate self-contained HTML report.

        Args:
            output_path: where to write the HTML file

        Returns:
            Path to generated file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        m = self.metrics

        # Generate charts
        equity_img = self._chart_equity()
        excess_img = self._chart_excess()
        dd_img = self._chart_drawdown()
        heatmap_img = self._chart_monthly_heatmap()
        monthly_table = self._monthly_stats_html()

        def pct(v: float) -> str:
            return f"{v:.2%}"

        def f2(v: float) -> str:
            return f"{v:.2f}"

        excess_section = ""
        if excess_img:
            excess_section = f"""
            <h2>Excess Return</h2>
            <img src="data:image/png;base64,{excess_img}" />
            """

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Backtest Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 3px solid #1976d2; padding-bottom: 10px; }}
  h2 {{ color: #1976d2; margin-top: 30px; }}
  table.metrics {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  table.metrics td, table.metrics th {{ padding: 8px 14px; border: 1px solid #ddd; text-align: right; }}
  table.metrics th {{ background: #1976d2; color: #fff; text-align: center; }}
  table.metrics tr:nth-child(even) {{ background: #f5f5f5; }}
  table.metrics td:first-child {{ text-align: left; font-weight: 600; }}
  table.stats {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
  table.stats td, table.stats th {{ padding: 6px 10px; border: 1px solid #ddd; text-align: right; }}
  table.stats th {{ background: #455a64; color: #fff; }}
  table.stats td:first-child {{ text-align: left; font-weight: 600; }}
  img {{ max-width: 100%; margin: 10px 0; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .positive {{ color: #2e7d32; }}
  .negative {{ color: #c62828; }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #999; font-size: 0.85em; }}
</style>
</head><body>
<h1>Quant2026 Backtest Report</h1>

<h2>Performance Summary</h2>
<table class="metrics">
  <tr><th colspan="2">Strategy</th><th colspan="2">Benchmark ({self.benchmark_name})</th></tr>
  <tr><td>Total Return</td><td>{pct(m.total_return)}</td><td>Total Return</td><td>{pct(m.benchmark_total_return)}</td></tr>
  <tr><td>Annual Return</td><td>{pct(m.annual_return)}</td><td>Annual Return</td><td>{pct(m.benchmark_annual_return)}</td></tr>
  <tr><td>Volatility</td><td>{pct(m.volatility)}</td><td>Excess Return</td><td>{pct(m.excess_return)}</td></tr>
  <tr><td>Sharpe Ratio</td><td>{f2(m.sharpe_ratio)}</td><td>Excess Annual</td><td>{pct(m.excess_annual_return)}</td></tr>
  <tr><td>Max Drawdown</td><td>{pct(m.max_drawdown)}</td><td>Information Ratio</td><td>{f2(m.information_ratio)}</td></tr>
  <tr><td>Max DD Duration</td><td>{m.max_drawdown_days} days</td><td>Tracking Error</td><td>{pct(m.tracking_error)}</td></tr>
  <tr><td>Calmar Ratio</td><td>{f2(m.calmar_ratio)}</td><td colspan="2"></td></tr>
  <tr><td>Win Rate</td><td>{pct(m.win_rate)}</td><td colspan="2"></td></tr>
</table>

<h2>NAV Curve</h2>
<img src="data:image/png;base64,{equity_img}" />

{excess_section}

<h2>Drawdown</h2>
<img src="data:image/png;base64,{dd_img}" />

<h2>Monthly Returns Heatmap</h2>
<img src="data:image/png;base64,{heatmap_img}" />

<h2>Monthly Return Statistics</h2>
{monthly_table}

<div class="footer">Generated by Quant2026 BacktestReporter</div>
</body></html>"""

        output_path.write_text(html, encoding="utf-8")
        logger.info(f"HTML report saved to {output_path}")
        return output_path
