"""Factor evaluation: IC/IR analysis, decay study, correlation, and HTML reporting."""

from __future__ import annotations

import base64
import io
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats


class FactorEvaluator:
    """Factor effectiveness testing toolkit."""

    # ── IC / IR ────────────────────────────────────────────────

    def compute_ic_series(
        self,
        factor_values: dict[date, pd.Series],
        forward_returns: dict[date, pd.Series],
    ) -> pd.Series:
        """Compute per-period Rank IC (Spearman correlation).

        Args:
            factor_values: {date: stock_code -> factor_value}
            forward_returns: {date: stock_code -> forward N-day return}

        Returns:
            Series indexed by date with Rank IC values.
        """
        ic_dict: dict[date, float] = {}
        for dt in sorted(factor_values.keys()):
            if dt not in forward_returns:
                continue
            fv = factor_values[dt].dropna()
            fr = forward_returns[dt].dropna()
            common = fv.index.intersection(fr.index)
            if len(common) < 5:
                continue
            corr = fv[common].corr(fr[common], method="spearman")
            if not np.isnan(corr):
                ic_dict[dt] = corr
        return pd.Series(ic_dict, dtype=float).sort_index()

    def compute_ir(self, ic_series: pd.Series) -> float:
        """Information Ratio = IC_mean / IC_std."""
        if len(ic_series) < 2 or ic_series.std() < 1e-10:
            return 0.0
        return float(ic_series.mean() / ic_series.std())

    def ic_summary(self, ic_series: pd.Series) -> dict:
        """Return IC summary statistics.

        Keys: ic_mean, ic_std, ir, ic_positive_ratio, ic_abs_gt_002_ratio, t_stat
        """
        n = len(ic_series)
        if n == 0:
            return {
                "ic_mean": np.nan, "ic_std": np.nan, "ir": np.nan,
                "ic_positive_ratio": np.nan, "ic_abs_gt_002_ratio": np.nan,
                "t_stat": np.nan, "n_periods": 0,
            }
        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std())
        ir = ic_mean / ic_std if ic_std != 0 else 0.0
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std != 0 else 0.0
        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ir": ir,
            "ic_positive_ratio": float((ic_series > 0).mean()),
            "ic_abs_gt_002_ratio": float((ic_series.abs() > 0.02).mean()),
            "t_stat": t_stat,
            "n_periods": n,
        }

    # ── IC Decay ───────────────────────────────────────────────

    @staticmethod
    def _compute_forward_returns(
        data: pd.DataFrame, period: int
    ) -> dict[date, pd.Series]:
        """Compute forward returns for each trading date.

        Args:
            data: daily data with columns [stock_code, date, close]
            period: forward holding period in trading days

        Returns:
            {date: stock_code -> forward return}
        """
        result: dict[date, pd.Series] = {}
        for code, grp in data.groupby("stock_code"):
            grp = grp.sort_values("date").reset_index(drop=True)
            closes = grp["close"].values
            dates = grp["date"].values
            for i in range(len(grp) - period):
                dt = dates[i]
                if isinstance(dt, str):
                    dt = date.fromisoformat(dt)
                elif hasattr(dt, "date"):
                    dt = dt.date() if callable(getattr(dt, "date")) else dt
                fwd_ret = closes[i + period] / closes[i] - 1
                if dt not in result:
                    result[dt] = {}
                result[dt][code] = fwd_ret  # type: ignore[index]
        return {dt: pd.Series(v) for dt, v in result.items()}

    def ic_decay(
        self,
        data: pd.DataFrame,
        factor_values: dict[date, pd.Series],
        periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """IC decay across different holding periods.

        Returns:
            DataFrame with index=period, columns=[ic_mean, ic_std, ir]
        """
        if periods is None:
            periods = [5, 10, 20, 40, 60]
        rows = []
        for p in periods:
            logger.debug(f"Computing IC decay for period={p}")
            fwd = self._compute_forward_returns(data, p)
            ic_s = self.compute_ic_series(factor_values, fwd)
            if len(ic_s) == 0:
                rows.append({"period": p, "ic_mean": np.nan, "ic_std": np.nan, "ir": np.nan})
            else:
                rows.append({
                    "period": p,
                    "ic_mean": float(ic_s.mean()),
                    "ic_std": float(ic_s.std()),
                    "ir": float(ic_s.mean() / ic_s.std()) if ic_s.std() != 0 else 0.0,
                })
        return pd.DataFrame(rows).set_index("period")

    # ── Factor Correlation ─────────────────────────────────────

    def factor_correlation(self, factor_matrix: pd.DataFrame) -> pd.DataFrame:
        """Spearman rank correlation matrix among factors.

        Args:
            factor_matrix: index=stock_code, columns=factor_name

        Returns:
            Correlation DataFrame (factor x factor).
        """
        return factor_matrix.corr(method="spearman")

    # ── HTML Report ────────────────────────────────────────────

    def generate_report(
        self,
        ic_summaries: dict[str, dict],
        ic_series_dict: dict[str, pd.Series],
        decay_dict: dict[str, pd.DataFrame],
        correlation: pd.DataFrame,
        output_dir: str,
    ) -> str:
        """Generate a self-contained HTML evaluation report.

        Includes: IC summary table, IC time-series plots, IC decay curves,
        factor correlation heatmap.

        Returns:
            Path to the generated HTML file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        sections: list[str] = []

        # ── 1. IC Summary Table ──
        rows_html = ""
        for name, s in ic_summaries.items():
            rows_html += (
                f"<tr><td>{name}</td>"
                f"<td>{s.get('ic_mean', 0):.4f}</td>"
                f"<td>{s.get('ic_std', 0):.4f}</td>"
                f"<td>{s.get('ir', 0):.4f}</td>"
                f"<td>{s.get('ic_positive_ratio', 0):.2%}</td>"
                f"<td>{s.get('ic_abs_gt_002_ratio', 0):.2%}</td>"
                f"<td>{s.get('t_stat', 0):.2f}</td>"
                f"<td>{s.get('n_periods', 0)}</td></tr>\n"
            )
        sections.append(
            "<h2>IC Summary</h2>"
            "<table><thead><tr>"
            "<th>Factor</th><th>IC Mean</th><th>IC Std</th><th>IR</th>"
            "<th>IC&gt;0 %</th><th>|IC|&gt;0.02 %</th><th>t-stat</th><th>N</th>"
            "</tr></thead><tbody>" + rows_html + "</tbody></table>"
        )

        # ── 2. IC Time-Series Plots ──
        for name, ic_s in ic_series_dict.items():
            if ic_s.empty:
                continue
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(range(len(ic_s)), ic_s.values, color=["g" if v > 0 else "r" for v in ic_s.values], alpha=0.7)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axhline(ic_s.mean(), color="blue", linestyle="--", linewidth=1, label=f"Mean={ic_s.mean():.4f}")
            ax.set_title(f"Rank IC Time Series: {name}")
            ax.set_xlabel("Period")
            ax.set_ylabel("Rank IC")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            sections.append(f"<h3>{name}</h3>" + self._fig_to_img(fig))
            plt.close(fig)

        # ── 3. IC Decay Curves ──
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, decay_df in decay_dict.items():
            if decay_df.empty:
                continue
            ax.plot(decay_df.index, decay_df["ic_mean"], marker="o", label=name)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("IC Decay across Holding Periods")
        ax.set_xlabel("Holding Period (days)")
        ax.set_ylabel("Mean IC")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        sections.append("<h2>IC Decay</h2>" + self._fig_to_img(fig))
        plt.close(fig)

        # ── 4. Correlation Heatmap ──
        if not correlation.empty:
            fig, ax = plt.subplots(figsize=(max(8, len(correlation) * 0.8), max(6, len(correlation) * 0.6)))
            sns.heatmap(
                correlation, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True,
            )
            ax.set_title("Factor Spearman Correlation")
            plt.tight_layout()
            sections.append("<h2>Factor Correlation</h2>" + self._fig_to_img(fig))
            plt.close(fig)

        # ── Assemble HTML ──
        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Factor Evaluation Report</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f9f9f9}"
            "h1{color:#333}h2{color:#555;border-bottom:2px solid #ddd;padding-bottom:5px}"
            "table{border-collapse:collapse;width:100%;margin:10px 0}"
            "th,td{border:1px solid #ddd;padding:8px;text-align:right}"
            "th{background:#4a90d9;color:white}tr:nth-child(even){background:#f2f2f2}"
            "td:first-child,th:first-child{text-align:left}"
            "img{max-width:100%;height:auto}"
            "</style></head><body>"
            "<h1>Factor Evaluation Report</h1>"
            + "\n".join(sections)
            + "</body></html>"
        )

        path = out / "factor_evaluation.html"
        path.write_text(html, encoding="utf-8")
        logger.info(f"Report saved to {path}")
        return str(path)

    @staticmethod
    def _fig_to_img(fig) -> str:
        """Convert matplotlib figure to base64-embedded <img> tag."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f'<img src="data:image/png;base64,{b64}">'
