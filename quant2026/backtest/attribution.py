"""Performance attribution analysis — Brinson-Fachler sector attribution,
regression-based factor attribution, and monthly return attribution."""

import base64
import io
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


class PerformanceAttribution:
    """Brinson-Fachler performance attribution engine."""

    def sector_attribution(
        self,
        portfolio_weights: dict[date, pd.Series],
        benchmark_weights: pd.Series,
        returns: pd.DataFrame,
        industry_map: pd.Series,
    ) -> pd.DataFrame:
        """Brinson single-period sector attribution aggregated over rebalance intervals.

        Args:
            portfolio_weights: {date: stock->weight} at each rebalance date
            benchmark_weights: stock->weight (e.g. equal-weight)
            returns: index=date, columns=stock, values=daily_return
            industry_map: stock->industry

        Returns:
            DataFrame index=industry, columns=[allocation, selection, interaction, total]
        """
        industries = industry_map.unique()
        all_stocks = returns.columns

        # Benchmark sector weights and returns
        bm_w = benchmark_weights.reindex(all_stocks, fill_value=0.0)
        bm_total = bm_w.sum()
        if bm_total > 0:
            bm_w = bm_w / bm_total

        # Map stocks to industries
        stock_ind = industry_map.reindex(all_stocks, fill_value="Other")

        # Compute benchmark sector weights
        bm_sector_w: dict[str, float] = {}
        for ind in industries:
            mask = stock_ind == ind
            bm_sector_w[ind] = bm_w[mask].sum()

        # Overall benchmark return (average daily, compounded)
        total_bm_return = (1 + returns.multiply(bm_w, axis=1).sum(axis=1)).prod() - 1

        # Benchmark sector returns
        bm_sector_ret: dict[str, float] = {}
        for ind in industries:
            mask = stock_ind == ind
            w_ind = bm_w[mask]
            if w_ind.sum() > 0:
                w_norm = w_ind / w_ind.sum()
                daily = returns.loc[:, mask].multiply(w_norm, axis=1).sum(axis=1)
                bm_sector_ret[ind] = (1 + daily).prod() - 1
            else:
                bm_sector_ret[ind] = 0.0

        # Aggregate portfolio weights across rebalance dates (time-weighted average)
        rebal_dates = sorted(portfolio_weights.keys())
        all_ret_dates = returns.index.tolist()

        # Compute portfolio sector weights and returns per rebalance interval
        port_sector_w: dict[str, float] = {ind: 0.0 for ind in industries}
        port_sector_ret: dict[str, float] = {ind: 0.0 for ind in industries}
        n_intervals = len(rebal_dates)

        for i, rd in enumerate(rebal_dates):
            pw = portfolio_weights[rd].reindex(all_stocks, fill_value=0.0)
            pw_total = pw.sum()
            if pw_total > 0:
                pw = pw / pw_total

            # Determine interval
            start_idx = all_ret_dates.index(rd) if rd in all_ret_dates else 0
            if i + 1 < len(rebal_dates):
                end_rd = rebal_dates[i + 1]
                end_idx = all_ret_dates.index(end_rd) if end_rd in all_ret_dates else len(all_ret_dates)
            else:
                end_idx = len(all_ret_dates)

            interval_dates = all_ret_dates[start_idx:end_idx]
            if not interval_dates:
                continue

            interval_ret = returns.loc[interval_dates]

            for ind in industries:
                mask = stock_ind == ind
                w_ind = pw[mask]
                port_sector_w[ind] += w_ind.sum() / n_intervals

                if w_ind.sum() > 0:
                    w_norm = w_ind / w_ind.sum()
                    daily = interval_ret.loc[:, mask].multiply(w_norm, axis=1).sum(axis=1)
                    port_sector_ret[ind] += ((1 + daily).prod() - 1) / n_intervals
                # else stays 0

        # Brinson decomposition
        rows = []
        for ind in industries:
            wp = port_sector_w[ind]
            wb = bm_sector_w.get(ind, 0.0)
            rp = port_sector_ret[ind]
            rb = bm_sector_ret.get(ind, 0.0)
            rb_total = total_bm_return

            allocation = (wp - wb) * rb
            selection = wb * (rp - rb)
            interaction = (wp - wb) * (rp - rb)
            total = allocation + selection + interaction

            rows.append({
                "industry": ind,
                "allocation": allocation,
                "selection": selection,
                "interaction": interaction,
                "total": total,
            })

        df = pd.DataFrame(rows).set_index("industry")
        logger.info(f"Sector attribution computed for {len(df)} industries")
        return df

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> dict:
        """Factor attribution via OLS regression.

        Args:
            portfolio_returns: daily portfolio returns (index=date)
            factor_returns: index=date, columns=factor_name, values=factor daily return

        Returns:
            dict with keys: factor_exposures, factor_contributions, alpha, r_squared
        """
        common = portfolio_returns.index.intersection(factor_returns.index)
        if len(common) < 10:
            logger.warning(f"Only {len(common)} common dates for factor attribution")
            return {
                "factor_exposures": {},
                "factor_contributions": {},
                "alpha": 0.0,
                "r_squared": 0.0,
            }

        y = portfolio_returns.loc[common].values
        X = factor_returns.loc[common].values
        factors = list(factor_returns.columns)

        # Add intercept
        X_with_const = np.column_stack([np.ones(len(y)), X])

        # OLS via numpy lstsq
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
        alpha = coeffs[0]
        betas = coeffs[1:]

        # R-squared
        y_pred = X_with_const @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Factor contributions = beta * mean(factor_return) * 252 (annualized)
        factor_exposures = {f: float(b) for f, b in zip(factors, betas)}
        factor_contributions = {
            f: float(b * factor_returns[f].loc[common].mean() * 252)
            for f, b in zip(factors, betas)
        }

        logger.info(f"Factor attribution: R²={r_squared:.4f}, alpha={alpha*252:.4f} (annualized)")
        return {
            "factor_exposures": factor_exposures,
            "factor_contributions": factor_contributions,
            "alpha": float(alpha),
            "r_squared": float(r_squared),
        }

    def monthly_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> pd.DataFrame:
        """Monthly return attribution.

        Args:
            portfolio_returns: daily portfolio returns
            benchmark_returns: daily benchmark returns

        Returns:
            DataFrame index=month (YYYY-MM), columns=[portfolio, benchmark, excess, cumulative_excess]
        """
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        pr = portfolio_returns.loc[common].copy()
        br = benchmark_returns.loc[common].copy()

        idx = pd.to_datetime(pr.index)
        pr.index = idx
        br.index = idx

        months = pr.groupby(idx.to_period("M"))
        rows = []
        cum_excess = 0.0

        for month, grp in months:
            p_ret = (1 + grp).prod() - 1
            b_ret = (1 + br.loc[grp.index]).prod() - 1
            excess = p_ret - b_ret
            cum_excess += excess
            rows.append({
                "month": str(month),
                "portfolio": p_ret,
                "benchmark": b_ret,
                "excess": excess,
                "cumulative_excess": cum_excess,
            })

        df = pd.DataFrame(rows).set_index("month")
        logger.info(f"Monthly attribution computed for {len(df)} months")
        return df

    # ── Report generation ──────────────────────────────────────

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def _chart_sector_attribution(self, sector_attr: pd.DataFrame) -> str:
        """Stacked bar chart for sector allocation/selection/interaction."""
        df = sector_attr.sort_values("total", ascending=True)
        fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
        y = range(len(df))
        ax.barh(y, df["allocation"], height=0.25, label="Allocation", color="#1976d2", align="center")
        ax.barh([i + 0.25 for i in y], df["selection"], height=0.25, label="Selection", color="#43a047", align="center")
        ax.barh([i + 0.5 for i in y], df["interaction"], height=0.25, label="Interaction", color="#ff9800", align="center")
        ax.set_yticks([i + 0.25 for i in y])
        ax.set_yticklabels(df.index, fontsize=8)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Sector Attribution (Brinson Decomposition)")
        ax.set_xlabel("Return Contribution")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        return self._fig_to_base64(fig)

    def _chart_factor_exposure(self, factor_attr: dict) -> str:
        """Radar chart for factor exposures."""
        exposures = factor_attr.get("factor_exposures", {})
        if not exposures:
            return ""
        labels = list(exposures.keys())
        values = [exposures[l] for l in labels]
        n = len(labels)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += [angles[0]]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values_plot, "o-", linewidth=2, color="#1976d2")
        ax.fill(angles, values_plot, alpha=0.15, color="#1976d2")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title("Factor Exposures (Beta)", y=1.08)
        return self._fig_to_base64(fig)

    def _chart_monthly_excess(self, monthly_attr: pd.DataFrame) -> str:
        """Bar chart for monthly excess returns + cumulative line."""
        fig, ax1 = plt.subplots(figsize=(10, 4))
        x = range(len(monthly_attr))
        colors = ["#43a047" if v >= 0 else "#e53935" for v in monthly_attr["excess"]]
        ax1.bar(x, monthly_attr["excess"] * 100, color=colors, alpha=0.7, label="Monthly Excess")
        ax1.set_xticks(x)
        ax1.set_xticklabels(monthly_attr.index, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("Excess Return (%)")
        ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(x, monthly_attr["cumulative_excess"] * 100, "b-o", linewidth=1.5, markersize=4, label="Cumulative")
        ax2.set_ylabel("Cumulative Excess (%)")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.set_title("Monthly Excess Return vs Benchmark")
        return self._fig_to_base64(fig)

    def generate_report(
        self,
        sector_attr: pd.DataFrame,
        factor_attr: dict,
        monthly_attr: pd.DataFrame,
        output_path: str,
    ) -> str:
        """Generate self-contained HTML attribution report.

        Args:
            sector_attr: from sector_attribution()
            factor_attr: from factor_attribution()
            monthly_attr: from monthly_attribution()
            output_path: path to write HTML

        Returns:
            path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate charts
        sector_img = self._chart_sector_attribution(sector_attr)
        factor_img = self._chart_factor_exposure(factor_attr)
        monthly_img = self._chart_monthly_excess(monthly_attr)

        # Sector attribution table
        sector_html = "<table class='metrics'><tr><th>Industry</th><th>Allocation</th><th>Selection</th><th>Interaction</th><th>Total</th></tr>"
        for ind, row in sector_attr.iterrows():
            sector_html += f"<tr><td>{ind}</td>"
            for col in ["allocation", "selection", "interaction", "total"]:
                v = row[col]
                color = "#2e7d32" if v >= 0 else "#c62828"
                sector_html += f'<td style="color:{color}">{v:.4%}</td>'
            sector_html += "</tr>"
        sector_html += "</table>"

        # Factor attribution table
        exposures = factor_attr.get("factor_exposures", {})
        contributions = factor_attr.get("factor_contributions", {})
        alpha = factor_attr.get("alpha", 0)
        r2 = factor_attr.get("r_squared", 0)

        factor_html = "<table class='metrics'><tr><th>Factor</th><th>Exposure (β)</th><th>Contribution (Ann.)</th></tr>"
        for f in exposures:
            factor_html += f"<tr><td>{f}</td><td>{exposures[f]:.4f}</td><td>{contributions.get(f, 0):.4%}</td></tr>"
        factor_html += f"<tr><td><b>Alpha (daily)</b></td><td colspan='2'>{alpha:.6f}</td></tr>"
        factor_html += f"<tr><td><b>R²</b></td><td colspan='2'>{r2:.4f}</td></tr>"
        factor_html += "</table>"

        # Monthly table
        monthly_html = "<table class='metrics'><tr><th>Month</th><th>Portfolio</th><th>Benchmark</th><th>Excess</th><th>Cum. Excess</th></tr>"
        for month, row in monthly_attr.iterrows():
            monthly_html += f"<tr><td>{month}</td>"
            for col in ["portfolio", "benchmark", "excess", "cumulative_excess"]:
                v = row[col]
                color = "#2e7d32" if v >= 0 else "#c62828"
                monthly_html += f'<td style="color:{color}">{v:.2%}</td>'
            monthly_html += "</tr>"
        monthly_html += "</table>"

        factor_section = ""
        if factor_img:
            factor_section = f'<img src="data:image/png;base64,{factor_img}" />'

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Performance Attribution Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
  h1 {{ border-bottom: 3px solid #1976d2; padding-bottom: 10px; }}
  h2 {{ color: #1976d2; margin-top: 30px; }}
  table.metrics {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
  table.metrics td, table.metrics th {{ padding: 8px 12px; border: 1px solid #ddd; text-align: right; }}
  table.metrics th {{ background: #1976d2; color: #fff; text-align: center; }}
  table.metrics td:first-child {{ text-align: left; font-weight: 600; }}
  img {{ max-width: 100%; margin: 10px 0; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #999; font-size: 0.85em; }}
</style>
</head><body>
<h1>Performance Attribution Report</h1>

<h2>Sector Attribution (Brinson-Fachler)</h2>
<img src="data:image/png;base64,{sector_img}" />
{sector_html}

<h2>Factor Attribution (Regression)</h2>
{factor_section}
{factor_html}

<h2>Monthly Excess Return</h2>
<img src="data:image/png;base64,{monthly_img}" />
{monthly_html}

<div class="footer">Generated by Quant2026 PerformanceAttribution</div>
</body></html>"""

        output_path.write_text(html, encoding="utf-8")
        logger.info(f"Attribution report saved to {output_path}")
        return str(output_path)
