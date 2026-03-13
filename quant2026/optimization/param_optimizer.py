"""Strategy parameter optimization framework.

Supports grid search, random search, and Bayesian optimization (GP + EI).
No dependency on optuna/hyperopt — uses sklearn GP and scipy for EI maximization.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from quant2026.backtest.engine import BacktestConfig, BacktestResult
from quant2026.strategy.base import Strategy


# ── Data classes ─────────────────────────────────────────────────


@dataclass
class ParamSpace:
    """Define search space for a single parameter.

    Args:
        name: Parameter name (must match strategy constructor kwarg).
        type: "float", "int", or "choice".
        low: Lower bound (float/int types).
        high: Upper bound (float/int types).
        step: Grid step size (grid search only).
        choices: Candidate values (choice type).
    """

    name: str
    type: str = "float"
    low: float | None = None
    high: float | None = None
    step: float | None = None
    choices: list | None = None

    def __post_init__(self) -> None:
        if self.type not in ("float", "int", "choice"):
            raise ValueError(f"ParamSpace type must be float/int/choice, got '{self.type}'")
        if self.type == "choice":
            if not self.choices or len(self.choices) == 0:
                raise ValueError(f"ParamSpace '{self.name}': choice type requires non-empty choices")
        else:
            if self.low is None or self.high is None:
                raise ValueError(f"ParamSpace '{self.name}': float/int type requires low and high")
            if self.low > self.high:
                raise ValueError(f"ParamSpace '{self.name}': low ({self.low}) > high ({self.high})")

    def grid_values(self) -> list:
        """Return discrete grid values for this parameter."""
        if self.type == "choice":
            return list(self.choices)  # type: ignore[arg-type]
        if self.step is None:
            raise ValueError(f"ParamSpace '{self.name}': grid search requires step")
        vals = np.arange(self.low, self.high + self.step * 0.5, self.step)
        if self.type == "int":
            vals = np.unique(vals.astype(int))
        return vals.tolist()

    def sample_random(self, rng: np.random.RandomState) -> Any:
        """Sample a random value from this space."""
        if self.type == "choice":
            return rng.choice(self.choices)  # type: ignore[arg-type]
        if self.type == "int":
            return int(rng.randint(int(self.low), int(self.high) + 1))  # type: ignore[arg-type]
        return float(rng.uniform(self.low, self.high))  # type: ignore[arg-type]


@dataclass
class OptimizationResult:
    """Result of a parameter optimization run.

    Attributes:
        best_params: Best parameter combination found.
        best_score: Objective value of the best combination.
        best_metrics: Full backtest metrics dict for the best run.
        all_results: DataFrame with columns [param1, param2, ..., score, + metric columns].
        optimization_time: Wall-clock seconds for the entire optimization.
    """

    best_params: dict = field(default_factory=dict)
    best_score: float = float("-inf")
    best_metrics: dict = field(default_factory=dict)
    all_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    optimization_time: float = 0.0


# ── Objective extraction ─────────────────────────────────────────

_OBJECTIVE_MAP: dict[str, str] = {
    "sharpe": "sharpe_ratio",
    "return": "annual_return",
    "calmar": "calmar_ratio",
    "sortino": "sortino_ratio",
}


def _extract_score(result: BacktestResult, objective: str) -> float:
    """Extract a scalar score from BacktestResult.metrics."""
    key = _OBJECTIVE_MAP.get(objective, objective)
    metrics = result.metrics
    if not metrics:
        return float("-inf")

    val = metrics.get(key)
    if val is None:
        # Try alternative keys
        val = metrics.get(objective)
    if val is None:
        return float("-inf")

    # Handle string values like "12.34%"
    if isinstance(val, str):
        val = val.replace("%", "").strip()
        try:
            return float(val)
        except ValueError:
            return float("-inf")
    return float(val)


# ── Main optimizer ───────────────────────────────────────────────


class StrategyOptimizer:
    """Strategy parameter optimizer supporting grid, random, and Bayesian search.

    Args:
        objective: Metric to maximize. One of "sharpe", "return", "calmar", "sortino".
        n_jobs: Reserved for future parallel support (currently single-threaded).
    """

    def __init__(self, objective: str = "sharpe", n_jobs: int = 1) -> None:
        self.objective = objective
        self.n_jobs = n_jobs

    # ── Grid search ──────────────────────────────────────────────

    def grid_search(
        self,
        param_spaces: list[ParamSpace],
        strategy_factory: Callable[[dict], Strategy],
        data: pd.DataFrame,
        backtest_config: BacktestConfig,
        pipeline_fn: Callable[..., BacktestResult],
    ) -> OptimizationResult:
        """Exhaustive grid search over all parameter combinations.

        Args:
            param_spaces: List of parameter search spaces (each needs ``step`` or ``choices``).
            strategy_factory: Callable that takes a param dict and returns a Strategy.
            data: Market data DataFrame.
            backtest_config: Backtest configuration.
            pipeline_fn: ``(strategy, data, config) -> BacktestResult``.

        Returns:
            OptimizationResult with all evaluated combinations.
        """
        grid_vals = [ps.grid_values() for ps in param_spaces]
        combos = list(itertools.product(*grid_vals))
        param_names = [ps.name for ps in param_spaces]

        logger.info(f"Grid search: {len(combos)} combinations for params {param_names}")
        return self._evaluate_combos(
            combos, param_names, strategy_factory, data, backtest_config, pipeline_fn
        )

    # ── Random search ────────────────────────────────────────────

    def random_search(
        self,
        param_spaces: list[ParamSpace],
        strategy_factory: Callable[[dict], Strategy],
        data: pd.DataFrame,
        backtest_config: BacktestConfig,
        pipeline_fn: Callable[..., BacktestResult],
        n_iter: int = 50,
        seed: int = 42,
    ) -> OptimizationResult:
        """Random search: sample ``n_iter`` random parameter combinations.

        Args:
            param_spaces: Parameter search spaces.
            strategy_factory: Param dict → Strategy.
            data: Market data.
            backtest_config: Backtest configuration.
            pipeline_fn: ``(strategy, data, config) -> BacktestResult``.
            n_iter: Number of random samples.
            seed: RNG seed for reproducibility.

        Returns:
            OptimizationResult.
        """
        rng = np.random.RandomState(seed)
        param_names = [ps.name for ps in param_spaces]
        combos = [
            tuple(ps.sample_random(rng) for ps in param_spaces) for _ in range(n_iter)
        ]
        logger.info(f"Random search: {n_iter} iterations for params {param_names}")
        return self._evaluate_combos(
            combos, param_names, strategy_factory, data, backtest_config, pipeline_fn
        )

    # ── Bayesian search ──────────────────────────────────────────

    def bayesian_search(
        self,
        param_spaces: list[ParamSpace],
        strategy_factory: Callable[[dict], Strategy],
        data: pd.DataFrame,
        backtest_config: BacktestConfig,
        pipeline_fn: Callable[..., BacktestResult],
        n_iter: int = 30,
        n_initial: int = 10,
    ) -> OptimizationResult:
        """Bayesian optimization with Gaussian Process surrogate and EI acquisition.

        Uses ``sklearn.gaussian_process.GaussianProcessRegressor`` for the surrogate
        and ``scipy.optimize.minimize`` to maximize Expected Improvement.

        Args:
            param_spaces: Parameter search spaces.
            strategy_factory: Param dict → Strategy.
            data: Market data.
            backtest_config: Backtest configuration.
            pipeline_fn: ``(strategy, data, config) -> BacktestResult``.
            n_iter: Total evaluations (including initial random points).
            n_initial: Number of initial random evaluations before GP kicks in.

        Returns:
            OptimizationResult.
        """
        from scipy.optimize import minimize as scipy_minimize
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern

        param_names = [ps.name for ps in param_spaces]
        n_initial = min(n_initial, n_iter)
        rng = np.random.RandomState(42)

        logger.info(
            f"Bayesian search: {n_iter} iterations ({n_initial} initial) for {param_names}"
        )

        # Build bounds for continuous space (encode choices as int indices)
        bounds_low: list[float] = []
        bounds_high: list[float] = []
        for ps in param_spaces:
            if ps.type == "choice":
                bounds_low.append(0.0)
                bounds_high.append(float(len(ps.choices) - 1))  # type: ignore[arg-type]
            else:
                bounds_low.append(float(ps.low))  # type: ignore[arg-type]
                bounds_high.append(float(ps.high))  # type: ignore[arg-type]

        bounds_low_arr = np.array(bounds_low)
        bounds_high_arr = np.array(bounds_high)

        def _decode(x: np.ndarray) -> dict:
            params: dict[str, Any] = {}
            for i, ps in enumerate(param_spaces):
                if ps.type == "choice":
                    idx = int(np.clip(np.round(x[i]), 0, len(ps.choices) - 1))  # type: ignore
                    params[ps.name] = ps.choices[idx]  # type: ignore
                elif ps.type == "int":
                    params[ps.name] = int(np.round(x[i]))
                else:
                    params[ps.name] = float(x[i])
            return params

        def _encode(params: dict) -> np.ndarray:
            x = []
            for ps in param_spaces:
                if ps.type == "choice":
                    x.append(float(ps.choices.index(params[ps.name])))  # type: ignore
                else:
                    x.append(float(params[ps.name]))
            return np.array(x)

        # Phase 1: random initial points
        X_observed: list[np.ndarray] = []
        y_observed: list[float] = []
        all_rows: list[dict] = []
        t0 = time.time()

        for i in tqdm(range(n_initial), desc="Bayesian (init)"):
            x = rng.uniform(bounds_low_arr, bounds_high_arr)
            params = _decode(x)
            score, metrics = self._run_single(
                params, strategy_factory, data, backtest_config, pipeline_fn
            )
            X_observed.append(_encode(params))
            y_observed.append(score)
            row = {**params, "score": score, **metrics}
            all_rows.append(row)
            logger.debug(f"[Bayesian {i+1}/{n_iter}] params={params} score={score:.4f}")

        # Phase 2: GP + EI
        gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )

        for i in tqdm(range(n_initial, n_iter), desc="Bayesian (GP)"):
            X_arr = np.array(X_observed)
            y_arr = np.array(y_observed)
            gp.fit(X_arr, y_arr)

            best_y = y_arr.max()

            def neg_ei(x: np.ndarray) -> float:
                x_2d = x.reshape(1, -1)
                mu, sigma = gp.predict(x_2d, return_std=True)
                sigma = max(sigma[0], 1e-9)
                z = (mu[0] - best_y) / sigma
                ei = (mu[0] - best_y) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei

            # Multi-start L-BFGS-B
            best_x = None
            best_neg_ei = float("inf")
            scipy_bounds = list(zip(bounds_low, bounds_high))
            for _ in range(20):
                x0 = rng.uniform(bounds_low_arr, bounds_high_arr)
                try:
                    res = scipy_minimize(neg_ei, x0, bounds=scipy_bounds, method="L-BFGS-B")
                    if res.fun < best_neg_ei:
                        best_neg_ei = res.fun
                        best_x = res.x
                except Exception:
                    continue

            if best_x is None:
                best_x = rng.uniform(bounds_low_arr, bounds_high_arr)

            params = _decode(best_x)
            score, metrics = self._run_single(
                params, strategy_factory, data, backtest_config, pipeline_fn
            )
            X_observed.append(_encode(params))
            y_observed.append(score)
            all_rows.append({**params, "score": score, **metrics})
            logger.debug(f"[Bayesian {i+1}/{n_iter}] params={params} score={score:.4f}")

        elapsed = time.time() - t0
        return self._build_result(all_rows, param_names, elapsed)

    # ── Report generation ────────────────────────────────────────

    def generate_report(
        self,
        result: OptimizationResult,
        param_spaces: list[ParamSpace],
        output_path: str,
    ) -> str:
        """Generate an HTML optimization report.

        Includes:
        - Best parameters table
        - Parameter sensitivity scatter plots (each param vs objective)
        - Heatmap if exactly 2 parameters
        - Full results ranking table
        - Best run metrics

        Args:
            result: Optimization result.
            param_spaces: Parameter spaces used.
            output_path: Path to write the HTML file.

        Returns:
            Absolute path of the generated report.
        """
        import base64
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        param_names = [ps.name for ps in param_spaces]
        df = result.all_results.copy()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        figures_html: list[str] = []

        # 1. Sensitivity scatter plots
        for pname in param_names:
            if pname not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df[pname], df["score"], alpha=0.6, edgecolors="k", linewidth=0.3)
            ax.set_xlabel(pname)
            ax.set_ylabel("Score")
            ax.set_title(f"Sensitivity: {pname} vs Objective")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            figures_html.append(_fig_to_html(fig))
            plt.close(fig)

        # 2. Heatmap for 2 params
        if len(param_names) == 2:
            p1, p2 = param_names
            try:
                pivot = df.pivot_table(index=p2, columns=p1, values="score", aggfunc="mean")
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(
                    pivot.values, aspect="auto", cmap="RdYlGn",
                    extent=[
                        pivot.columns.min(), pivot.columns.max(),
                        pivot.index.min(), pivot.index.max(),
                    ],
                    origin="lower",
                )
                ax.set_xlabel(p1)
                ax.set_ylabel(p2)
                ax.set_title("Parameter Heatmap")
                fig.colorbar(im, label="Score")
                fig.tight_layout()
                figures_html.append(_fig_to_html(fig))
                plt.close(fig)
            except Exception:
                pass

        # 3. Build HTML
        ranked = df.sort_values("score", ascending=False).reset_index(drop=True)
        ranked.index = ranked.index + 1
        ranked.index.name = "Rank"

        best_params_html = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in result.best_params.items()
        )
        best_metrics_html = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in result.best_metrics.items()
        )

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Optimization Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #1a1a2e; }} h2 {{ color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #0f3460; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.fig {{ text-align: center; margin: 20px 0; }}
.summary {{ background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 10px 0; }}
</style></head><body>
<h1>🔍 Strategy Optimization Report</h1>
<div class="summary">
<p><b>Objective:</b> {self.objective} | <b>Best Score:</b> {result.best_score:.4f} | <b>Total Time:</b> {result.optimization_time:.1f}s | <b>Evaluations:</b> {len(df)}</p>
</div>

<h2>Best Parameters</h2>
<table><tr><th>Parameter</th><th>Value</th></tr>{best_params_html}</table>

<h2>Best Run Metrics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{best_metrics_html}</table>

<h2>Parameter Sensitivity</h2>
{"".join(f'<div class="fig">{f}</div>' for f in figures_html)}

<h2>All Results (Ranked)</h2>
{ranked.to_html(max_rows=200)}

</body></html>"""

        out.write_text(html, encoding="utf-8")
        logger.info(f"Report saved to {out.resolve()}")
        return str(out.resolve())

    # ── Internal helpers ─────────────────────────────────────────

    def _run_single(
        self,
        params: dict,
        strategy_factory: Callable[[dict], Strategy],
        data: pd.DataFrame,
        backtest_config: BacktestConfig,
        pipeline_fn: Callable[..., BacktestResult],
    ) -> tuple[float, dict]:
        """Run a single param combo through the pipeline and return (score, metrics)."""
        try:
            strategy = strategy_factory(params)
            result = pipeline_fn(strategy, data, backtest_config)
            score = _extract_score(result, self.objective)
            metrics = result.metrics or {}
        except Exception as e:
            logger.warning(f"Trial failed for {params}: {e}")
            score = float("-inf")
            metrics = {}
        return score, metrics

    def _evaluate_combos(
        self,
        combos: list[tuple],
        param_names: list[str],
        strategy_factory: Callable[[dict], Strategy],
        data: pd.DataFrame,
        backtest_config: BacktestConfig,
        pipeline_fn: Callable[..., BacktestResult],
    ) -> OptimizationResult:
        """Evaluate a list of param combinations sequentially."""
        rows: list[dict] = []
        t0 = time.time()

        for combo in tqdm(combos, desc=f"Optimizing ({self.objective})"):
            params = dict(zip(param_names, combo))
            score, metrics = self._run_single(
                params, strategy_factory, data, backtest_config, pipeline_fn
            )
            row = {**params, "score": score, **metrics}
            rows.append(row)
            logger.debug(f"params={params} score={score:.4f}")

        elapsed = time.time() - t0
        return self._build_result(rows, param_names, elapsed)

    @staticmethod
    def _build_result(rows: list[dict], param_names: list[str], elapsed: float) -> OptimizationResult:
        """Build OptimizationResult from evaluated rows."""
        df = pd.DataFrame(rows)
        if df.empty:
            return OptimizationResult(optimization_time=elapsed)

        best_idx = df["score"].idxmax()
        best_row = df.loc[best_idx]
        best_params = {p: best_row[p] for p in param_names}
        non_meta = set(param_names) | {"score"}
        best_metrics = {k: v for k, v in best_row.items() if k not in non_meta}

        return OptimizationResult(
            best_params=best_params,
            best_score=float(best_row["score"]),
            best_metrics=best_metrics,
            all_results=df,
            optimization_time=elapsed,
        )


def _fig_to_html(fig: Any) -> str:
    """Convert matplotlib figure to an inline HTML <img> tag."""
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" />'
