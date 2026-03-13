"""Visualization utilities for backtest results."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    title: str = "Portfolio Equity Curve",
    save_path: str | None = None,
) -> None:
    """Plot equity curve with optional benchmark."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax1 = axes[0]
    ax1.plot(equity.index, equity.values, label="Portfolio", linewidth=1.5)
    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values, label="Benchmark", linewidth=1, alpha=0.7)
    ax1.set_title(title)
    ax1.set_ylabel("Portfolio Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="red")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
