"""Figures for scalability and spatial robustness experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import (
    COLORS,
    MARKERS,
    METHOD_LABELS,
    METHOD_ORDER,
    apply_style,
    save_figure,
    style_axes,
)


apply_style()


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(raw: str) -> float:
    text = raw.strip().lower()
    if text == "inf":
        return float("inf")
    if text == "nan":
        return float("nan")
    return float(text)


def _plot_metric_by_experiment(
    ax: plt.Axes,
    rows: List[Dict[str, str]],
    experiment: str,
    metric: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    exp_rows = [row for row in rows if row["experiment"] == experiment]
    infeasible_points: Dict[str, List[float]] = {}
    for method in METHOD_ORDER:
        method_rows = [row for row in exp_rows if row["method"] == method]
        method_rows.sort(key=lambda row: _to_float(row["value"]))

        xs: List[float] = []
        ys: List[float] = []
        for row in method_rows:
            y = _to_float(row[metric])
            if not np.isfinite(y):
                infeasible_points.setdefault(method, []).append(_to_float(row["value"]))
                continue
            xs.append(_to_float(row["value"]))
            ys.append(y)

        if not xs:
            continue

        ax.plot(
            xs,
            ys,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.0,
            markersize=5.8,
            markeredgewidth=0.5,
            markeredgecolor="#333333",
            zorder=3,
        )

    if infeasible_points:
        y_min, y_max = ax.get_ylim()
        y_marker = y_min + (y_max - y_min) * 0.95
        for method, xs in infeasible_points.items():
            ax.scatter(
                xs,
                [y_marker] * len(xs),
                color=COLORS[method],
                marker="x",
                s=44,
                linewidths=1.4,
                zorder=4,
            )
        ax.text(
            0.98,
            0.95,
            "x: infeasible",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#444444",
        )

    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    style_axes(ax)


def plot_scalability_summary(
    csv_path: str | Path = "results/scalability_summary.csv",
    output_base: str | Path = "results/fig7_scalability",
) -> None:
    """Create the four-panel scalability summary figure."""
    rows = _read_rows(Path(csv_path))

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    ax_uav, ax_task, ax_area, ax_runtime = axes.ravel()

    _plot_metric_by_experiment(
        ax_uav,
        rows,
        "uav_count",
        "mean_D_per_task_ms",
        xlabel="Number of UAVs",
        ylabel="Mean Delay per Task (ms)",
        title="Varying UAV Count",
    )
    _plot_metric_by_experiment(
        ax_task,
        rows,
        "task_count",
        "mean_D_per_task_ms",
        xlabel="Number of Tasks",
        ylabel="Mean Delay per Task (ms)",
        title="Varying Task Count",
    )
    _plot_metric_by_experiment(
        ax_area,
        rows,
        "area_size",
        "mean_D_per_task_ms",
        xlabel="Area Size (m x m)",
        ylabel="Mean Delay per Task (ms)",
        title="Varying Deployment Area",
    )
    _plot_metric_by_experiment(
        ax_runtime,
        rows,
        "task_count",
        "mean_runtime_ms",
        xlabel="Number of Tasks",
        ylabel="Runtime (ms)",
        title="Runtime Overhead",
    )

    for idx, ax in enumerate(axes.ravel()):
        ax.text(
            -0.10,
            1.03,
            f"({chr(97 + idx)})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    handles, labels = ax_uav.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle(
        "Scalability and Spatial Robustness",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.97))
    save_figure(fig, Path(output_base))


def main() -> None:
    plot_scalability_summary()
    print("Saved scalability figure: results/fig7_scalability.{png,pdf}")


if __name__ == "__main__":
    main()
