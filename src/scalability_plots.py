"""Figures for scalability and spatial robustness experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import (
    COLORS,
    FIG_SINGLE,
    MARKERS,
    METHOD_LABELS,
    METHOD_ORDER,
    apply_style,
    save_figure,
    style_axes,
)


apply_style()

LINESTYLES: Dict[str, str] = {
    "Proposed": "-",
    "Random Placement": "--",
    "Importance-based Placement": "-.",
    "No-similarity Placement": ":",
}


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
    x_values = sorted({_to_float(row["value"]) for row in exp_rows})
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
            linestyle=LINESTYLES[method],
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
    if x_values:
        ax.set_xticks(x_values)
        ax.set_xlim(x_values[0], x_values[-1])
    style_axes(ax)


def plot_scalability_summary(
    csv_path: str | Path = "results/scalability_summary.csv",
    output_base: str | Path = "results/fig7_scalability",
) -> None:
    """Create the four-panel scalability summary figure."""
    rows = _read_rows(Path(csv_path))

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.2))
    ax_uav, ax_task, ax_area_delay, ax_area_success, ax_full, ax_runtime = axes.ravel()

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
        ax_area_delay,
        rows,
        "area_size",
        "mean_D_per_task_ms",
        xlabel="Area Size (m x m)",
        ylabel="Mean Delay per Task (ms)",
        title="Fixed Resources: Area Expansion",
    )
    _plot_metric_by_experiment(
        ax_area_success,
        rows,
        "area_size",
        "success_rate",
        xlabel="Area Size (m x m)",
        ylabel="Success Rate",
        title="Fixed Resources: Feasibility",
    )
    ax_area_success.set_ylim(-0.05, 1.05)
    _plot_metric_by_experiment(
        ax_full,
        rows,
        "full_scale_area",
        "mean_D_per_task_ms",
        xlabel="Area Size (m x m)",
        ylabel="Mean Delay per Task (ms)",
        title="Full System Scaling",
    )
    _plot_metric_by_experiment(
        ax_runtime,
        rows,
        "task_count",
        "mean_runtime_ms",
        xlabel="Number of Tasks",
        ylabel="Runtime (ms)",
        title="Planning Runtime",
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


def _save_single_metric_figure(
    rows: List[Dict[str, str]],
    experiment: str,
    metric: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_base: str | Path,
) -> None:
    """Save a single-panel line figure for one scalability sweep."""
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    _plot_metric_by_experiment(
        ax,
        rows,
        experiment,
        metric,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0))
    save_figure(fig, Path(output_base))


def plot_uav_count_scalability(
    csv_path: str | Path = "results/scalability_summary.csv",
    output_base: str | Path = "results/fig7_uav_count",
) -> None:
    """Save the UAV-count scalability figure."""
    rows = _read_rows(Path(csv_path))
    _save_single_metric_figure(
        rows,
        "uav_count",
        "mean_D_per_task_ms",
        xlabel="Number of UAVs",
        ylabel="Mean Delay per Task (ms)",
        title="Scalability with UAV Count",
        output_base=output_base,
    )


def plot_task_count_scalability(
    csv_path: str | Path = "results/scalability_summary.csv",
    output_base: str | Path = "results/fig8_task_count",
) -> None:
    """Save the task-count scalability figure."""
    rows = _read_rows(Path(csv_path))
    _save_single_metric_figure(
        rows,
        "task_count",
        "mean_D_per_task_ms",
        xlabel="Number of Tasks",
        ylabel="Mean Delay per Task (ms)",
        title="Scalability with Task Count",
        output_base=output_base,
    )


def plot_expert_scalability(
    csv_path: str | Path = "results/scalability_summary.csv",
    output_base: str | Path = "results/fig9_expert_count",
) -> None:
    """Save the expert-count delay scalability figure."""
    rows = _read_rows(Path(csv_path))
    _save_single_metric_figure(
        rows,
        "expert_count",
        "mean_D_per_task_ms",
        xlabel="Number of Experts",
        ylabel="Mean Delay per Task (ms)",
        title="Expert Pool Scalability",
        output_base=output_base,
    )


def plot_area_scalability(
    csv_path: str | Path = "results/scalability_summary.csv",
    output_base: str | Path = "results/fig10_area_scaling",
) -> None:
    """Save the deployment-area scalability figure."""
    rows = _read_rows(Path(csv_path))
    fig, axes = plt.subplots(1, 3, figsize=FIG_SINGLE)
    ax_fixed_delay, ax_fixed_success, ax_full_delay = axes

    _plot_metric_by_experiment(
        ax_fixed_delay,
        rows,
        "area_size",
        "mean_D_per_task_ms",
        xlabel="Area Size (m x m)",
        ylabel="Mean Delay per Task (ms)",
        title="Fixed Resources: Delay",
    )
    _plot_metric_by_experiment(
        ax_fixed_success,
        rows,
        "area_size",
        "success_rate",
        xlabel="Area Size (m x m)",
        ylabel="Success Rate",
        title="Fixed Resources: Feasibility",
    )
    ax_fixed_success.set_ylim(-0.05, 1.05)
    _plot_metric_by_experiment(
        ax_full_delay,
        rows,
        "full_scale_area",
        "mean_D_per_task_ms",
        xlabel="Area Size (m x m)",
        ylabel="Mean Delay per Task (ms)",
        title="Full System Scaling",
    )

    for idx, ax in enumerate(axes):
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

    handles, labels = ax_fixed_delay.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    fig.suptitle("Deployment Area Scalability", fontsize=13, fontweight="bold", y=0.96)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.90))
    save_figure(fig, Path(output_base))


def plot_advantage_scenario(
    csv_path: str | Path = "results/scalability_summary.csv",
    output_base: str | Path = "results/fig11_advantage_scenario",
) -> None:
    """Save the resource-constrained heterogeneous-demand advantage figure."""
    rows = _read_rows(Path(csv_path))
    fig, axes = plt.subplots(1, 2, figsize=FIG_SINGLE)
    ax_delay, ax_success = axes

    _plot_metric_by_experiment(
        ax_delay,
        rows,
        "advantage_memory",
        "mean_D_per_task_ms",
        xlabel="UAV Memory Scale",
        ylabel="Mean Delay per Task (ms)",
        title="Delay under Tight Expert Memory",
    )
    _plot_metric_by_experiment(
        ax_success,
        rows,
        "advantage_memory",
        "success_rate",
        xlabel="UAV Memory Scale",
        ylabel="Success Rate",
        title="Feasibility under Tight Expert Memory",
    )
    ax_success.set_ylim(-0.05, 1.05)

    for idx, ax in enumerate(axes):
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

    handles, labels = ax_delay.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.01),
        frameon=False,
    )
    fig.suptitle(
        "Advantage Scenario: Heterogeneous Experts with Limited UAV Memory",
        fontsize=13,
        fontweight="bold",
        y=0.96,
    )
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.90))
    save_figure(fig, Path(output_base))


def main() -> None:
    plot_uav_count_scalability()
    plot_task_count_scalability()
    plot_expert_scalability()
    plot_area_scalability()
    plot_advantage_scenario()
    print("Saved UAV-count figure: results/fig7_uav_count.{png,pdf}")
    print("Saved task-count figure: results/fig8_task_count.{png,pdf}")
    print("Saved expert-count figure: results/fig9_expert_count.{png,pdf}")
    print("Saved area-scaling figure: results/fig10_area_scaling.{png,pdf}")
    print("Saved advantage-scenario figure: results/fig11_advantage_scenario.{png,pdf}")


if __name__ == "__main__":
    main()
