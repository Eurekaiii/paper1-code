"""Plot sensitivity figures from saved CSV files.

This module reads existing CSV outputs only. It does not rerun experiments or
modify any algorithm, baseline, or scenario code.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "Proposed": "Proposed",
    "Random Placement": "Random",
    "Importance-based Placement": "Importance",
    "No-similarity Placement": "No-Sim",
}

METHOD_ORDER = [
    "Proposed",
    "Random Placement",
    "Importance-based Placement",
    "No-similarity Placement",
]

COLORS = {
    "Proposed": "#2F6DB3",
    "Random Placement": "#8E8E8E",
    "Importance-based Placement": "#D9903D",
    "No-similarity Placement": "#7A6BB7",
}

MARKERS = {
    "Proposed": "o",
    "Random Placement": "s",
    "Importance-based Placement": "^",
    "No-similarity Placement": "D",
}


def _to_float(value: str) -> float:
    if value.lower() == "inf":
        return float("inf")
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_sensitivity(
    csv_path: Path,
    output_path: Path,
    xlabel: str,
    title: str,
    min_x: float | None = None,
    mark_infeasible: bool = False,
) -> None:
    rows = _read_rows(csv_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    infeasible_points: List[tuple[float, str]] = []

    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: _to_float(row["value"]))

        x_values: List[float] = []
        y_values: List[float] = []
        std_values: List[float] = []

        for row in method_rows:
            x = _to_float(row["value"])
            mean = _to_float(row["mean_D_total_ms"])
            std = _to_float(row["std_D_total_ms"])
            if min_x is not None and x < min_x:
                if mark_infeasible and not np.isfinite(mean):
                    infeasible_points.append((x, method))
                continue
            if not np.isfinite(mean):
                if mark_infeasible:
                    infeasible_points.append((x, method))
                continue
            x_values.append(x)
            y_values.append(mean)
            std_values.append(std if np.isfinite(std) else 0.0)

        if not x_values:
            continue

        ax.errorbar(
            x_values,
            y_values,
            yerr=std_values,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.0,
            markersize=5.5,
            capsize=3.5,
        )

    if mark_infeasible and infeasible_points:
        y_top = ax.get_ylim()[1]
        for x, method in infeasible_points:
            ax.annotate(
                "infeasible",
                xy=(x, y_top * 0.96),
                xytext=(4, -2),
                textcoords="offset points",
                fontsize=8,
                color=COLORS.get(method, "#333333"),
                rotation=90,
                va="top",
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean D_total (ms)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_all_sensitivity_figures(results_dir: str | Path = "results") -> None:
    """Create Figures 4-6 from sensitivity CSV files."""
    root = Path(results_dir)
    _plot_sensitivity(
        csv_path=root / "sensitivity_xi.csv",
        output_path=root / "fig4_sensitivity_xi.png",
        xlabel="Similarity threshold ξ",
        title="Effect of Similarity Threshold",
    )
    _plot_sensitivity(
        csv_path=root / "sensitivity_mid_size.csv",
        output_path=root / "fig5_sensitivity_mid_size.png",
        xlabel="S_mid scale",
        title="Effect of Intermediate Feature Size",
    )
    _plot_sensitivity(
        csv_path=root / "sensitivity_memory.csv",
        output_path=root / "fig6_sensitivity_memory.png",
        xlabel="Memory scale",
        title="Effect of UAV Memory Capacity",
        min_x=1.0,
        mark_infeasible=True,
    )


def main() -> None:
    plot_all_sensitivity_figures()
    print("Saved sensitivity figures:")
    print("  results/fig4_sensitivity_xi.png")
    print("  results/fig5_sensitivity_mid_size.png")
    print("  results/fig6_sensitivity_memory.png")


if __name__ == "__main__":
    main()
