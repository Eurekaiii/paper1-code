"""Export experiment summaries and figures.

This module only consumes existing multi-seed results. It does not change
placement, scheduling, baseline, or scenario logic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import MethodMetrics, run_multi_seed_evaluation
from .scenario import build_controlled_random_hotspot_scenario_v2


METHOD_ORDER = [
    "Proposed",
    "Random Placement",
    "Importance-based Placement",
    "No-similarity Placement",
]


def summarize_multi_seed(
    metrics_by_method: Dict[str, List[MethodMetrics]],
) -> List[Dict[str, float | str]]:
    """Convert per-seed metrics into one summary row per method."""
    rows: List[Dict[str, float | str]] = []
    for method in METHOD_ORDER:
        metrics = metrics_by_method[method]
        rows.append(
            {
                "Method": method,
                "mean_D_total_ms": float(np.mean([m.D_total for m in metrics]) * 1e3),
                "std_D_total_ms": float(np.std([m.D_total for m in metrics]) * 1e3),
                "mean_Substitutions": float(np.mean([m.substitutions for m in metrics])),
                "mean_AvgCompute_ms": float(
                    np.mean([m.avg_D_compute for m in metrics]) * 1e3
                ),
                "mean_AvgTrans_ms": float(
                    np.mean([m.avg_D_trans for m in metrics]) * 1e3
                ),
            }
        )
    return rows


def save_summary_csv(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    """Save summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Method",
        "mean_D_total_ms",
        "std_D_total_ms",
        "mean_Substitutions",
        "mean_AvgCompute_ms",
        "mean_AvgTrans_ms",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def _method_labels(rows: List[Dict[str, float | str]]) -> List[str]:
    return [str(row["Method"]) for row in rows]


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_total_delay(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    """Plot mean total delay with standard-deviation error bars."""
    labels = _method_labels(rows)
    means = [float(row["mean_D_total_ms"]) for row in rows]
    stds = [float(row["std_D_total_ms"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, means, yerr=stds, capsize=5, color="#5B8DEF", edgecolor="#244A7F")
    ax.set_ylabel("Mean D_total (ms)")
    ax.set_xlabel("Method")
    ax.set_title("Total Delay on Controlled Random Hotspot v2")
    ax.tick_params(axis="x", rotation=18)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_delay_breakdown(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    """Plot mean compute and transmission delay per method."""
    labels = _method_labels(rows)
    compute = [float(row["mean_AvgCompute_ms"]) for row in rows]
    trans = [float(row["mean_AvgTrans_ms"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, compute, width, label="AvgCompute", color="#4C9A6A")
    ax.bar(x + width / 2, trans, width, label="AvgTrans", color="#D9903D")
    ax.set_ylabel("Delay (ms)")
    ax.set_xlabel("Method")
    ax.set_title("Delay Breakdown on Controlled Random Hotspot v2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_substitutions(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    """Plot mean substitution count per method."""
    labels = _method_labels(rows)
    substitutions = [float(row["mean_Substitutions"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, substitutions, color="#7A6BB7", edgecolor="#3C3565")
    ax.set_ylabel("Mean Substitutions")
    ax.set_xlabel("Method")
    ax.set_title("Substitutions on Controlled Random Hotspot v2")
    ax.tick_params(axis="x", rotation=18)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def export_controlled_random_hotspot_v2_results(
    output_dir: str | Path = "results",
    seeds: List[int] | None = None,
) -> List[Dict[str, float | str]]:
    """Run the v2 multi-seed evaluation and export CSV plus figures."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    metrics = run_multi_seed_evaluation(
        seeds=seeds,
        scenario_builder=build_controlled_random_hotspot_scenario_v2,
    )
    rows = summarize_multi_seed(metrics)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_summary_csv(rows, output_path / "controlled_random_hotspot_v2_summary.csv")
    plot_total_delay(rows, output_path / "fig_total_delay_controlled_random_v2.png")
    plot_delay_breakdown(
        rows,
        output_path / "fig_delay_breakdown_controlled_random_v2.png",
    )
    plot_substitutions(rows, output_path / "fig_substitutions_controlled_random_v2.png")
    return rows


def main() -> None:
    rows = export_controlled_random_hotspot_v2_results()
    print("Exported controlled_random_hotspot_v2 results:")
    for row in rows:
        print(
            f"  {row['Method']}: "
            f"mean_D_total={float(row['mean_D_total_ms']):.2f} ms, "
            f"std={float(row['std_D_total_ms']):.2f} ms"
        )


if __name__ == "__main__":
    main()
