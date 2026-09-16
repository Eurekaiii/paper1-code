"""Compare Proposed results under synthetic and real MoE workloads."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..plot_style import BAR_EDGE, FIG_SINGLE, apply_style, save_figure, style_axes

apply_style()

SYN_COLOR = "#8C8C8C"
REAL_COLOR = "#225EA8"


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value: str) -> float:
    text = value.strip().lower()
    if text == "inf":
        return float("inf")
    if text == "nan":
        return float("nan")
    return float(text)


def _synthetic_proposed(path: Path) -> Dict[str, str]:
    for row in _read(path):
        if row.get("Method") == "Proposed" or row.get("method") == "Proposed":
            return row
    raise ValueError(f"No Proposed row in {path}")


def _real_proposed(path: Path) -> Dict[str, str]:
    for row in _read(path):
        if row.get("method") == "Proposed" or row.get("Method") == "Proposed":
            return row
    raise ValueError(f"No Proposed row in {path}")


def _bar_pair(
    synthetic_value: float,
    real_value: float,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    labels = ["Synthetic", "Real MoE"]
    values = np.array([synthetic_value, real_value])
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bars = ax.bar(
        np.arange(2),
        values,
        width=0.52,
        color=[SYN_COLOR, REAL_COLOR],
        edgecolor=BAR_EDGE,
        linewidth=0.7,
        zorder=3,
    )
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, fontsize=14.3, fontweight="bold", pad=10)
    ax.set_ylim(0, values.max() * 1.25 if values.max() > 0 else 1.0)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + values.max() * 0.025,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
    style_axes(ax)
    save_figure(fig, output)


def plot_base_comparisons(
    synthetic_summary: Path,
    real_summary: Path,
    output_dir: Path,
) -> None:
    syn = _synthetic_proposed(synthetic_summary)
    real = _real_proposed(real_summary)

    _bar_pair(
        _f(syn["mean_D_total_ms"]),
        _f(real["D_total_ms"]),
        "Total Delay (ms)",
        "Proposed: Synthetic vs Real MoE Workload",
        output_dir / "proposed_total_delay_comparison",
    )

    labels = ["Compute", "UAV-UAV"]
    syn_vals = np.array([_f(syn["mean_AvgCompute_ms"]), _f(syn["mean_AvgTrans_ms"])])
    real_vals = np.array([_f(real["AvgCompute_ms"]), _f(real["AvgTrans_ms"])])
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.bar(
        x - width / 2,
        syn_vals,
        width,
        label="Synthetic",
        color=SYN_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.7,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        real_vals,
        width,
        label="Real MoE",
        color=REAL_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.7,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Average Delay (ms)", labelpad=8)
    ax.set_title("Proposed: Delay Components", fontsize=14.3, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_ylim(0, max(syn_vals.max(), real_vals.max()) * 1.25)
    style_axes(ax)
    save_figure(fig, output_dir / "proposed_delay_breakdown_comparison")

    _bar_pair(
        _f(syn["mean_Substitutions"]),
        _f(real["Substitutions"]),
        "Expert Substitutions",
        "Proposed: Expert Substitutions",
        output_dir / "proposed_substitutions_comparison",
    )


def _proposed_series(
    path: Path,
    value_field: str,
    delay_field: str,
    method_field: str = "method",
) -> Tuple[np.ndarray, np.ndarray]:
    rows = [
        row for row in _read(path)
        if row.get(method_field) == "Proposed" or row.get("Method") == "Proposed"
    ]
    rows.sort(key=lambda row: _f(row[value_field]))
    x = np.array([_f(row[value_field]) for row in rows])
    y = np.array([_f(row[delay_field]) for row in rows])
    finite = np.isfinite(y)
    return x[finite], y[finite]


def _line_compare(
    synthetic_csv: Path,
    real_csv: Path,
    output: Path,
    xlabel: str,
    title: str,
    synthetic_delay_field: str = "mean_D_total_ms",
    real_delay_field: str = "D_total_ms",
) -> None:
    sx, sy = _proposed_series(
        synthetic_csv,
        value_field="value",
        delay_field=synthetic_delay_field,
    )
    rx, ry = _proposed_series(
        real_csv,
        value_field="value",
        delay_field=real_delay_field,
    )
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(
        sx,
        sy,
        label="Synthetic",
        color=SYN_COLOR,
        marker="s",
        linestyle="--",
        linewidth=2,
        zorder=3,
    )
    ax.plot(
        rx,
        ry,
        label="Real MoE",
        color=REAL_COLOR,
        marker="o",
        linestyle="-",
        linewidth=2,
        zorder=3,
    )
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel("Total Delay (ms)", labelpad=8)
    ax.set_title(title, fontsize=14.3, fontweight="bold", pad=10)
    ax.legend(loc="best", fontsize=12)
    style_axes(ax)
    save_figure(fig, output)


def plot_sweep_comparisons(
    synthetic_results: Path,
    real_results: Path,
    output_dir: Path,
) -> None:
    _line_compare(
        synthetic_results / "sensitivity_xi.csv",
        real_results / "real_world_sensitivity_xi.csv",
        output_dir / "proposed_sensitivity_xi_comparison",
        "Similarity Threshold xi",
        "Proposed: xi Sensitivity",
    )
    _line_compare(
        synthetic_results / "sensitivity_mid_size.csv",
        real_results / "real_world_sensitivity_mid_size.csv",
        output_dir / "proposed_sensitivity_mid_size_comparison",
        "Intermediate Feature Size Scale",
        "Proposed: Mid-feature Size Sensitivity",
    )
    _line_compare(
        synthetic_results / "sensitivity_memory.csv",
        real_results / "real_world_sensitivity_memory.csv",
        output_dir / "proposed_sensitivity_memory_comparison",
        "UAV Memory Scale",
        "Proposed: Memory Sensitivity",
    )
    _line_compare(
        synthetic_results / "scalability_summary.csv",
        real_results / "real_world_uav_count.csv",
        output_dir / "proposed_uav_count_comparison",
        "Number of UAVs",
        "Proposed: UAV Count",
        synthetic_delay_field="mean_D_total_ms",
    )
    _line_compare(
        synthetic_results / "scalability_summary.csv",
        real_results / "real_world_task_count.csv",
        output_dir / "proposed_task_count_comparison",
        "Number of Tasks",
        "Proposed: Task Count",
        synthetic_delay_field="mean_D_total_ms",
    )
    _line_compare(
        synthetic_results / "scalability_summary.csv",
        real_results / "real_world_expert_count.csv",
        output_dir / "proposed_expert_count_comparison",
        "Number of Experts",
        "Proposed: Expert Count",
        synthetic_delay_field="mean_D_total_ms",
    )


def plot_all(
    synthetic_results: str | Path = "results",
    real_results: str | Path = "results/real_world/fig1_9",
    output_dir: str | Path = "results/synthetic_vs_real_moe",
) -> None:
    syn_root = Path(synthetic_results)
    real_root = Path(real_results)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_base_comparisons(
        syn_root / "controlled_random_hotspot_v2_summary.csv",
        real_root / "real_world_base.csv",
        out,
    )
    plot_sweep_comparisons(syn_root, real_root, out)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Proposed under synthetic and real MoE workloads.")
    parser.add_argument("--synthetic-results", default="results")
    parser.add_argument("--real-results", default="results/real_world/fig1_9")
    parser.add_argument("--output-dir", default="results/synthetic_vs_real_moe")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    plot_all(
        synthetic_results=args.synthetic_results,
        real_results=args.real_results,
        output_dir=args.output_dir,
    )
    print(f"Wrote synthetic-vs-real Proposed figures to {args.output_dir}")


if __name__ == "__main__":
    main()

