"""Generate fig1-fig9 for real MoE trace-driven experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..plot_style import (
    BAR_EDGE,
    COLORS,
    FIG_SINGLE,
    MARKERS,
    METHOD_LABELS,
    METHOD_ORDER,
    apply_style,
    label_bars,
    save_figure,
    style_axes,
)

apply_style()

LINESTYLES = {
    "Proposed": "-",
    "Random Placement": "--",
    "Importance-based Placement": "-.",
    "No-similarity Placement": ":",
}


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


def _ordered(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_method = {row["method"]: row for row in rows}
    return [by_method[method] for method in METHOD_ORDER if method in by_method]


def _labels(rows: List[Dict[str, str]]) -> List[str]:
    return [METHOD_LABELS.get(row["method"], row["method"]) for row in rows]


def _base_bar(
    rows: List[Dict[str, str]],
    field: str,
    ylabel: str,
    title: str,
    output: Path,
    fmt: str = ".1f",
) -> None:
    rows = _ordered(rows)
    values = np.array([_f(row[field]) for row in rows])
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.bar(
        x,
        values,
        width=0.55,
        color=[COLORS[row["method"]] for row in rows],
        edgecolor=BAR_EDGE,
        linewidth=0.7,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(rows), rotation=0, ha="center")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, fontsize=14.3, fontweight="bold", pad=10)
    ax.set_ylim(0, values.max() * 1.25 if values.max() > 0 else 1.0)
    label_bars(ax, fmt=fmt, include_zero=True)
    style_axes(ax)
    save_figure(fig, output)


def plot_fig1_total_delay(base_csv: Path, output: Path) -> None:
    _base_bar(
        _read(base_csv),
        "D_total_ms",
        "Total Delay (ms)",
        "Fig. 1 Real MoE: Total Delay",
        output,
    )


def plot_fig2_delay_breakdown(base_csv: Path, output: Path) -> None:
    rows = _ordered(_read(base_csv))
    x = np.arange(len(rows))
    labels = _labels(rows)
    access = np.array([_f(row["AvgAccess_ms"]) for row in rows])
    compute = np.array([_f(row["AvgCompute_ms"]) for row in rows])
    trans = np.array([_f(row["AvgTrans_ms"]) for row in rows])
    ret = np.array([_f(row["AvgReturn_ms"]) for row in rows])

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bottom = np.zeros(len(rows))
    for label, values, color in [
        ("Access", access, "#6F8FBF"),
        ("Compute", compute, "#4F7F5F"),
        ("UAV-UAV", trans, "#C28A45"),
        ("Return", ret, "#9B6BA8"),
    ]:
        ax.bar(
            x,
            values,
            width=0.55,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor=BAR_EDGE,
            linewidth=0.45,
            zorder=3,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel("Average Per-task Delay (ms)", labelpad=8)
    ax.set_title("Fig. 2 Real MoE: Delay Breakdown", fontsize=14.3, fontweight="bold", pad=10)
    ax.set_ylim(0, bottom.max() * 1.25)
    ax.legend(loc="upper right", ncols=2, fontsize=11)
    style_axes(ax)
    save_figure(fig, output)


def plot_fig3_substitutions(base_csv: Path, output: Path) -> None:
    _base_bar(
        _read(base_csv),
        "Substitutions",
        "Expert Substitutions",
        "Fig. 3 Real MoE: Substitutions",
        output,
        fmt=".0f",
    )


def _line_plot(
    csv_path: Path,
    output: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    field: str = "D_total_ms",
) -> None:
    rows = _read(csv_path)
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    for method in METHOD_ORDER:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: _f(row["value"]),
        )
        if not method_rows:
            continue
        x = np.array([_f(row["value"]) for row in method_rows])
        y = np.array([_f(row[field]) for row in method_rows])
        finite = np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size == 0:
            continue
        ax.plot(
            x,
            y,
            label=METHOD_LABELS.get(method, method),
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.0,
            markersize=6,
            zorder=3,
        )

    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.set_title(title, fontsize=14.3, fontweight="bold", pad=10)
    ax.legend(loc="best", fontsize=11)
    style_axes(ax)
    save_figure(fig, output)


def plot_all_fig1_9(
    csv_dir: str | Path = "results/real_world/fig1_9",
    output_dir: str | Path = "results/real_world/fig1_9/figures",
) -> None:
    root = Path(csv_dir)
    out = Path(output_dir)
    plot_fig1_total_delay(root / "real_world_base.csv", out / "fig1_total_delay")
    plot_fig2_delay_breakdown(root / "real_world_base.csv", out / "fig2_delay_breakdown")
    plot_fig3_substitutions(root / "real_world_base.csv", out / "fig3_substitutions")
    _line_plot(
        root / "real_world_sensitivity_xi.csv",
        out / "fig4_sensitivity_xi",
        "Similarity Threshold xi",
        "Total Delay (ms)",
        "Fig. 4 Real MoE: Similarity Threshold",
    )
    _line_plot(
        root / "real_world_sensitivity_mid_size.csv",
        out / "fig5_sensitivity_mid_size",
        "Intermediate Feature Size Scale",
        "Total Delay (ms)",
        "Fig. 5 Real MoE: Intermediate Feature Size",
    )
    _line_plot(
        root / "real_world_sensitivity_memory.csv",
        out / "fig6_sensitivity_memory",
        "UAV Memory Scale",
        "Total Delay (ms)",
        "Fig. 6 Real MoE: UAV Memory",
    )
    _line_plot(
        root / "real_world_uav_count.csv",
        out / "fig7_uav_count",
        "Number of UAVs",
        "Total Delay (ms)",
        "Fig. 7 Real MoE: UAV Count",
    )
    _line_plot(
        root / "real_world_task_count.csv",
        out / "fig8_task_count",
        "Number of Tasks",
        "Total Delay (ms)",
        "Fig. 8 Real MoE: Task Count",
    )
    _line_plot(
        root / "real_world_expert_count.csv",
        out / "fig9_expert_count",
        "Number of Experts",
        "Total Delay (ms)",
        "Fig. 9 Real MoE: Expert Count",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate real MoE fig1-fig9 plots.")
    parser.add_argument("--csv-dir", default="results/real_world/fig1_9")
    parser.add_argument("--output-dir", default="results/real_world/fig1_9/figures")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    plot_all_fig1_9(csv_dir=args.csv_dir, output_dir=args.output_dir)
    print(f"Wrote real-world fig1-fig9 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
