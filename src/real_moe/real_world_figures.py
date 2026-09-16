"""Figures for real MoE trace-driven UAV experiments."""

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
    FIG_WIDE_2,
    METHOD_LABELS,
    METHOD_ORDER,
    apply_style,
    label_bars,
    save_figure,
    style_axes,
)

apply_style()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str) -> float:
    text = value.strip().lower()
    if text == "inf":
        return float("inf")
    if text == "nan":
        return float("nan")
    return float(text)


def _ordered_rows(summary_csv: Path) -> List[Dict[str, str]]:
    rows = _read_csv(summary_csv)
    by_method = {row["Method"]: row for row in rows}
    return [by_method[method] for method in METHOD_ORDER if method in by_method]


def _labels(rows: List[Dict[str, str]]) -> List[str]:
    return [METHOD_LABELS.get(row["Method"], row["Method"]) for row in rows]


def plot_total_delay(summary_csv: Path, output_base: Path, title: str) -> None:
    rows = _ordered_rows(summary_csv)
    values = np.array([_to_float(row["D_total_ms"]) for row in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.bar(
        x,
        values,
        width=0.55,
        color=[COLORS[row["Method"]] for row in rows],
        edgecolor=BAR_EDGE,
        linewidth=0.7,
        zorder=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(_labels(rows), rotation=0, ha="center")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel("Total Delay (ms)", labelpad=8)
    ax.set_title(title, fontsize=14.3, fontweight="bold", pad=10)
    ax.set_ylim(0, values.max() * 1.25)
    label_bars(ax, fmt=".1f")
    style_axes(ax)
    save_figure(fig, output_base)


def plot_delay_breakdown(summary_csv: Path, output_base: Path, title: str) -> None:
    rows = _ordered_rows(summary_csv)
    labels = _labels(rows)
    access = np.array([_to_float(row["AvgAccess_ms"]) for row in rows])
    compute = np.array([_to_float(row["AvgCompute_ms"]) for row in rows])
    trans = np.array([_to_float(row["AvgTrans_ms"]) for row in rows])
    ret = np.array([_to_float(row["AvgReturn_ms"]) for row in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    bottom = np.zeros(len(rows))
    segments = [
        ("Access", access, "#6F8FBF"),
        ("Compute", compute, "#4F7F5F"),
        ("UAV-UAV", trans, "#C28A45"),
        ("Return", ret, "#9B6BA8"),
    ]
    for label, values, color in segments:
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
    ax.set_title(title, fontsize=14.3, fontweight="bold", pad=10)
    ax.set_ylim(0, bottom.max() * 1.25)
    ax.legend(loc="upper right", ncols=2, fontsize=11)
    style_axes(ax)
    save_figure(fig, output_base)


def plot_substitutions_deployments(
    summary_csv: Path,
    output_base: Path,
    title: str,
) -> None:
    rows = _ordered_rows(summary_csv)
    labels = _labels(rows)
    substitutions = np.array([_to_float(row["Substitutions"]) for row in rows])
    deployments = np.array([_to_float(row["Deployments"]) for row in rows])
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    b1 = ax.bar(
        x - width / 2,
        substitutions,
        width,
        label="Substitutions",
        color="#4F7F5F",
        edgecolor=BAR_EDGE,
        linewidth=0.7,
        zorder=3,
    )
    b2 = ax.bar(
        x + width / 2,
        deployments,
        width,
        label="Deployments",
        color="#C28A45",
        edgecolor=BAR_EDGE,
        linewidth=0.7,
        zorder=3,
    )
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + max(substitutions.max(), deployments.max()) * 0.02,
                f"{h:.0f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel("Count", labelpad=8)
    ax.set_title(title, fontsize=14.3, fontweight="bold", pad=10)
    ax.set_ylim(0, max(substitutions.max(), deployments.max()) * 1.28)
    ax.legend(loc="upper right", fontsize=12)
    style_axes(ax)
    save_figure(fig, output_base)


def plot_scale_comparison(results_dir: Path, output_base: Path) -> None:
    scenarios = [
        ("Raw", results_dir / "raw" / "real_moe_summary.csv"),
        ("Scaled", results_dir / "expert_scale_80" / "real_moe_summary.csv"),
    ]
    rows_by_scenario = [(name, _ordered_rows(path)) for name, path in scenarios]
    methods = [row["Method"] for row in rows_by_scenario[0][1]]
    x = np.arange(len(scenarios))
    width = 0.18

    fig, ax = plt.subplots(figsize=FIG_WIDE_2)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods):
        values = []
        for _scenario_name, rows in rows_by_scenario:
            row = next(item for item in rows if item["Method"] == method)
            values.append(_to_float(row["D_total_ms"]))
        ax.bar(
            x + offset,
            values,
            width,
            label=METHOD_LABELS.get(method, method),
            color=COLORS[method],
            edgecolor=BAR_EDGE,
            linewidth=0.7,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Original experts", "Scaled experts"], ha="center")
    ax.set_xlabel("Real MoE Trace Scenario", labelpad=6)
    ax.set_ylabel("Total Delay (ms)", labelpad=8)
    ax.set_title("Real MoE Trace: Total Delay Under Memory Pressure",
                 fontsize=14.3, fontweight="bold", pad=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.legend(loc="upper left", fontsize=12)
    style_axes(ax)
    save_figure(fig, output_base)


def plot_all_real_world_figures(
    results_dir: str | Path = "results/real_world",
    output_dir: str | Path = "results/real_world/figures",
) -> None:
    results = Path(results_dir)
    out = Path(output_dir)
    scenarios = [
        (
            "raw",
            "Real MoE Trace: Original Expert Size",
            results / "raw" / "real_moe_summary.csv",
        ),
        (
            "expert_scale_80",
            "Real MoE Trace: Scaled Expert Memory",
            results / "expert_scale_80" / "real_moe_summary.csv",
        ),
    ]
    for slug, title, summary_csv in scenarios:
        plot_total_delay(summary_csv, out / f"{slug}_total_delay", title)
        plot_delay_breakdown(summary_csv, out / f"{slug}_delay_breakdown", title)
        plot_substitutions_deployments(
            summary_csv,
            out / f"{slug}_substitutions_deployments",
            title,
        )
    plot_scale_comparison(results, out / "scenario_total_delay_comparison")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot real-world MoE experiment figures.")
    parser.add_argument("--results-dir", default="results/real_world")
    parser.add_argument("--output-dir", default="results/real_world/figures")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    plot_all_real_world_figures(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
    print(f"Wrote real-world figures to {args.output_dir}")


if __name__ == "__main__":
    main()

