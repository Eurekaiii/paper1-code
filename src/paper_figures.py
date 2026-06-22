"""Generate publication-style figures from saved experiment CSV files.

This module only reads CSV files under results/. It does not rerun simulations
or modify algorithm, baseline, scheduling, or scenario logic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = [
    "Proposed",
    "Random Placement",
    "Importance-based Placement",
    "No-similarity Placement",
]

METHOD_LABELS = {
    "Proposed": "Proposed",
    "Random Placement": "Random",
    "Importance-based Placement": "Importance",
    "No-similarity Placement": "No-Sim",
}

COLORS = {
    "Proposed": "#2F5F8F",
    "Random Placement": "#8A8A8A",
    "Importance-based Placement": "#B36B2C",
    "No-similarity Placement": "#6E5C9A",
}

MARKERS = {
    "Proposed": "o",
    "Random Placement": "s",
    "Importance-based Placement": "^",
    "No-similarity Placement": "D",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csvfile:
        return list(csv.DictReader(csvfile))


def _to_float(value: str) -> float:
    text = value.strip().lower()
    if text == "inf":
        return float("inf")
    if text == "nan":
        return float("nan")
    return float(value)


def _ordered_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_method = {row["Method"]: row for row in rows}
    return [by_method[method] for method in METHOD_ORDER if method in by_method]


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _method_xticks(ax: plt.Axes, labels: List[str]) -> None:
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")


def plot_fig1_total_delay(summary_csv: Path, output_base: Path) -> None:
    rows = _ordered_rows(_read_csv(summary_csv))
    labels = [METHOD_LABELS[row["Method"]] for row in rows]
    means = [_to_float(row["mean_D_total_ms"]) for row in rows]
    stds = [_to_float(row["std_D_total_ms"]) for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=[COLORS[row["Method"]] for row in rows],
        edgecolor="#333333",
        linewidth=0.7,
    )
    _method_xticks(ax, labels)
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean total delay (ms)")
    _style_axes(ax)
    _save(fig, output_base)


def plot_fig2_delay_breakdown(summary_csv: Path, output_base: Path) -> None:
    rows = _ordered_rows(_read_csv(summary_csv))
    labels = [METHOD_LABELS[row["Method"]] for row in rows]
    compute = [_to_float(row["mean_AvgCompute_ms"]) for row in rows]
    trans = [_to_float(row["mean_AvgTrans_ms"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(
        x - width / 2,
        compute,
        width,
        label="Computation",
        color="#4F7F5F",
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        trans,
        width,
        label="Transmission",
        color="#C28A45",
        edgecolor="#333333",
        linewidth=0.6,
    )
    _method_xticks(ax, labels)
    ax.set_xlabel("Method")
    ax.set_ylabel("Delay (ms)")
    ax.legend(frameon=False, fontsize=10)
    _style_axes(ax)
    _save(fig, output_base)


def plot_fig3_substitutions(summary_csv: Path, output_base: Path) -> None:
    rows = _ordered_rows(_read_csv(summary_csv))
    labels = [METHOD_LABELS[row["Method"]] for row in rows]
    substitutions = [_to_float(row["mean_Substitutions"]) for row in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(
        x,
        substitutions,
        color=[COLORS[row["Method"]] for row in rows],
        edgecolor="#333333",
        linewidth=0.7,
    )
    _method_xticks(ax, labels)
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean substitutions")
    _style_axes(ax)
    _save(fig, output_base)


def _plot_sensitivity(
    csv_path: Path,
    output_base: Path,
    xlabel: str,
) -> None:
    rows = _read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for method in METHOD_ORDER:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: _to_float(row["value"]))

        xs: List[float] = []
        means: List[float] = []
        stds: List[float] = []
        for row in method_rows:
            x = _to_float(row["value"])
            mean = _to_float(row["mean_D_total_ms"])
            std = _to_float(row["std_D_total_ms"])
            if not np.isfinite(mean):
                continue
            xs.append(x)
            means.append(mean)
            stds.append(std if np.isfinite(std) else 0.0)

        if not xs:
            continue

        ax.errorbar(
            xs,
            means,
            yerr=stds,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.9,
            markersize=5.2,
            capsize=3,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean total delay (ms)")
    ax.legend(frameon=False, fontsize=9)
    _style_axes(ax)
    _save(fig, output_base)


def plot_all_paper_figures(results_dir: str | Path = "results") -> None:
    root = Path(results_dir)
    plot_fig1_total_delay(
        root / "controlled_random_hotspot_v2_summary.csv",
        root / "fig1_total_delay",
    )
    plot_fig2_delay_breakdown(
        root / "controlled_random_hotspot_v2_summary.csv",
        root / "fig2_delay_breakdown",
    )
    plot_fig3_substitutions(
        root / "controlled_random_hotspot_v2_summary.csv",
        root / "fig3_substitutions",
    )
    _plot_sensitivity(
        root / "sensitivity_xi.csv",
        root / "fig4_sensitivity_xi",
        "Similarity threshold ξ",
    )
    _plot_sensitivity(
        root / "sensitivity_mid_size.csv",
        root / "fig5_sensitivity_mid_size",
        "Intermediate feature size scale",
    )
    _plot_sensitivity(
        root / "sensitivity_memory.csv",
        root / "fig6_sensitivity_memory",
        "UAV memory capacity scale",
    )


def main() -> None:
    plot_all_paper_figures()
    print("Saved paper figures fig1-fig6 as PNG and PDF in results/.")


if __name__ == "__main__":
    main()
