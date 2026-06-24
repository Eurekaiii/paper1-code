"""Export experiment summaries and basic figures.

Uses the shared style module so all outputs are visually consistent.
For the polished publication figures, prefer :mod:`.paper_figures`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .evaluation import MethodMetrics, run_multi_seed_evaluation
from .scenario import build_controlled_random_hotspot_scenario_v2
from .plot_style import (
    COLORS,
    FIG_SINGLE,
    METHOD_LABELS,
    METHOD_ORDER,
    apply_style,
    label_bars,
    save_figure,
    style_axes,
)

apply_style()

# ═══════════════════════════════════════════════════════════════════════════
#  CSV summary
# ═══════════════════════════════════════════════════════════════════════════

def summarize_multi_seed(
    metrics_by_method: Dict[str, List[MethodMetrics]],
) -> List[Dict[str, float | str]]:
    """Convert per-seed metrics into one summary row per method."""
    rows: List[Dict[str, float | str]] = []
    for method in METHOD_ORDER:
        metrics = metrics_by_method[method]
        rows.append({
            "Method":               method,
            "mean_D_total_ms":      float(np.mean([m.D_total for m in metrics]) * 1e3),
            "std_D_total_ms":       float(np.std([m.D_total for m in metrics]) * 1e3),
            "mean_Substitutions":   float(np.mean([m.substitutions for m in metrics])),
            "mean_AvgCompute_ms":   float(np.mean([m.avg_D_compute for m in metrics]) * 1e3),
            "mean_AvgTrans_ms":     float(np.mean([m.avg_D_trans for m in metrics]) * 1e3),
        })
    return rows


def save_summary_csv(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    """Save summary rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Method", "mean_D_total_ms", "std_D_total_ms",
        "mean_Substitutions", "mean_AvgCompute_ms", "mean_AvgTrans_ms",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: f"{v:.6f}" if isinstance(v, float) else v
                for k, v in row.items()
            })


# ═══════════════════════════════════════════════════════════════════════════
#  Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_total_delay(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    """Bar chart: mean total delay per method."""
    ordered = sorted(rows, key=lambda r: METHOD_ORDER.index(str(r["Method"])))
    labels  = [METHOD_LABELS.get(str(r["Method"]), str(r["Method"])) for r in ordered]
    means   = np.array([float(r["mean_D_total_ms"]) for r in ordered])
    x       = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    bar_colors = [COLORS.get(str(r["Method"]), "#888888") for r in ordered]
    ax.bar(
        x, means,
        width=0.55,
        color=bar_colors,
        edgecolor="#4A4A4A", linewidth=0.7,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel("Mean Total Delay  (ms)", labelpad=8)
    ax.set_ylim(0, means.max() * 1.25)

    label_bars(ax, fmt=".1f")
    ax.set_title("Total Delay — Controlled Random Hotspot v2",
                 fontsize=12, fontweight="bold", pad=12)
    style_axes(ax)
    save_figure(fig, output_path)


def plot_delay_breakdown(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    """Grouped bars: mean computation and transmission delay per method."""
    ordered = sorted(rows, key=lambda r: METHOD_ORDER.index(str(r["Method"])))
    labels  = [METHOD_LABELS.get(str(r["Method"]), str(r["Method"])) for r in ordered]
    compute = np.array([float(r["mean_AvgCompute_ms"]) for r in ordered])
    trans   = np.array([float(r["mean_AvgTrans_ms"]) for r in ordered])
    x       = np.arange(len(ordered))
    width   = 0.32
    gap     = 0.02

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    b1 = ax.bar(
        x - width / 2 - gap / 2, compute, width,
        label="Computation", color="#4F7F5F",
        edgecolor="#3A5C45", linewidth=0.7, zorder=3,
    )
    b2 = ax.bar(
        x + width / 2 + gap / 2, trans, width,
        label="Transmission", color="#C28A45",
        edgecolor="#9A6B32", linewidth=0.7, zorder=3,
    )

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#333333",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel("Delay  (ms)", labelpad=8)
    ax.set_ylim(0, max(compute.max(), trans.max()) * 1.25)
    ax.legend(fontsize=9.5, loc="upper right")

    ax.set_title("Delay Breakdown — Controlled Random Hotspot v2",
                 fontsize=12, fontweight="bold", pad=12)
    style_axes(ax)
    save_figure(fig, output_path)


def plot_substitutions(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    """Bar chart: mean expert substitutions per method."""
    ordered = sorted(rows, key=lambda r: METHOD_ORDER.index(str(r["Method"])))
    labels  = [METHOD_LABELS.get(str(r["Method"]), str(r["Method"])) for r in ordered]
    subs    = np.array([float(r["mean_Substitutions"]) for r in ordered])
    x       = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    bar_colors = [COLORS.get(str(r["Method"]), "#888888") for r in ordered]
    ax.bar(
        x, subs,
        width=0.55,
        color=bar_colors,
        edgecolor="#4A4A4A", linewidth=0.7,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel("Mean Substitutions", labelpad=8)
    ax.set_ylim(0, subs.max() * 1.25)

    label_bars(ax, fmt=".1f")

    ax.set_title("Expert Substitutions — Controlled Random Hotspot v2",
                 fontsize=12, fontweight="bold", pad=12)
    style_axes(ax)
    save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Run + export
# ═══════════════════════════════════════════════════════════════════════════

def export_controlled_random_hotspot_v2_results(
    output_dir: str | Path = "results",
    seeds: List[int] | None = None,
) -> List[Dict[str, float | str]]:
    """Run the v2 multi-seed evaluation and export CSV + figures."""
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
    plot_total_delay(     rows, output_path / "fig_total_delay_controlled_random_v2")
    plot_delay_breakdown( rows, output_path / "fig_delay_breakdown_controlled_random_v2")
    plot_substitutions(   rows, output_path / "fig_substitutions_controlled_random_v2")
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
