"""Stand-alone sensitivity line charts — quick single-panel versions.

For the multi-panel publication figures, see :mod:`.paper_figures`.
This module shares the same colour palette and styling rules from
:mod:`.plot_style`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import (
    COLORS,
    COLORS_LIGHT,
    FIG_SINGLE,
    MARKERS,
    METHOD_LABELS,
    METHOD_ORDER,
    apply_style,
    save_figure,
    style_axes,
)

apply_style()


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_float(value: str) -> float:
    text = value.strip().lower()
    if text == "inf":
        return float("inf")
    if text == "nan":
        return float("nan")
    return float(value)


def _read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ═══════════════════════════════════════════════════════════════════════════
#  Single-panel sensitivity plot
# ═══════════════════════════════════════════════════════════════════════════

def _plot_sensitivity(
    csv_path: Path,
    output_path: Path,
    xlabel: str,
    title: str,
    min_x: float | None = None,
    mark_infeasible: bool = False,
) -> None:
    """Draw one sensitivity line chart (single metric: D_total)."""
    rows = _read_rows(csv_path)

    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    infeasible_xs: List[float] = []

    for method in METHOD_ORDER:
        method_rows = [r for r in rows if r["method"] == method]
        method_rows.sort(key=lambda r: _to_float(r["value"]))

        x_vals: List[float] = []
        y_vals: List[float] = []

        for row in method_rows:
            x = _to_float(row["value"])
            mean = _to_float(row["mean_D_total_ms"])

            if min_x is not None and x < min_x:
                if mark_infeasible and not np.isfinite(mean):
                    infeasible_xs.append(x)
                continue
            if not np.isfinite(mean):
                if mark_infeasible:
                    infeasible_xs.append(x)
                continue
            x_vals.append(x)
            y_vals.append(mean)

        if not x_vals:
            continue

        xs = np.array(x_vals)
        means = np.array(y_vals)

        ax.plot(
            xs, means,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.2,
            markersize=6.0,
            markeredgewidth=0.4,
            markeredgecolor="#333333",
            zorder=3,
        )

    # ── Infeasible shading ───────────────────────────────────────────
    if mark_infeasible and infeasible_xs:
        unique_xs = sorted(set(infeasible_xs))
        # Shade the whole infeasible region
        ax.axvspan(
            unique_xs[0], unique_xs[-1],
            color="#CC3333", alpha=0.08, linewidth=0, zorder=0,
        )
        ax.text(
            0.02, 0.97, "Infeasible",
            transform=ax.transAxes,
            fontsize=9, color="#CC3333", fontweight="bold",
            fontstyle="italic",
            ha="left", va="top",
        )

    # ── Axis labelling ───────────────────────────────────────────────
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel("Mean Total Delay  (ms)", labelpad=8)

    # Expand x-range slightly
    all_xs = sorted({_to_float(r["value"]) for r in rows if np.isfinite(_to_float(r["value"]))})
    if all_xs:
        margin = (all_xs[-1] - all_xs[0]) * 0.08
        ax.set_xlim(all_xs[0] - margin, all_xs[-1] + margin)

    ax.legend(fontsize=9.5, loc="best")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    style_axes(ax)
    save_figure(fig, output_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_sensitivity_figures(results_dir: str | Path = "results") -> None:
    """Create stand-alone single-panel sensitivity figures."""
    root = Path(results_dir)
    _plot_sensitivity(
        root / "sensitivity_xi.csv",
        root / "fig4_sensitivity_xi",
        xlabel="Similarity Threshold  ξ",
        title="Effect of Similarity Threshold  ξ",
    )
    _plot_sensitivity(
        root / "sensitivity_mid_size.csv",
        root / "fig5_sensitivity_mid_size",
        xlabel="Intermediate Feature Size  (scale factor)",
        title="Effect of Intermediate Feature Size",
    )
    _plot_sensitivity(
        root / "sensitivity_memory.csv",
        root / "fig6_sensitivity_memory",
        xlabel="UAV Memory Capacity  (scale factor)",
        title="Effect of UAV Memory Capacity",
        min_x=1.0,
        mark_infeasible=True,
    )
    _plot_sensitivity(
        root / "sensitivity_compute_window.csv",
        root / "fig_compute_window_sensitivity",
        xlabel="Scheduling Window  (s)",
        title="Effect of Compute Scheduling Window",
        mark_infeasible=True,
    )


def main() -> None:
    plot_all_sensitivity_figures()
    print("Saved sensitivity figures:")
    print("  results/fig4_sensitivity_xi.{png,pdf}")
    print("  results/fig5_sensitivity_mid_size.{png,pdf}")
    print("  results/fig6_sensitivity_memory.{png,pdf}")
    print("  results/fig_compute_window_sensitivity.{png,pdf}")


if __name__ == "__main__":
    main()
