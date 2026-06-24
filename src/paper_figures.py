"""Publication-quality paper figures from experiment CSV files.

All figures share the colour palette, font sizes, and styling rules
defined in :mod:`.plot_style`.  This module only **reads** CSV files; it
never re-runs simulations.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import (
    COLORS,
    COLORS_LIGHT,
    FIG_SINGLE,
    FIG_TALL_2,
    FIG_TALL_3,
    FIG_WIDE_2,
    MARKERS,
    METHOD_LABELS,
    METHOD_ORDER,
    annotate_improvement,
    apply_style,
    label_bars,
    save_figure,
    style_axes,
)

# Apply global rcParams on import so figures are consistent.
apply_style()

# ═══════════════════════════════════════════════════════════════════════════
#  CSV helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(raw: str) -> float:
    text = raw.strip().lower()
    if text == "inf":
        return float("inf")
    if text == "nan":
        return float("nan")
    return float(text)


def _order(rows: List[Dict[str, str]], key: str = "Method") -> List[Dict[str, str]]:
    """Return *rows* ordered by ``METHOD_ORDER``, keyed by *key*."""
    by = {row[key]: row for row in rows}
    return [by[m] for m in METHOD_ORDER if m in by]


# ═══════════════════════════════════════════════════════════════════════════
#  Fig. 1 — Total delay (bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig1_total_delay(summary_csv: Path, output_base: Path) -> None:
    """Bar chart comparing mean total delay across methods."""
    rows = _order(_read_csv(summary_csv))
    labels = [METHOD_LABELS[r["Method"]] for r in rows]
    means  = np.array([_to_float(r["mean_D_total_ms"]) for r in rows])
    x      = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    bar_colors = [COLORS[r["Method"]] for r in rows]

    ax.bar(
        x, means,
        width=0.55,
        color=bar_colors,
        edgecolor="#4A4A4A", linewidth=0.7,
        zorder=3,
    )

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_xlabel("Method", labelpad=6)

    # Y-axis
    ax.set_ylabel("Mean Total Delay  (ms)", labelpad=8)
    ax.set_ylim(0, means.max() * 1.25)  # leave room for annotations

    # Value labels on bars
    label_bars(ax, fmt=".1f")

    # Improvement annotation vs best baseline
    proposed_mean = means[0]
    best_baseline = means[1:].min()
    best_baseline_label = labels[1:][np.argmin(means[1:])]
    annotate_improvement(ax, 0.0, proposed_mean, best_baseline, best_baseline_label)

    style_axes(ax)
    ax.set_title("Mean Total Delay Comparison", fontsize=12, fontweight="bold", pad=12)
    save_figure(fig, output_base)


# ═══════════════════════════════════════════════════════════════════════════
#  Fig. 2 — Delay breakdown (grouped bars)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig2_delay_breakdown(summary_csv: Path, output_base: Path) -> None:
    """Grouped bars: mean computation delay and mean transmission delay."""
    rows = _order(_read_csv(summary_csv))
    labels = [METHOD_LABELS[r["Method"]] for r in rows]
    compute = np.array([_to_float(r["mean_AvgCompute_ms"]) for r in rows])
    trans   = np.array([_to_float(r["mean_AvgTrans_ms"])   for r in rows])
    x       = np.arange(len(rows))
    width   = 0.32
    gap     = 0.02

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    b1 = ax.bar(
        x - width / 2 - gap / 2, compute, width,
        label="Computation",
        color="#4F7F5F",
        edgecolor="#3A5C45", linewidth=0.7,
        zorder=3,
    )
    b2 = ax.bar(
        x + width / 2 + gap / 2, trans, width,
        label="Transmission",
        color="#C28A45",
        edgecolor="#9A6B32", linewidth=0.7,
        zorder=3,
    )

    # Value labels
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
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel("Mean Delay  (ms)", labelpad=8)

    # Leave headroom for value labels
    y_max = max(compute.max(), trans.max()) * 1.25
    ax.set_ylim(0, y_max)

    ax.legend(fontsize=9.5, loc="upper right")
    style_axes(ax)
    ax.set_title("Delay Breakdown by Component", fontsize=12, fontweight="bold", pad=12)
    save_figure(fig, output_base)


# ═══════════════════════════════════════════════════════════════════════════
#  Fig. 3 — Substitutions (bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig3_substitutions(summary_csv: Path, output_base: Path) -> None:
    """Bar chart comparing mean expert substitutions across methods."""
    rows = _order(_read_csv(summary_csv))
    labels = [METHOD_LABELS[r["Method"]] for r in rows]
    subs   = np.array([_to_float(r["mean_Substitutions"]) for r in rows])
    x      = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=FIG_SINGLE)

    bar_colors = [COLORS[r["Method"]] for r in rows]
    ax.bar(
        x, subs,
        width=0.55,
        color=bar_colors,
        edgecolor="#4A4A4A", linewidth=0.7,
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_xlabel("Method", labelpad=6)
    ax.set_ylabel("Mean Expert Substitutions", labelpad=8)
    ax.set_ylim(0, subs.max() * 1.25)

    label_bars(ax, fmt=".1f")

    style_axes(ax)
    ax.set_title("Expert Substitution Count", fontsize=12, fontweight="bold", pad=12)
    save_figure(fig, output_base)


# ═══════════════════════════════════════════════════════════════════════════
#  Sensitivity helpers
# ═══════════════════════════════════════════════════════════════════════════

def _add_sensitivity_lines(
    ax: plt.Axes,
    rows: List[Dict[str, str]],
    field: str = "mean_D_total_ms",
    std_field: str = "std_D_total_ms",
) -> None:
    """Draw one mean-value line per method on *ax*."""
    grouped: Dict[str, Tuple[List[float], List[float]]] = {}
    for method in METHOD_ORDER:
        method_rows = [r for r in rows if r["method"] == method]
        method_rows.sort(key=lambda r: _to_float(r["value"]))
        xs, means = [], []
        for r in method_rows:
            m = _to_float(r[field])
            if not np.isfinite(m):
                continue
            xs.append(_to_float(r["value"]))
            means.append(m)
        if xs:
            grouped[method] = (xs, means)

    for method in METHOD_ORDER:
        entry = grouped.get(method)
        if entry is None:
            continue
        xs, means = entry

        x_arr = np.array(xs)
        m_arr = np.array(means)

        ax.plot(
            x_arr, m_arr,
            label=METHOD_LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2.0,
            markersize=6.0,
            markeredgewidth=0.5,
            markeredgecolor="#333333",
            zorder=3,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Sensitivity figures — multi-panel
# ═══════════════════════════════════════════════════════════════════════════

def _plot_sensitivity_3panel(
    csv_path: Path,
    output_base: Path,
    xlabel: str,
    title: str,
    fields: List[Tuple[str, str, str]],
    mark_infeasible: bool = False,
    infeasible_min_x: float | None = None,
) -> None:
    """Generic 3-row sensitivity figure.

    Each row plots one metric from *fields* (label, csv_column, std_column).
    All rows share the same x-axis and a single legend placed below.
    """
    rows = _read_csv(csv_path)
    n_panels = len(fields)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(FIG_TALL_3[0], FIG_TALL_3[1] * n_panels / 3),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    # ── Determine infeasible region ──────────────────────────────────
    infeasible_min: float | None = None
    all_x_vals = sorted({_to_float(r["value"]) for r in rows})
    if mark_infeasible and infeasible_min_x is not None:
        # Find the largest x that is < infeasible_min_x
        below = [xv for xv in all_x_vals if xv < infeasible_min_x]
        if below:
            infeasible_min = min(all_x_vals)
            infeasible_max = max(below)

    # ── Draw each panel ───────────────────────────────────────────────
    for idx, (ylabel, field, std_field) in enumerate(fields):
        ax = axes[idx]
        _add_sensitivity_lines(ax, rows, field=field, std_field=std_field)

        # Shade infeasible region (left portion)
        if infeasible_min is not None:
            ax.axvspan(
                infeasible_min, infeasible_max,
                color="#CC3333", alpha=0.08, linewidth=0, zorder=0,
            )
            # Label on first panel only
            if idx == 0:
                ax.text(
                    (infeasible_min + infeasible_max) / 2, 0.98,
                    "Infeasible",
                    transform=ax.get_yaxis_transform(),
                    ha="center", va="top",
                    fontsize=9, color="#CC3333", fontweight="bold",
                    fontstyle="italic",
                )

        ax.set_ylabel(ylabel, labelpad=6)
        style_axes(ax)
        # Sub-panel label  (a), (b), (c)
        ax.text(
            -0.06, 1.02, f"({chr(97 + idx)})",
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left",
        )

    # ── Shared x-label ────────────────────────────────────────────────
    axes[-1].set_xlabel(xlabel, labelpad=8)
    # Expand x-range slightly for visual breathing room
    all_xs = sorted({_to_float(r["value"]) for r in rows if np.isfinite(_to_float(r["value"]))})
    if all_xs:
        margin = (all_xs[-1] - all_xs[0]) * 0.08
        axes[-1].set_xlim(all_xs[0] - margin, all_xs[-1] + margin)

    # ── Suptitle ──────────────────────────────────────────────────────
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # ── Shared legend below all panels ────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(METHOD_ORDER),
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.06),
        frameon=False,
    )

    fig.tight_layout()
    save_figure(fig, output_base)


def _plot_sensitivity_2panel(
    csv_path: Path,
    output_base: Path,
    xlabel: str,
    title: str,
) -> None:
    """Two-panel sensitivity: (a) total delay, (b) delay breakdown.

    Panel (b) overlays AvgCompute (dashed) and AvgTrans (dotted) per method.
    """
    rows = _read_csv(csv_path)

    fig, axes = plt.subplots(2, 1, figsize=FIG_TALL_2, sharex=True)
    ax1, ax2 = axes

    # ── Panel (a): Total delay ────────────────────────────────────
    _add_sensitivity_lines(ax1, rows, field="mean_D_total_ms")
    ax1.set_ylabel("Mean Total Delay  (ms)", labelpad=6)
    style_axes(ax1)
    ax1.text(-0.06, 1.02, "(a)", transform=ax1.transAxes,
             fontsize=11, fontweight="bold", va="bottom", ha="left")

    # ── Panel (b): Breakdown ──────────────────────────────────────
    # For clarity we overlay only Proposed and the best baseline
    for method in METHOD_ORDER:
        method_rows = [r for r in rows if r["method"] == method]
        method_rows.sort(key=lambda r: _to_float(r["value"]))
        xs = [_to_float(r["value"]) for r in method_rows]
        comp = [_to_float(r["mean_AvgCompute_ms"]) for r in method_rows]
        trans = [_to_float(r["mean_AvgTrans_ms"]) for r in method_rows]

        if not xs:
            continue

        ax2.plot(xs, comp, color=COLORS[method], linewidth=1.6, linestyle="--",
                 marker="s", markersize=4.5, markeredgewidth=0.3,
                 markeredgecolor="#333333", alpha=0.85)
        ax2.plot(xs, trans, color=COLORS[method], linewidth=1.6, linestyle=":",
                 marker="^", markersize=4.5, markeredgewidth=0.3,
                 markeredgecolor="#333333", alpha=0.85)

    # Dummy lines for the breakdown legend
    from matplotlib.lines import Line2D
    legend_comp = Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.6, label="Computation")
    legend_trans = Line2D([0], [0], color="#555555", linestyle=":", linewidth=1.6, label="Transmission")
    ax2.legend(handles=[legend_comp, legend_trans], fontsize=9, loc="upper right")

    ax2.set_ylabel("Delay Breakdown  (ms)", labelpad=6)
    style_axes(ax2)
    ax2.text(-0.06, 1.02, "(b)", transform=ax2.transAxes,
             fontsize=11, fontweight="bold", va="bottom", ha="left")

    # ── Shared x-axis ─────────────────────────────────────────────
    axes[-1].set_xlabel(xlabel, labelpad=8)
    all_xs = sorted({_to_float(r["value"]) for r in rows})
    if all_xs:
        margin = (all_xs[-1] - all_xs[0]) * 0.08
        axes[-1].set_xlim(all_xs[0] - margin, all_xs[-1] + margin)

    # ── Shared method legend ──────────────────────────────────────
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(METHOD_ORDER),
               fontsize=9.5, bbox_to_anchor=(0.5, -0.07), frameon=False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, output_base)


# ═══════════════════════════════════════════════════════════════════════════
#  Public entry points for each sensitivity figure
# ═══════════════════════════════════════════════════════════════════════════

_SENSITIVITY_FIELDS: List[Tuple[str, str, str]] = [
    ("Total Delay  (ms)",          "mean_D_total_ms",    "std_D_total_ms"),
    ("Avg Computation  (ms)",      "mean_AvgCompute_ms", "__no_std__"),
    ("Avg Transmission  (ms)",     "mean_AvgTrans_ms",   "__no_std__"),
]


def plot_fig4_sensitivity_xi(csv_path: Path, output_base: Path) -> None:
    """ξ (similarity threshold) → 3-panel sensitivity chart."""
    _plot_sensitivity_3panel(
        csv_path, output_base,
        xlabel="Similarity Threshold  ξ",
        title="Effect of Similarity Threshold  ξ",
        fields=_SENSITIVITY_FIELDS,
    )


def plot_fig5_sensitivity_mid_size(csv_path: Path, output_base: Path) -> None:
    """Intermediate feature size → 3-panel sensitivity chart."""
    _plot_sensitivity_3panel(
        csv_path, output_base,
        xlabel="Intermediate Feature Size  (scale factor)",
        title="Effect of Intermediate Feature Size",
        fields=_SENSITIVITY_FIELDS,
    )


def plot_fig6_sensitivity_memory(csv_path: Path, output_base: Path) -> None:
    """UAV memory capacity → 3-panel sensitivity chart with infeasible region."""
    _plot_sensitivity_3panel(
        csv_path, output_base,
        xlabel="UAV Memory Capacity  (scale factor)",
        title="Effect of UAV Memory Capacity",
        fields=_SENSITIVITY_FIELDS,
        mark_infeasible=True,
        infeasible_min_x=1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Master dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def plot_all_paper_figures(results_dir: str | Path = "results") -> None:
    """Generate all paper figures (Fig. 1 – 6)."""
    root = Path(results_dir)

    # ― Bar charts (from summary CSV) ――――――――――――――――――――――――――――――――
    summary = root / "controlled_random_hotspot_v2_summary.csv"
    plot_fig1_total_delay(summary,      root / "fig1_total_delay")
    plot_fig2_delay_breakdown(summary,  root / "fig2_delay_breakdown")
    plot_fig3_substitutions(summary,    root / "fig3_substitutions")

    # ― Sensitivity line charts ―――――――――――――――――――――――――――――――――――――
    plot_fig4_sensitivity_xi(       root / "sensitivity_xi.csv",       root / "fig4_sensitivity_xi")
    plot_fig5_sensitivity_mid_size( root / "sensitivity_mid_size.csv", root / "fig5_sensitivity_mid_size")
    plot_fig6_sensitivity_memory(   root / "sensitivity_memory.csv",   root / "fig6_sensitivity_memory")


def main() -> None:
    plot_all_paper_figures()
    print("Saved paper figures fig1–fig6 (PNG + PDF) under results/.")


if __name__ == "__main__":
    main()
