"""Shared plotting style, colours, and helpers for all paper figures.

Import this module once and call :func:`apply_style` before creating any
figure.  Every plotting function in this project should use the colour
palettes, dimension constants, and save helper defined here so that all
outputs are visually consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
#  Method metadata
# ═══════════════════════════════════════════════════════════════════════════

METHOD_ORDER: List[str] = [
    "Proposed",
    "Random Placement",
    "Importance-based Placement",
    "No-similarity Placement",
]

METHOD_LABELS: Dict[str, str] = {
    "Proposed":                     "Proposed",
    "Random Placement":             "Random",
    "Importance-based Placement":   "Importance",
    "No-similarity Placement":      "No-Sim",
}

# Paul Tol / ColorBrewer inspired palette – colourblind-friendly.
# Proposed is the strongest / most saturated colour so it draws the eye
# first; baseline colours share a similar medium-low saturation.
COLORS: Dict[str, str] = {
    "Proposed":                     "#225EA8",   # vivid blue  — hero
    "Random Placement":             "#AAAAAA",   # mid grey    — naïve baseline
    "Importance-based Placement":   "#DD8833",   # warm orange
    "No-similarity Placement":      "#8466AA",   # muted purple
}

# Lighter shades for fill / background (derived from the main colours).
COLORS_LIGHT: Dict[str, str] = {
    "Proposed":                     "#AAC8E8",
    "Random Placement":             "#D5D5D5",
    "Importance-based Placement":   "#F0C090",
    "No-similarity Placement":      "#C0B0D8",
}

MARKERS: Dict[str, str] = {
    "Proposed":                     "o",
    "Random Placement":             "s",
    "Importance-based Placement":   "^",
    "No-similarity Placement":      "D",
}

# Bar-chart edge colour (dark grey for definition without harshness).
BAR_EDGE: str = "#4A4A4A"

# ═══════════════════════════════════════════════════════════════════════════
#  Figure dimensions (inches)
# ═══════════════════════════════════════════════════════════════════════════

FIG_SINGLE  = (7.0, 4.6)    # single-panel bar / line chart
FIG_WIDE_2  = (12.0, 4.8)   # two-panel horizontal
FIG_TALL_2  = (7.0, 8.0)    # two-panel vertical
FIG_TALL_3  = (7.0, 9.0)    # three-panel vertical

# ═══════════════════════════════════════════════════════════════════════════
#  Global rcParams
# ═══════════════════════════════════════════════════════════════════════════

def apply_style() -> None:
    """Set matplotlib rcParams for a clean, publication-ready look."""
    plt.rcParams.update({
        # Font
        "font.family":          "sans-serif",
        "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":            11,
        "axes.titlesize":       12,
        "axes.labelsize":       11,
        "xtick.labelsize":      10,
        "ytick.labelsize":      10,
        "legend.fontsize":      9,
        # Figure
        "figure.dpi":           150,
        "savefig.dpi":          300,
        "savefig.bbox":         None,
        "savefig.pad_inches":   0.04,
        # Spine defaults
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.9,
        "xtick.major.width":    0.7,
        "ytick.major.width":    0.7,
        "xtick.major.size":     4,
        "ytick.major.size":     4,
        # Grid defaults (can be overridden per-axes)
        "axes.grid":            False,
        "axes.axisbelow":       True,
        # Legend defaults
        "legend.frameon":       False,
        "legend.handletextpad": 0.6,
        "legend.borderpad":     0.3,
        # Misc
        "errorbar.capsize":     3.5,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Per-axes styling helpers
# ═══════════════════════════════════════════════════════════════════════════

def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    """Apply the house style to a single Axes.

    Parameters
    ----------
    ax:
        The axes to style.
    grid:
        When True, add faint horizontal-only grid lines behind data.
    """
    if grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save *fig* as PNG (300 dpi) and PDF (vector), then close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(
            path.with_suffix(f".{fmt}"),
            dpi=300,
            bbox_inches=None,
            facecolor="white",
            edgecolor="none",
        )
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  Annotation helpers for bar charts
# ═══════════════════════════════════════════════════════════════════════════

def label_bars(
    ax: plt.Axes,
    fmt: str = ".1f",
    offset_frac: float = 0.015,
    include_zero: bool = False,
    **kwargs,
) -> None:
    """Place value labels above every bar in *ax*.

    *offset_frac* is multiplied by the y-axis span to compute the vertical
    offset, so the label height adapts to the data range.
    """
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset_y = y_span * offset_frac

    kw = dict(
        ha="center", va="bottom",
        fontsize=8.5, fontweight="bold",
        color="#333333",
    )
    kw.update(kwargs)

    for bar in ax.patches:
        h = bar.get_height()
        if (h > 0 or include_zero) and np.isfinite(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset_y,
                f"{h:{fmt}}",
                **kw,
            )


def add_reference_line(
    ax: plt.Axes,
    y: float,
    label: str = "",
    color: str = "#CC3333",
    linestyle: str = "--",
    linewidth: float = 1.0,
) -> None:
    """Draw a dashed horizontal reference line across *ax*."""
    ax.axhline(y=y, color=color, linestyle=linestyle,
               linewidth=linewidth, alpha=0.75, zorder=2)
    if label:
        ax.text(
            0.98, 0.96, f"← {label}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8, color=color, alpha=0.85,
            fontweight="bold",
        )


def annotate_improvement(
    ax: plt.Axes,
    x: float,
    proposed_val: float,
    baseline_val: float,
    baseline_label: str,
) -> None:
    """Annotate the percentage reduction of *proposed_val* vs *baseline_val*.

    Places a text label and a downward arrow above the proposed bar.
    """
    if baseline_val == 0:
        return
    pct = (baseline_val - proposed_val) / baseline_val * 100
    if pct <= 0.5:   # negligible – don't annotate
        return

    # Place annotation part-way between the bar top and the axes top.
    y_min, y_max = ax.get_ylim()
    anchor_y = proposed_val + (y_max - proposed_val) * 0.25

    ax.annotate(
        f"↓ {pct:.0f}%\nvs {baseline_label}",
        xy=(x, proposed_val),
        xytext=(x, anchor_y),
        fontsize=8, color="#CC3333", fontweight="bold",
        ha="center", va="bottom",
        arrowprops=dict(
            arrowstyle="->", color="#CC3333", lw=0.9,
            connectionstyle="arc3,rad=0.15",
        ),
    )
