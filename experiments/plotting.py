"""
Experiment Plotting
=====================
Generate paper-ready figures from experiment reports.

Requires: matplotlib
"""

from __future__ import annotations
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from experiments.runner import ExperimentReport, ScenarioResult


# ======================================================================
# Colour palette & style
# ======================================================================
BASELINE_COLORS = {
    "AeroMoDE":        "#2166AC",   # blue
    "NoSubstitute":    "#B2182B",   # red
    "NoRedundancy":    "#D6604D",   # light red
    "RandomPlacement": "#4DAF4A",   # green
    "RoundRobin":      "#FF7F00",   # orange
}

BASELINE_MARKERS = {
    "AeroMoDE":        "o",
    "NoSubstitute":    "s",
    "NoRedundancy":    "D",
    "RandomPlacement": "^",
    "RoundRobin":      "v",
}

BASELINE_LINESTYLE = {
    "AeroMoDE":        "-",
    "NoSubstitute":    "--",
    "NoRedundancy":    "-.",
    "RandomPlacement": ":",
    "RoundRobin":      (0, (3, 1, 1, 1)),
}


def _setup_style():
    """Configure matplotlib for publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
        })
        return plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting. "
                          "Install with: pip install matplotlib")


# ======================================================================
# Helper: group & extract
# ======================================================================
def _group_by_prefix(scenarios: List[ScenarioResult], prefix: str) \
        -> List[ScenarioResult]:
    """Filter scenarios whose name starts with `prefix`."""
    return [s for s in scenarios if s.scenario_name.startswith(prefix)]


def _get_metric(sr: ScenarioResult, baseline: str, metric: str,
                default: float = np.nan) -> float:
    """Safely extract a metric value."""
    bd = sr.baselines.get(baseline, {})
    if "error" in bd:
        return default
    return bd.get(metric, default)


# ======================================================================
# Figure 1: UAV count sweep (bar chart)
# ======================================================================
def plot_uav_count(report: ExperimentReport, output_dir: str = "./figures"):
    """D_total vs number of UAVs, grouped bar chart."""
    plt = _setup_style()
    os.makedirs(output_dir, exist_ok=True)

    scs = _group_by_prefix(report.scenarios, "homogeneous_")
    if not scs:
        print("  [plot_uav_count] No homogeneous_* scenarios found.")
        return

    # Sort by UAV count
    scs.sort(key=lambda s: s.num_uavs)

    bls = report.baseline_names
    x_labels = [f"{s.num_uavs}" for s in scs]
    x = np.arange(len(x_labels))
    n_bars = len(bls)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for bi, bname in enumerate(bls):
        vals = [_get_metric(s, bname, "D_total_ms") for s in scs]
        offset = (bi - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=bname,
                      color=BASELINE_COLORS.get(bname, "#888888"),
                      edgecolor="white", linewidth=0.5)
        # Annotate values
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Number of UAVs")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("Total Inference Latency vs. UAV Count")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper right", ncol=2)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_uav_count.pdf"))
    fig.savefig(os.path.join(output_dir, "fig_uav_count.png"))
    plt.close(fig)
    print(f"  Saved fig_uav_count.*  to {output_dir}")


# ======================================================================
# Figure 2: Bandwidth sweep (line chart)
# ======================================================================
def plot_bandwidth_sweep(report: ExperimentReport, output_dir: str = "./figures"):
    """D_total vs bandwidth, multi-line chart."""
    plt = _setup_style()
    os.makedirs(output_dir, exist_ok=True)

    scs = _group_by_prefix(report.scenarios, "bandwidth_")
    if not scs:
        print("  [plot_bandwidth_sweep] No bandwidth_* scenarios found.")
        return

    scs.sort(key=lambda s: s.bandwidth_mhz)
    bls = report.baseline_names
    x_vals = [s.bandwidth_mhz for s in scs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for bname in bls:
        y_vals = [_get_metric(s, bname, "D_total_ms") for s in scs]
        ax.plot(x_vals, y_vals,
                color=BASELINE_COLORS.get(bname),
                marker=BASELINE_MARKERS.get(bname),
                linestyle=BASELINE_LINESTYLE.get(bname),
                linewidth=1.8, markersize=7, label=bname)

    ax.set_xlabel("Bandwidth (MHz)")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("Total Inference Latency vs. UAV-UAV Bandwidth")
    ax.legend()
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_bandwidth.pdf"))
    fig.savefig(os.path.join(output_dir, "fig_bandwidth.png"))
    plt.close(fig)
    print(f"  Saved fig_bandwidth.*  to {output_dir}")


# ======================================================================
# Figure 3: Heterogeneous environment comparison (bar chart)
# ======================================================================
def plot_heterogeneous(report: ExperimentReport, output_dir: str = "./figures"):
    """D_total across heterogeneous environments D, E, F."""
    plt = _setup_style()
    os.makedirs(output_dir, exist_ok=True)

    scs = _group_by_prefix(report.scenarios, "hetero_")
    if not scs:
        print("  [plot_heterogeneous] No hetero_* scenarios found.")
        return

    # Order: D, E, F
    order = {"hetero_D": 0, "hetero_E": 1, "hetero_F": 2}
    scs.sort(key=lambda s: order.get(s.scenario_name, 99))

    bls = report.baseline_names
    x_labels = [s.scenario_name.replace("hetero_", "Env ") for s in scs]
    x = np.arange(len(x_labels))
    n_bars = len(bls)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for bi, bname in enumerate(bls):
        vals = [_get_metric(s, bname, "D_total_ms") for s in scs]
        offset = (bi - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=bname,
                      color=BASELINE_COLORS.get(bname, "#888888"),
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Environment")
    ax.set_ylabel("Total Latency (ms)")
    ax.set_title("Performance on Heterogeneous UAV Environments")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_heterogeneous.pdf"))
    fig.savefig(os.path.join(output_dir, "fig_heterogeneous.png"))
    plt.close(fig)
    print(f"  Saved fig_heterogeneous.*  to {output_dir}")


# ======================================================================
# Figure 4: Latency decomposition (stacked bar)
# ======================================================================
def plot_latency_breakdown(report: ExperimentReport, output_dir: str = "./figures"):
    """Stacked bar: D_access + D_compute + D_trans + D_return."""
    plt = _setup_style()
    os.makedirs(output_dir, exist_ok=True)

    scs = _group_by_prefix(report.scenarios, "homogeneous_")
    if not scs:
        scs = report.scenarios[:4]  # fallback

    bls = report.baseline_names
    x_labels = [s.scenario_name for s in scs]
    x = np.arange(len(x_labels))
    components = ["D_access_ms", "D_compute_ms", "D_trans_ms", "D_return_ms"]
    comp_labels = ["Access", "Compute", "Transmission", "Return"]
    comp_colors = ["#FDB863", "#B2ABD2", "#E66101", "#5E3C99"]

    fig, axes = plt.subplots(1, len(bls), figsize=(5 * len(bls), 4.5),
                             sharey=True)
    if len(bls) == 1:
        axes = [axes]

    for ax, bname in zip(axes, bls):
        data = []
        for s in scs:
            row = [_get_metric(s, bname, c, 0.0) for c in components]
            data.append(row)
        data = np.array(data)

        bottom = np.zeros(len(scs))
        for ci, (comp, label, color) in enumerate(
            zip(components, comp_labels, comp_colors)
        ):
            vals = data[:, ci] if ci < data.shape[1] else np.zeros(len(scs))
            ax.bar(x, vals, width=0.6, bottom=bottom, label=label,
                   color=color, edgecolor="white", linewidth=0.5)
            bottom += vals

        ax.set_title(bname, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Latency (ms)")

    # Single legend
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Latency Breakdown by Component", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    fig.savefig(os.path.join(output_dir, "fig_latency_breakdown.pdf"))
    fig.savefig(os.path.join(output_dir, "fig_latency_breakdown.png"))
    plt.close(fig)
    print(f"  Saved fig_latency_breakdown.*  to {output_dir}")


# ======================================================================
# Figure 5: Memory utilisation (bar chart)
# ======================================================================
def plot_memory_utilisation(report: ExperimentReport, output_dir: str = "./figures"):
    """Mean and max memory utilisation per scenario."""
    plt = _setup_style()
    os.makedirs(output_dir, exist_ok=True)

    scs = _group_by_prefix(report.scenarios, "homogeneous_")
    if not scs:
        scs = report.scenarios[:4]

    bls = report.baseline_names
    x_labels = [s.scenario_name for s in scs]
    x = np.arange(len(x_labels))
    n_groups = len(bls)
    width = 0.35 / n_groups

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for bi, bname in enumerate(bls):
        mean_vals = [_get_metric(s, bname, "mem_util_mean", 0.0) * 100 for s in scs]
        max_vals = [_get_metric(s, bname, "mem_util_max", 0.0) * 100 for s in scs]
        offset = (bi - (n_groups - 1) / 2) * width

        ax1.bar(x + offset, mean_vals, width, label=bname,
                color=BASELINE_COLORS.get(bname, "#888"),
                edgecolor="white", linewidth=0.5)
        ax2.bar(x + offset, max_vals, width, label=bname,
                color=BASELINE_COLORS.get(bname, "#888"),
                edgecolor="white", linewidth=0.5)

    ax1.set_title("Mean Memory Utilisation")
    ax1.set_ylabel("Utilisation (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax1.set_ylim(0, 105)

    ax2.set_title("Max Memory Utilisation")
    ax2.set_ylabel("Utilisation (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0, 105)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(bls),
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("UAV Memory Utilisation")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    fig.savefig(os.path.join(output_dir, "fig_memory.pdf"))
    fig.savefig(os.path.join(output_dir, "fig_memory.png"))
    plt.close(fig)
    print(f"  Saved fig_memory.*  to {output_dir}")


# ======================================================================
# Figure 6: Substitution rate analysis
# ======================================================================
def plot_substitution_rate(report: ExperimentReport, output_dir: str = "./figures"):
    """Substitution rate vs scenarios."""
    plt = _setup_style()
    os.makedirs(output_dir, exist_ok=True)

    # Only AeroMoDE uses substitution
    if "AeroMoDE" not in report.baseline_names:
        print("  [plot_substitution_rate] AeroMoDE not in baselines.")
        return

    scs = [s for s in report.scenarios
           if "hetero_" in s.scenario_name or "homogeneous_" in s.scenario_name]
    if not scs:
        scs = report.scenarios

    x_labels = [s.scenario_name for s in scs]
    x = np.arange(len(x_labels))
    sub_rates = [_get_metric(s, "AeroMoDE", "substitution_rate", 0.0) * 100
                 for s in scs]
    n_experts = [s.num_experts for s in scs]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    bars = ax1.bar(x, sub_rates, width=0.5, color="#2166AC", edgecolor="white",
                   label="Substitution Rate (%)")
    ax1.set_ylabel("Substitution Rate (%)")
    ax1.set_ylim(0, max(max(sub_rates) * 1.3, 10))

    # Annotate with n_experts
    for bar, rate, ne in zip(bars, sub_rates, n_experts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{rate:.1f}%\n({ne} exp)",
                 ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax1.set_title("Expert Substitution Rate Across Scenarios")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_substitution.pdf"))
    fig.savefig(os.path.join(output_dir, "fig_substitution.png"))
    plt.close(fig)
    print(f"  Saved fig_substitution.*  to {output_dir}")


# ======================================================================
# Figure 7: Parameter sensitivity (line charts)
# ======================================================================
def plot_sensitivity(
    sensitivity_reports: Dict[str, ExperimentReport],
    output_dir: str = "./figures",
):
    """Generate sensitivity analysis subplots for each parameter."""
    plt = _setup_style()
    os.makedirs(output_dir, exist_ok=True)

    param_labels = {
        "xi":     ("Similarity Threshold ξ", "ξ"),
        "lambda": ("Error Penalty λ", "λ"),
        "eta":    ("Neighbour Demand Weight η", "η"),
        "nu":     ("Redundancy Penalty ν", "ν"),
    }

    valid_params = {k: v for k, v in sensitivity_reports.items()
                    if v.scenarios}
    if not valid_params:
        print("  [plot_sensitivity] No sensitivity data.")
        return

    n_params = len(valid_params)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for ax, (param, report) in zip(axes, valid_params.items()):
        label, xlabel = param_labels.get(param, (param, param))
        bls = report.baseline_names

        # Extract x values from scenario names (e.g. "xi_0.7" → 0.7)
        x_vals = []
        for s in report.scenarios:
            try:
                x_vals.append(float(s.scenario_name.split("_")[-1]))
            except ValueError:
                x_vals.append(len(x_vals))
        # Sort by x
        order = np.argsort(x_vals)
        x_sorted = [x_vals[i] for i in order]

        for bname in bls:
            y_vals = [_get_metric(report.scenarios[i], bname, "D_total_ms")
                      for i in order]
            ax.plot(x_sorted, y_vals,
                    color=BASELINE_COLORS.get(bname),
                    marker=BASELINE_MARKERS.get(bname),
                    linestyle=BASELINE_LINESTYLE.get(bname),
                    linewidth=1.5, markersize=6, label=bname)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Total Latency (ms)")
        ax.set_title(f"Sensitivity to {label}")
        if param == "xi":
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Parameter Sensitivity Analysis", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(os.path.join(output_dir, "fig_sensitivity.pdf"))
    fig.savefig(os.path.join(output_dir, "fig_sensitivity.png"))
    plt.close(fig)
    print(f"  Saved fig_sensitivity.*  to {output_dir}")


# ======================================================================
# Generate all figures
# ======================================================================
def generate_all_figures(
    report: ExperimentReport,
    sensitivity_reports: Optional[Dict[str, ExperimentReport]] = None,
    output_dir: str = "./figures",
):
    """Generate all paper figures from experiment reports."""
    print("\n" + "=" * 60)
    print("  Generating Figures")
    print("=" * 60)

    plot_uav_count(report, output_dir)
    plot_bandwidth_sweep(report, output_dir)
    plot_heterogeneous(report, output_dir)
    plot_latency_breakdown(report, output_dir)
    plot_memory_utilisation(report, output_dir)
    plot_substitution_rate(report, output_dir)

    if sensitivity_reports:
        plot_sensitivity(sensitivity_reports, output_dir)

    print(f"\nAll figures saved to {output_dir}/")
    print("=" * 60)
