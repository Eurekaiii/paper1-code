"""Sensitivity experiments for the controlled random hotspot v2 scenario.

This module only varies experiment inputs and configuration values. It does
not modify placement, scheduling, baseline, or scenario implementations.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from .config import SystemConfig
from .evaluation import MethodMetrics, compute_metrics, run_all_methods
from .models import Expert, Task, UAV
from .scenario import build_controlled_random_hotspot_scenario_v2


SEEDS = [0, 1, 2, 3, 4]
METHOD_ORDER = [
    "Proposed",
    "Random Placement",
    "Importance-based Placement",
    "No-similarity Placement",
]


ScenarioMutator = Callable[[List[UAV], List[Expert], List[Task]], None]


def _run_one_setting(
    parameter: str,
    value: float,
    configure: Callable[[SystemConfig], None] | None = None,
    mutate_scenario: ScenarioMutator | None = None,
) -> List[Dict[str, float | str]]:
    """Run all methods across fixed seeds for one sensitivity setting."""
    metrics_by_method: Dict[str, List[MethodMetrics]] = {
        method: [] for method in METHOD_ORDER
    }

    for seed in SEEDS:
        cfg = SystemConfig()
        cfg.seed = seed
        if configure is not None:
            configure(cfg)

        uavs, experts, tasks = build_controlled_random_hotspot_scenario_v2(cfg)
        if mutate_scenario is not None:
            mutate_scenario(uavs, experts, tasks)

        results = run_all_methods(uavs, experts, tasks, cfg)
        for method, result in results:
            metrics_by_method[method].append(compute_metrics(method, result))

    rows: List[Dict[str, float | str]] = []
    for method in METHOD_ORDER:
        metrics = metrics_by_method[method]
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "method": method,
                "mean_D_total_ms": float(np.mean([m.D_total for m in metrics]) * 1e3),
                "std_D_total_ms": float(np.std([m.D_total for m in metrics]) * 1e3),
                "mean_Substitutions": float(np.mean([m.substitutions for m in metrics])),
                "mean_AvgCompute_ms": float(
                    np.mean([m.avg_D_compute for m in metrics]) * 1e3
                ),
                "mean_AvgTrans_ms": float(
                    np.mean([m.avg_D_trans for m in metrics]) * 1e3
                ),
                "mean_Deployments": float(np.mean([m.deployments for m in metrics])),
            }
        )
    return rows


def _save_csv(rows: List[Dict[str, float | str]], path: Path, fields: List[str]) -> None:
    """Save sensitivity rows with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        f"{row[field]:.6f}"
                        if isinstance(row.get(field), float)
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def _print_summary(
    title: str,
    rows: List[Dict[str, float | str]],
    include_deployments: bool = False,
    include_success: bool = False,
) -> None:
    """Print compact terminal summary for a sensitivity experiment."""
    headers = [
        "value",
        "method",
        *(
            ["success_rate", "infeasible_count"]
            if include_success
            else []
        ),
        "mean_D_total_ms",
        "std_D_total_ms",
        "mean_Substitutions",
        "mean_AvgCompute_ms",
        "mean_AvgTrans_ms",
    ]
    if include_deployments:
        headers.append("mean_Deployments")

    table = [
        [
            f"{float(row['value']):.2f}",
            str(row["method"]),
            *(
                [
                    f"{float(row.get('success_rate', 0.0)):.2f}",
                    f"{float(row.get('infeasible_count', 0.0)):.0f}",
                ]
                if include_success
                else []
            ),
            f"{float(row['mean_D_total_ms']):.2f}",
            f"{float(row['std_D_total_ms']):.2f}",
            f"{float(row['mean_Substitutions']):.2f}",
            f"{float(row['mean_AvgCompute_ms']):.2f}",
            f"{float(row['mean_AvgTrans_ms']):.2f}",
            *(
                [f"{float(row['mean_Deployments']):.2f}"]
                if include_deployments
                else []
            ),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table))
        for i in range(len(headers))
    ]

    print(f"\n=== {title} ===")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in table:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def run_xi_sensitivity(output_dir: Path) -> List[Dict[str, float | str]]:
    """Run similarity-threshold sensitivity."""
    rows: List[Dict[str, float | str]] = []
    for xi in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        rows.extend(
            _run_one_setting(
                parameter="xi",
                value=xi,
                configure=lambda cfg, xi=xi: setattr(cfg.similarity, "xi", xi),
            )
        )

    fields = [
        "parameter",
        "value",
        "method",
        "mean_D_total_ms",
        "std_D_total_ms",
        "mean_Substitutions",
        "mean_AvgCompute_ms",
        "mean_AvgTrans_ms",
    ]
    _save_csv(rows, output_dir / "sensitivity_xi.csv", fields)
    _print_summary("Sensitivity: similarity threshold xi", rows)
    return rows


def run_memory_sensitivity(output_dir: Path) -> List[Dict[str, float | str]]:
    """Run UAV-memory-scale sensitivity."""
    rows: List[Dict[str, float | str]] = []
    for scale in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
        def mutate_memory(
            uavs: List[UAV],
            _experts: List[Expert],
            _tasks: List[Task],
            scale: float = scale,
        ) -> None:
            for uav in uavs:
                uav.M_u *= scale

        rows.extend(
            _run_one_setting(
                parameter="memory_scale",
                value=scale,
                mutate_scenario=mutate_memory,
            )
        )

    fields = [
        "parameter",
        "value",
        "method",
        "mean_D_total_ms",
        "std_D_total_ms",
        "mean_Substitutions",
        "mean_AvgCompute_ms",
        "mean_AvgTrans_ms",
        "mean_Deployments",
    ]
    _save_csv(rows, output_dir / "sensitivity_memory.csv", fields)
    _print_summary("Sensitivity: UAV memory scale", rows, include_deployments=True)
    return rows


def run_compute_window_sensitivity(output_dir: Path) -> List[Dict[str, float | str]]:
    """Run scheduling-window compute budget sensitivity."""
    rows: List[Dict[str, float | str]] = []

    def finite_mean(values: List[float], empty_value: float = float("inf")) -> float:
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else empty_value

    def finite_std(values: List[float]) -> float:
        finite = [value for value in values if np.isfinite(value)]
        return float(np.std(finite)) if finite else float("nan")

    for window_s in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        metrics_by_method: Dict[str, List[MethodMetrics]] = {
            method: [] for method in METHOD_ORDER
        }
        for seed in SEEDS:
            cfg = SystemConfig()
            cfg.seed = seed
            cfg.scheduling.compute_window_s = window_s
            uavs, experts, tasks = build_controlled_random_hotspot_scenario_v2(cfg)
            results = run_all_methods(uavs, experts, tasks, cfg)
            for method, result in results:
                metrics_by_method[method].append(compute_metrics(method, result))

        for method, metrics in metrics_by_method.items():
            feasible_count = sum(1 for metric in metrics if metric.feasible)
            rows.append(
                {
                    "parameter": "compute_window_s",
                    "value": window_s,
                    "method": method,
                    "success_rate": feasible_count / len(metrics),
                    "infeasible_count": float(len(metrics) - feasible_count),
                    "mean_D_total_ms": finite_mean([m.D_total for m in metrics]) * 1e3,
                    "std_D_total_ms": finite_std([m.D_total for m in metrics]) * 1e3,
                    "mean_Substitutions": finite_mean([
                        float(m.substitutions) for m in metrics if m.feasible
                    ], empty_value=0.0),
                    "mean_AvgCompute_ms": finite_mean([
                        m.avg_D_compute for m in metrics
                    ]) * 1e3,
                    "mean_AvgTrans_ms": finite_mean([
                        m.avg_D_trans for m in metrics
                    ]) * 1e3,
                }
            )

    fields = [
        "parameter",
        "value",
        "method",
        "success_rate",
        "infeasible_count",
        "mean_D_total_ms",
        "std_D_total_ms",
        "mean_Substitutions",
        "mean_AvgCompute_ms",
        "mean_AvgTrans_ms",
    ]
    _save_csv(rows, output_dir / "sensitivity_compute_window.csv", fields)
    _print_summary(
        "Sensitivity: compute scheduling window",
        rows,
        include_success=True,
    )
    return rows


def run_mid_size_sensitivity(output_dir: Path) -> List[Dict[str, float | str]]:
    """Run intermediate-feature-size sensitivity."""
    rows: List[Dict[str, float | str]] = []
    for scale in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        def mutate_mid(_uavs: List[UAV], _experts: List[Expert], tasks: List[Task], scale: float = scale) -> None:
            for task in tasks:
                task.S_mid = [value * scale for value in task.S_mid]

        rows.extend(
            _run_one_setting(
                parameter="mid_scale",
                value=scale,
                mutate_scenario=mutate_mid,
            )
        )

    fields = [
        "parameter",
        "value",
        "method",
        "mean_D_total_ms",
        "std_D_total_ms",
        "mean_Substitutions",
        "mean_AvgCompute_ms",
        "mean_AvgTrans_ms",
    ]
    _save_csv(rows, output_dir / "sensitivity_mid_size.csv", fields)
    _print_summary("Sensitivity: intermediate feature size scale", rows)
    return rows


def run_all_sensitivity_experiments(output_dir: str | Path = "results") -> None:
    """Run all requested sensitivity experiments and save CSV files."""
    output_path = Path(output_dir)
    run_xi_sensitivity(output_path)
    run_memory_sensitivity(output_path)
    run_compute_window_sensitivity(output_path)
    run_mid_size_sensitivity(output_path)


def main() -> None:
    run_all_sensitivity_experiments()


if __name__ == "__main__":
    main()
