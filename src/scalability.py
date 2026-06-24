"""Scalability and spatial robustness experiments.

This module sweeps UAV count, task count, and square area size using the
parameterized hotspot scenario. It writes one consolidated CSV so figures and
tables can be regenerated without rerunning simulations.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from .baselines import (
    run_importance_placement,
    run_no_similarity_placement,
    run_random_placement,
)
from .config import SystemConfig
from .evaluation import MethodMetrics, compute_metrics
from .models import Expert, SystemResult, Task, UAV
from .pipeline import run_pipeline
from .plot_style import METHOD_ORDER
from .scenario import build_scalable_hotspot_scenario


SEEDS = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class ScalabilitySetting:
    """One scalability sweep setting."""

    experiment: str
    value: float
    num_uavs: int
    num_tasks: int
    area_size: float


@dataclass
class TimedMetrics:
    """Method metrics plus wall-clock runtime for one seed."""

    metrics: MethodMetrics
    runtime_s: float


MethodRunner = Callable[
    [List[UAV], List[Expert], List[Task], SystemConfig],
    SystemResult,
]


METHOD_RUNNERS: Dict[str, MethodRunner] = {
    "Proposed": run_pipeline,
    "Random Placement": run_random_placement,
    "Importance-based Placement": run_importance_placement,
    "No-similarity Placement": run_no_similarity_placement,
}


def _failed_result() -> SystemResult:
    return SystemResult(
        deployment={},
        execution_plans=[],
        D_total=float("inf"),
        D_weighted=float("inf"),
    )


def _run_method_timed(
    name: str,
    runner: MethodRunner,
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> TimedMetrics:
    """Run one method once and return metrics with runtime."""
    start = time.perf_counter()
    try:
        result = runner(uavs, experts, tasks, cfg)
    except RuntimeError as exc:
        print(f"{name} infeasible under scalability setting: {exc}")
        result = _failed_result()
    runtime_s = time.perf_counter() - start
    return TimedMetrics(compute_metrics(name, result), runtime_s)


def _finite_mean(values: Iterable[float]) -> float:
    vals = [value for value in values if np.isfinite(value)]
    if not vals:
        return float("inf")
    return float(np.mean(vals))


def _finite_std(values: Iterable[float]) -> float:
    vals = [value for value in values if np.isfinite(value)]
    if not vals:
        return float("nan")
    return float(np.std(vals))


def _summarize_setting(
    setting: ScalabilitySetting,
    metrics_by_method: Dict[str, List[TimedMetrics]],
) -> List[Dict[str, float | int | str]]:
    """Aggregate per-seed metrics into CSV rows."""
    rows: List[Dict[str, float | int | str]] = []
    for method in METHOD_ORDER:
        timed = metrics_by_method[method]
        metrics = [item.metrics for item in timed]
        d_totals = [m.D_total for m in metrics]
        per_task = [m.D_total / setting.num_tasks for m in metrics]
        feasible_count = sum(1 for m in metrics if m.feasible)
        infeasible_count = len(metrics) - feasible_count

        rows.append(
            {
                "experiment": setting.experiment,
                "value": setting.value,
                "num_uavs": setting.num_uavs,
                "num_tasks": setting.num_tasks,
                "area_size": setting.area_size,
                "method": method,
                "success_rate": feasible_count / len(metrics),
                "infeasible_count": infeasible_count,
                "mean_D_total_ms": _finite_mean(d_totals) * 1e3,
                "std_D_total_ms": _finite_std(d_totals) * 1e3,
                "mean_D_per_task_ms": _finite_mean(per_task) * 1e3,
                "std_D_per_task_ms": _finite_std(per_task) * 1e3,
                "mean_runtime_ms": float(
                    np.mean([item.runtime_s for item in timed]) * 1e3
                ),
                "mean_Substitutions": _finite_mean(
                    [m.substitutions for m in metrics if m.feasible]
                ),
                "mean_AvgCompute_ms": _finite_mean(
                    [m.avg_D_compute for m in metrics]
                )
                * 1e3,
                "mean_AvgTrans_ms": _finite_mean([m.avg_D_trans for m in metrics])
                * 1e3,
                "mean_Deployments": _finite_mean(
                    [m.deployments for m in metrics if m.feasible]
                ),
            }
        )
    return rows


def run_scalability_setting(
    setting: ScalabilitySetting,
    seeds: List[int] | None = None,
) -> List[Dict[str, float | int | str]]:
    """Run all methods for one scalability setting over multiple seeds."""
    if seeds is None:
        seeds = SEEDS

    metrics_by_method: Dict[str, List[TimedMetrics]] = {
        method: [] for method in METHOD_ORDER
    }

    for seed in seeds:
        cfg = SystemConfig()
        cfg.seed = seed
        cfg.max_copies_per_expert = max(1, math.ceil(setting.num_uavs / 3))
        uavs, experts, tasks = build_scalable_hotspot_scenario(
            cfg,
            num_uavs=setting.num_uavs,
            num_tasks=setting.num_tasks,
            area_size=setting.area_size,
        )
        for method in METHOD_ORDER:
            metrics_by_method[method].append(
                _run_method_timed(
                    method,
                    METHOD_RUNNERS[method],
                    uavs,
                    experts,
                    tasks,
                    cfg,
                )
            )

    return _summarize_setting(setting, metrics_by_method)


def default_scalability_settings() -> List[ScalabilitySetting]:
    """Return the default sweep design used for the paper add-on experiments."""
    settings: List[ScalabilitySetting] = []

    for num_uavs in [5, 10, 20]:
        settings.append(
            ScalabilitySetting(
                experiment="uav_count",
                value=float(num_uavs),
                num_uavs=num_uavs,
                num_tasks=60,
                area_size=240.0,
            )
        )

    for num_tasks in [20, 40, 60, 80, 100]:
        settings.append(
            ScalabilitySetting(
                experiment="task_count",
                value=float(num_tasks),
                num_uavs=10,
                num_tasks=num_tasks,
                area_size=240.0,
            )
        )

    for area_size in [240.0, 500.0, 1000.0]:
        settings.append(
            ScalabilitySetting(
                experiment="area_size",
                value=area_size,
                num_uavs=10,
                num_tasks=60,
                area_size=area_size,
            )
        )

    return settings


def save_scalability_csv(
    rows: List[Dict[str, float | int | str]],
    path: Path,
) -> None:
    """Save scalability rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "value",
        "num_uavs",
        "num_tasks",
        "area_size",
        "method",
        "success_rate",
        "infeasible_count",
        "mean_D_total_ms",
        "std_D_total_ms",
        "mean_D_per_task_ms",
        "std_D_per_task_ms",
        "mean_runtime_ms",
        "mean_Substitutions",
        "mean_AvgCompute_ms",
        "mean_AvgTrans_ms",
        "mean_Deployments",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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


def run_all_scalability_experiments(
    output_dir: str | Path = "results",
    seeds: List[int] | None = None,
) -> List[Dict[str, float | int | str]]:
    """Run all scalability sweeps and save the consolidated CSV."""
    output_path = Path(output_dir)
    all_rows: List[Dict[str, float | int | str]] = []

    for setting in default_scalability_settings():
        print(
            f"Running {setting.experiment}={setting.value:g} "
            f"(UAVs={setting.num_uavs}, tasks={setting.num_tasks}, "
            f"area={setting.area_size:g}x{setting.area_size:g})"
        )
        all_rows.extend(run_scalability_setting(setting, seeds=seeds))

    save_scalability_csv(all_rows, output_path / "scalability_summary.csv")
    return all_rows


def _print_compact_summary(rows: List[Dict[str, float | int | str]]) -> None:
    """Print the Proposed row and best baseline for each setting."""
    grouped: Dict[Tuple[str, float], List[Dict[str, float | int | str]]] = {}
    for row in rows:
        key = (str(row["experiment"]), float(row["value"]))
        grouped.setdefault(key, []).append(row)

    print("\n=== Scalability summary ===")
    for (experiment, value), group in grouped.items():
        proposed = next(r for r in group if r["method"] == "Proposed")
        baselines = [r for r in group if r["method"] != "Proposed"]
        best_baseline = min(
            baselines,
            key=lambda r: float(r["mean_D_per_task_ms"]),
        )
        print(
            f"{experiment}={value:g}: Proposed "
            f"{float(proposed['mean_D_per_task_ms']):.2f} ms/task, "
            f"success={float(proposed['success_rate']):.2f}; "
            f"best baseline={best_baseline['method']} "
            f"{float(best_baseline['mean_D_per_task_ms']):.2f} ms/task"
        )


def main() -> None:
    rows = run_all_scalability_experiments()
    _print_compact_summary(rows)


if __name__ == "__main__":
    main()
