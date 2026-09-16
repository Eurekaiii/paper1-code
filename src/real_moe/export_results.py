"""CSV exports for real MoE trace-driven experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..evaluation import compute_metrics
from ..models import SystemResult


def _fmt(value: object) -> object:
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.6f}"
        return str(value)
    return value


def save_summary_csv(
    results: List[Tuple[str, SystemResult]],
    output_path: str | Path,
) -> None:
    """Save one summary row per method."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Method",
        "D_total_ms",
        "D_weighted_ms",
        "Substitutions",
        "Deployments",
        "AvgAccess_ms",
        "AvgCompute_ms",
        "AvgTrans_ms",
        "AvgReturn_ms",
        "Feasible",
    ]
    rows: List[Dict[str, object]] = []
    for name, result in results:
        metrics = compute_metrics(name, result)
        rows.append(
            {
                "Method": name,
                "D_total_ms": metrics.D_total * 1e3,
                "D_weighted_ms": result.D_weighted * 1e3,
                "Substitutions": metrics.substitutions,
                "Deployments": metrics.deployments,
                "AvgAccess_ms": metrics.avg_D_access * 1e3,
                "AvgCompute_ms": metrics.avg_D_compute * 1e3,
                "AvgTrans_ms": metrics.avg_D_trans * 1e3,
                "AvgReturn_ms": metrics.avg_D_return * 1e3,
                "Feasible": metrics.feasible,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row[key]) for key in fields})


def save_deployment_csv(
    results: List[Tuple[str, SystemResult]],
    output_path: str | Path,
) -> None:
    """Save deployed expert-UAV pairs for every method."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Method", "Expert", "UAV", "Deployed"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in results:
            for (expert_id, uav_id), deployed in sorted(result.deployment.items()):
                writer.writerow(
                    {
                        "Method": method,
                        "Expert": expert_id,
                        "UAV": uav_id,
                        "Deployed": deployed,
                    }
                )


def save_task_delay_csv(
    results: List[Tuple[str, SystemResult]],
    output_path: str | Path,
) -> None:
    """Save per-task delay breakdowns."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Method",
        "Task",
        "AccessUAV",
        "D_total_ms",
        "D_access_ms",
        "D_compute_ms",
        "D_trans_ms",
        "D_return_ms",
        "Substitutions",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, result in results:
            for plan in result.execution_plans:
                substitutions = sum(1 for step in plan.steps if step.is_substituted)
                row = {
                    "Method": method,
                    "Task": plan.task_id,
                    "AccessUAV": plan.access_uav,
                    "D_total_ms": plan.D_total * 1e3,
                    "D_access_ms": plan.D_access * 1e3,
                    "D_compute_ms": plan.D_compute * 1e3,
                    "D_trans_ms": plan.D_trans * 1e3,
                    "D_return_ms": plan.D_return * 1e3,
                    "Substitutions": substitutions,
                }
                writer.writerow({key: _fmt(row[key]) for key in fields})


def export_real_moe_results(
    results: List[Tuple[str, SystemResult]],
    output_dir: str | Path = "results/real_moe",
) -> None:
    """Write all real MoE experiment CSV artifacts."""

    root = Path(output_dir)
    save_summary_csv(results, root / "real_moe_summary.csv")
    save_deployment_csv(results, root / "real_moe_deployment.csv")
    save_task_delay_csv(results, root / "real_moe_task_delays.csv")


def summarize_results(results: Iterable[Tuple[str, SystemResult]]) -> str:
    """Return a short terminal summary."""

    lines = ["Real MoE trace experiment:"]
    for name, result in results:
        metrics = compute_metrics(name, result)
        lines.append(
            f"  {name}: D_total={metrics.D_total * 1e3:.2f} ms, "
            f"subs={metrics.substitutions}, deployments={metrics.deployments}"
        )
    return "\n".join(lines)

