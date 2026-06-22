"""Evaluation helpers for comparing Proposed and baseline methods."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .baselines import (
    run_importance_placement,
    run_no_similarity_placement,
    run_random_placement,
)
from .config import SystemConfig
from .models import Expert, SystemResult, Task, UAV
from .pipeline import run_pipeline


@dataclass
class MethodMetrics:
    """Aggregated delay metrics for one method."""

    name: str
    D_total: float
    substitutions: int
    deployments: int
    avg_D_access: float
    avg_D_compute: float
    avg_D_trans: float
    avg_D_return: float
    feasible: bool = True


def compute_metrics(name: str, result: SystemResult) -> MethodMetrics:
    """Compute table metrics from a system result."""
    plans = result.execution_plans
    deployments = sum(result.deployment.values())
    if not plans:
        return MethodMetrics(
            name,
            np.inf,
            0,
            deployments,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            False,
        )

    return MethodMetrics(
        name=name,
        D_total=result.D_total,
        substitutions=sum(
            1 for plan in plans for step in plan.steps if step.is_substituted
        ),
        deployments=deployments,
        avg_D_access=float(np.mean([p.D_access for p in plans])),
        avg_D_compute=float(np.mean([p.D_compute for p in plans])),
        avg_D_trans=float(np.mean([p.D_trans for p in plans])),
        avg_D_return=float(np.mean([p.D_return for p in plans])),
        feasible=np.isfinite(result.D_total),
    )


def count_substitution_pairs(result: SystemResult) -> Dict[Tuple[int, int], int]:
    """Count how often each original expert is replaced by another expert."""
    counts: Dict[Tuple[int, int], int] = {}
    for plan in result.execution_plans:
        for step in plan.steps:
            if not step.is_substituted:
                continue
            pair = (step.original_expert, step.actual_expert)
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def run_all_methods(
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> List[Tuple[str, SystemResult]]:
    """Run Proposed and the three requested baselines on one shared scenario."""
    methods: List[Tuple[str, Callable[[], SystemResult]]] = [
        ("Proposed", lambda: run_pipeline(uavs, experts, tasks, cfg)),
        ("Random Placement", lambda: run_random_placement(uavs, experts, tasks, cfg)),
        (
            "Importance-based Placement",
            lambda: run_importance_placement(uavs, experts, tasks, cfg),
        ),
        (
            "No-similarity Placement",
            lambda: run_no_similarity_placement(uavs, experts, tasks, cfg),
        ),
    ]

    results: List[Tuple[str, SystemResult]] = []
    for name, run_method in methods:
        try:
            results.append((name, run_method()))
        except RuntimeError as exc:
            print(f"{name} infeasible under strict single-hop scheduling: {exc}")
            results.append(
                (
                    name,
                    SystemResult(
                        deployment={},
                        execution_plans=[],
                        D_total=float("inf"),
                        D_weighted=float("inf"),
                    ),
                )
            )
    return results


def run_multi_seed_evaluation(
    seeds: List[int],
    scenario_builder: Callable[[SystemConfig], Tuple[List[UAV], List[Expert], List[Task]]],
) -> Dict[str, List[MethodMetrics]]:
    """Run all methods on multiple seeds and return per-method metrics."""
    metrics_by_method: Dict[str, List[MethodMetrics]] = {}

    for seed in seeds:
        cfg = SystemConfig()
        cfg.seed = seed
        uavs, experts, tasks = scenario_builder(cfg)
        results = run_all_methods(uavs, experts, tasks, cfg)
        for name, result in results:
            metrics_by_method.setdefault(name, []).append(compute_metrics(name, result))

    return metrics_by_method
