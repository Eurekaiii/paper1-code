"""
Batch Experiment Runner
=========================
Runs multiple scenarios × baselines, collects results, saves to JSON.
"""

from __future__ import annotations
import json
import time
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

import numpy as np

from src.config import SystemConfig
from src.models import SystemResult, TaskExecutionPlan
from experiments.scenarios import Scenario
from experiments.baselines import BASELINES


# ======================================================================
# Result containers
# ======================================================================
@dataclass
class ScenarioResult:
    """Aggregated result for one scenario."""
    scenario_name: str
    num_uavs: int
    num_experts: int
    num_tasks: int
    bandwidth_mhz: float

    # Per-baseline metrics
    baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Raw execution plans (optional, for detailed analysis)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentReport:
    """Full experiment report."""
    timestamp: str
    scenarios: List[ScenarioResult]
    baseline_names: List[str]


# ======================================================================
# Metrics extraction
# ======================================================================
def extract_metrics(result: SystemResult, uavs, experts, tasks) \
        -> Dict[str, float]:
    """Extract key metrics from a SystemResult."""
    metrics: Dict[str, float] = {}

    # Delay
    metrics["D_total_ms"] = result.D_total * 1e3
    metrics["D_weighted_ms"] = result.D_weighted * 1e3
    metrics["D_avg_per_task_ms"] = (result.D_total / max(len(tasks), 1)) * 1e3

    # Decomposition
    total_access = sum(p.D_access for p in result.execution_plans)
    total_compute = sum(p.D_compute for p in result.execution_plans)
    total_trans = sum(p.D_trans for p in result.execution_plans)
    total_return = sum(p.D_return for p in result.execution_plans)
    metrics["D_access_ms"] = total_access * 1e3
    metrics["D_compute_ms"] = total_compute * 1e3
    metrics["D_trans_ms"] = total_trans * 1e3
    metrics["D_return_ms"] = total_return * 1e3

    # Substitution stats
    n_substituted = 0
    n_total_steps = 0
    for p in result.execution_plans:
        for s in p.steps:
            if s.is_substituted:
                n_substituted += 1
            n_total_steps += 1
    metrics["substitution_rate"] = (n_substituted / max(n_total_steps, 1))
    metrics["n_substituted"] = float(n_substituted)
    metrics["n_total_steps"] = float(n_total_steps)

    # Memory utilisation
    usage = {}
    for (e, u), m in result.deployment.items():
        if m == 1:
            w_e = next((exp.W_e for exp in experts if exp.id == e), 0)
            usage[u] = usage.get(u, 0.0) + w_e
    mem_utils = []
    for u in uavs:
        used = usage.get(u.id, 0.0)
        cap = u.M_u
        mem_utils.append(used / max(cap, 1))
    metrics["mem_util_mean"] = float(np.mean(mem_utils)) if mem_utils else 0.0
    metrics["mem_util_max"] = float(np.max(mem_utils)) if mem_utils else 0.0
    metrics["n_uavs_used"] = float(sum(1 for u in uavs if usage.get(u.id, 0) > 0))
    metrics["n_experts_deployed"] = float(sum(1 for v in result.deployment.values() if v == 1))

    return metrics


# ======================================================================
# Runner
# ======================================================================
def run_scenario(scenario: Scenario, baselines_to_run: List[str] = None,
                 verbose: bool = True) -> ScenarioResult:
    """Run a single scenario against all (or specified) baselines."""
    if baselines_to_run is None:
        baselines_to_run = list(BASELINES.keys())

    if verbose:
        print(f"  Scenario: {scenario.name}  "
              f"({scenario.cfg.channel.B/1e6:.0f} MHz, "
              f"{len(scenario.uavs)} UAVs, "
              f"{len(scenario.experts)} experts, "
              f"{len(scenario.tasks)} tasks)")

    sr = ScenarioResult(
        scenario_name=scenario.name,
        num_uavs=len(scenario.uavs),
        num_experts=len(scenario.experts),
        num_tasks=len(scenario.tasks),
        bandwidth_mhz=scenario.cfg.channel.B / 1e6,
    )

    for bname in baselines_to_run:
        if bname not in BASELINES:
            continue
        try:
            t0 = time.perf_counter()
            result = BASELINES[bname](
                scenario.uavs, scenario.experts, scenario.tasks, scenario.cfg,
            )
            elapsed = time.perf_counter() - t0

            metrics = extract_metrics(
                result, scenario.uavs, scenario.experts, scenario.tasks,
            )
            metrics["runtime_s"] = elapsed
            sr.baselines[bname] = metrics

            if verbose:
                print(f"    {bname:<20s}: D_total={metrics['D_total_ms']:.1f} ms, "
                      f"sub_rate={metrics['substitution_rate']:.2%}, "
                      f"mem_util={metrics['mem_util_mean']:.1%}, "
                      f"({elapsed:.3f}s)")

        except Exception as e:
            if verbose:
                print(f"    {bname:<20s}: FAILED — {e}")
            sr.baselines[bname] = {"error": str(e)}

    return sr


def run_all_scenarios(
    scenarios: List[Scenario],
    baselines_to_run: List[str] = None,
    verbose: bool = True,
) -> ExperimentReport:
    """Run all scenarios and return a report."""
    if baselines_to_run is None:
        baselines_to_run = list(BASELINES.keys())

    results: List[ScenarioResult] = []
    n_total = len(scenarios)
    for i, sc in enumerate(scenarios):
        if verbose:
            print(f"\n[{i+1}/{n_total}]", end=" ")
        sr = run_scenario(sc, baselines_to_run, verbose)
        results.append(sr)

    return ExperimentReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        scenarios=results,
        baseline_names=baselines_to_run,
    )


# ======================================================================
# JSON serialisation
# ======================================================================
def report_to_dict(report: ExperimentReport) -> dict:
    """Convert report to JSON-serialisable dict."""
    return {
        "timestamp": report.timestamp,
        "baseline_names": report.baseline_names,
        "scenarios": [asdict(sr) for sr in report.scenarios],
    }


def save_report(report: ExperimentReport, filepath: str):
    """Save report to JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
                exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_to_dict(report), f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {filepath}")


def load_report(filepath: str) -> ExperimentReport:
    """Load report from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = ExperimentReport(
        timestamp=data["timestamp"],
        baseline_names=data["baseline_names"],
        scenarios=[],
    )
    for sc_data in data["scenarios"]:
        sr = ScenarioResult(
            scenario_name=sc_data["scenario_name"],
            num_uavs=sc_data["num_uavs"],
            num_experts=sc_data["num_experts"],
            num_tasks=sc_data["num_tasks"],
            bandwidth_mhz=sc_data["bandwidth_mhz"],
            baselines=sc_data["baselines"],
            raw=sc_data.get("raw", {}),
        )
        report.scenarios.append(sr)
    return report
