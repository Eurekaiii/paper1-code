"""Parameter-weight sensitivity experiments for AeroMoDE.

This module sweeps key Table-III algorithm weights in both the controlled
synthetic scenario and the real-MoE UAV scenario. Outputs are grouped under a
dedicated results directory so they can be cited independently from ablations.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

from .config import SystemConfig
from .evaluation import compute_metrics
from .models import Expert, SystemResult, Task, UAV
from .pipeline import run_pipeline
from .real_moe.scenario_builder import build_real_moe_trace_scenario
from .scenario import build_controlled_random_hotspot_scenario_v2


Scenario = Tuple[List[UAV], List[Expert], List[Task]]
ConfigTransform = Callable[[SystemConfig], SystemConfig]


PLACEMENT_PROFILES: Dict[str, Tuple[float, float, float, float]] = {
    "default": (0.40, 0.25, 0.20, 0.15),
    "demand_heavy": (0.55, 0.20, 0.15, 0.10),
    "compute_heavy": (0.30, 0.45, 0.15, 0.10),
    "network_heavy": (0.30, 0.20, 0.40, 0.10),
    "memory_heavy": (0.30, 0.20, 0.15, 0.35),
}

DEFAULT_SWEEP_VALUES = {
    "redundancy_nu": "0.3",
    "scheduling_lambda": "0.1",
    "remote_eta": "0.5",
    "placement_profile": "default",
}


def _fmt(value: object) -> object:
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.6f}"
        return str(value)
    return value


def _result_or_infeasible(
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
    label: str,
) -> SystemResult:
    try:
        return run_pipeline(uavs, experts, tasks, cfg)
    except RuntimeError as exc:
        print(f"{label} infeasible: {exc}")
        return SystemResult(
            deployment={},
            execution_plans=[],
            D_total=float("inf"),
            D_weighted=float("inf"),
        )


def _row(
    scenario_name: str,
    sweep: str,
    value: str,
    result: SystemResult,
) -> Dict[str, object]:
    metrics = compute_metrics(value, result)
    return {
        "Scenario": scenario_name,
        "Sweep": sweep,
        "Value": value,
        "D_total_ms": metrics.D_total * 1e3,
        "Substitutions": metrics.substitutions,
        "Deployments": metrics.deployments,
        "AvgAccess_ms": metrics.avg_D_access * 1e3,
        "AvgCompute_ms": metrics.avg_D_compute * 1e3,
        "AvgTrans_ms": metrics.avg_D_trans * 1e3,
        "AvgReturn_ms": metrics.avg_D_return * 1e3,
        "Feasible": metrics.feasible,
    }


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Scenario",
        "Sweep",
        "Value",
        "D_total_ms",
        "Substitutions",
        "Deployments",
        "AvgAccess_ms",
        "AvgCompute_ms",
        "AvgTrans_ms",
        "AvgReturn_ms",
        "Feasible",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row[field]) for field in fields})


def _sweep_values(
    scenario_name: str,
    scenario: Scenario,
    base_cfg: SystemConfig,
    sweep_name: str,
    values: List[float],
    apply_value: Callable[[SystemConfig, float], None],
) -> List[Dict[str, object]]:
    uavs, experts, tasks = scenario
    rows: List[Dict[str, object]] = []
    for value in values:
        cfg = deepcopy(base_cfg)
        apply_value(cfg, value)
        label = f"{scenario_name}/{sweep_name}={value:g}"
        result = _result_or_infeasible(uavs, experts, tasks, cfg, label)
        rows.append(_row(scenario_name, sweep_name, f"{value:g}", result))
    return rows


def _sweep_placement_profiles(
    scenario_name: str,
    scenario: Scenario,
    base_cfg: SystemConfig,
) -> List[Dict[str, object]]:
    uavs, experts, tasks = scenario
    rows: List[Dict[str, object]] = []
    for name, (alpha, beta, gamma, mu) in PLACEMENT_PROFILES.items():
        cfg = deepcopy(base_cfg)
        cfg.deployment.alpha = alpha
        cfg.deployment.beta = beta
        cfg.deployment.gamma = gamma
        cfg.deployment.mu = mu
        result = _result_or_infeasible(
            uavs,
            experts,
            tasks,
            cfg,
            f"{scenario_name}/placement_profile={name}",
        )
        rows.append(_row(scenario_name, "placement_profile", name, result))
    return rows


def _run_one_scenario(
    scenario_name: str,
    scenario: Scenario,
    base_cfg: SystemConfig,
    output_dir: Path,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    sweeps = [
        (
            "redundancy_nu",
            [0.0, 0.1, 0.3, 0.5, 0.7],
            lambda cfg, value: setattr(cfg.deployment, "nu", value),
        ),
        (
            "scheduling_lambda",
            [0.0, 0.05, 0.1, 0.2, 0.4],
            lambda cfg, value: setattr(cfg.scheduling, "lamb", value),
        ),
        (
            "remote_eta",
            [0.0, 0.25, 0.5, 0.75, 1.0],
            lambda cfg, value: setattr(cfg.deployment, "eta", value),
        ),
    ]

    for sweep_name, values, apply_value in sweeps:
        sweep_rows = _sweep_values(
            scenario_name,
            scenario,
            base_cfg,
            sweep_name,
            values,
            apply_value,
        )
        _write_csv(sweep_rows, output_dir / f"{sweep_name}.csv")
        rows.extend(sweep_rows)

    profile_rows = _sweep_placement_profiles(scenario_name, scenario, base_cfg)
    _write_csv(profile_rows, output_dir / "placement_profile.csv")
    rows.extend(profile_rows)

    _write_csv(rows, output_dir / "weight_sensitivity_summary.csv")
    return rows


def _write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AeroMoDE Weight Sensitivity",
        "",
        "| Scenario | Sweep | Best value | Best delay (ms) | Default delay (ms) | Range (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["Scenario"]), str(row["Sweep"])), []).append(row)

    for (scenario, sweep), group in sorted(grouped.items()):
        finite = [row for row in group if np.isfinite(float(row["D_total_ms"]))]
        if not finite:
            continue
        best = min(finite, key=lambda row: float(row["D_total_ms"]))
        delays = [float(row["D_total_ms"]) for row in finite]
        default_value = DEFAULT_SWEEP_VALUES.get(sweep)
        default = next((row for row in finite if row["Value"] == default_value), None)
        default_text = f"{float(default['D_total_ms']):.2f}" if default else "n/a"
        lines.append(
            f"| {scenario} | {sweep} | {best['Value']} | "
            f"{float(best['D_total_ms']):.2f} | {default_text} | "
            f"{min(delays):.2f}-{max(delays):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_weight_sensitivity(
    output_dir: str | Path = "results/weight_sensitivity",
    trace_dir: str | Path = "data/real_world/moe/switch_base_8_trace",
    uav_scenario: str | Path = "data/real_world/uav/uav_edge_scenario_4node.json",
    expert_memory_scale: float = 80.0,
) -> None:
    root = Path(output_dir)
    all_rows: List[Dict[str, object]] = []

    synthetic_cfg = SystemConfig()
    synthetic_cfg.max_copies_per_expert = 1
    synthetic = build_controlled_random_hotspot_scenario_v2(synthetic_cfg)
    all_rows.extend(
        _run_one_scenario(
            "synthetic_hotspot_v2",
            synthetic,
            synthetic_cfg,
            root / "synthetic",
        )
    )

    real_cfg = SystemConfig()
    real_cfg.max_copies_per_expert = 1
    real_world = build_real_moe_trace_scenario(
        real_cfg,
        trace_dir=trace_dir,
        uav_scenario_path=uav_scenario,
        expert_memory_scale=expert_memory_scale,
    )
    all_rows.extend(
        _run_one_scenario(
            "real_moe_uav",
            real_world,
            real_cfg,
            root / "real_world",
        )
    )

    _write_csv(all_rows, root / "weight_sensitivity_all.csv")
    _write_markdown(all_rows, root / "README.md")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AeroMoDE weight sensitivity sweeps.")
    parser.add_argument("--output-dir", default="results/weight_sensitivity")
    parser.add_argument("--trace-dir", default="data/real_world/moe/switch_base_8_trace")
    parser.add_argument("--uav-scenario", default="data/real_world/uav/uav_edge_scenario_4node.json")
    parser.add_argument("--expert-memory-scale", type=float, default=80.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_weight_sensitivity(
        output_dir=args.output_dir,
        trace_dir=args.trace_dir,
        uav_scenario=args.uav_scenario,
        expert_memory_scale=args.expert_memory_scale,
    )
    print(f"Wrote weight sensitivity results to {args.output_dir}")


if __name__ == "__main__":
    main()
