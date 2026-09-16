"""Ablation experiments for AeroMoDE.

The ablations keep the same scheduler and placement pipeline while disabling
one design component at a time:

- w/o Similarity: disable substitute experts by setting xi = 1.
- w/o Communication: remove remote demand and topology terms from placement.
- w/o Redundancy: remove the dynamic redundancy penalty.
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
from .scenario import build_controlled_random_hotspot_scenario_v2
from .real_moe.scenario_builder import build_real_moe_trace_scenario


Scenario = Tuple[List[UAV], List[Expert], List[Task]]
Variant = Tuple[str, Callable[[SystemConfig], SystemConfig]]


def _full(cfg: SystemConfig) -> SystemConfig:
    return deepcopy(cfg)


def _without_similarity(cfg: SystemConfig) -> SystemConfig:
    out = deepcopy(cfg)
    out.similarity.xi = 1.0
    return out


def _without_communication(cfg: SystemConfig) -> SystemConfig:
    out = deepcopy(cfg)
    # Remove communication-aware placement components while preserving local
    # demand, compute capability, and memory pressure. The removed topology
    # weight is assigned to local demand so the score scale remains comparable.
    out.deployment.eta = 0.0
    out.deployment.alpha += out.deployment.gamma
    out.deployment.gamma = 0.0
    return out


def _without_redundancy(cfg: SystemConfig) -> SystemConfig:
    out = deepcopy(cfg)
    out.deployment.nu = 0.0
    return out


VARIANTS: List[Variant] = [
    ("Full AeroMoE", _full),
    ("w/o Similarity", _without_similarity),
    ("w/o Communication", _without_communication),
    ("w/o Redundancy", _without_redundancy),
]


def _fmt(value: object) -> object:
    if isinstance(value, float):
        if np.isfinite(value):
            return f"{value:.6f}"
        return str(value)
    return value


def _run_variants(
    base_cfg: SystemConfig,
    scenario: Scenario,
) -> List[Tuple[str, SystemResult]]:
    uavs, experts, tasks = scenario
    rows: List[Tuple[str, SystemResult]] = []
    for name, transform in VARIANTS:
        cfg = transform(base_cfg)
        try:
            result = run_pipeline(uavs, experts, tasks, cfg)
        except RuntimeError as exc:
            print(f"{name} infeasible: {exc}")
            result = SystemResult(
                deployment={},
                execution_plans=[],
                D_total=float("inf"),
                D_weighted=float("inf"),
            )
        rows.append((name, result))
    return rows


def _summary_rows(
    scenario_name: str,
    results: List[Tuple[str, SystemResult]],
) -> List[Dict[str, object]]:
    full_total = None
    rows: List[Dict[str, object]] = []
    for name, result in results:
        metrics = compute_metrics(name, result)
        if name == "Full AeroMoE":
            full_total = metrics.D_total
        rows.append(
            {
                "Scenario": scenario_name,
                "Variant": name,
                "D_total_ms": metrics.D_total * 1e3,
                "Delta_vs_full_pct": 0.0,
                "Substitutions": metrics.substitutions,
                "Deployments": metrics.deployments,
                "AvgAccess_ms": metrics.avg_D_access * 1e3,
                "AvgCompute_ms": metrics.avg_D_compute * 1e3,
                "AvgTrans_ms": metrics.avg_D_trans * 1e3,
                "AvgReturn_ms": metrics.avg_D_return * 1e3,
                "Feasible": metrics.feasible,
            }
        )

    if full_total is not None and np.isfinite(full_total) and full_total > 0:
        for row in rows:
            total = float(row["D_total_ms"]) / 1e3
            row["Delta_vs_full_pct"] = (total - full_total) / full_total * 100.0
    return rows


def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Scenario",
        "Variant",
        "D_total_ms",
        "Delta_vs_full_pct",
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


def _write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# AeroMoDE Ablation Summary",
        "",
        "| Scenario | Variant | Total delay (ms) | Delta vs full | Substitutions | Deployments | Feasible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delay = row["D_total_ms"]
        delta = row["Delta_vs_full_pct"]
        delay_text = f"{delay:.2f}" if isinstance(delay, float) and np.isfinite(delay) else str(delay)
        delta_text = f"{delta:.2f}%" if isinstance(delta, float) and np.isfinite(delta) else str(delta)
        lines.append(
            f"| {row['Scenario']} | {row['Variant']} | {delay_text} | "
            f"{delta_text} | {row['Substitutions']} | {row['Deployments']} | {row['Feasible']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablations(
    output_dir: str | Path = "results/ablations",
    trace_dir: str | Path = "data/real_world/moe/switch_base_8_trace",
    uav_scenario: str | Path = "data/real_world/uav/uav_edge_scenario_4node.json",
    expert_memory_scale: float = 80.0,
) -> None:
    root = Path(output_dir)
    all_rows: List[Dict[str, object]] = []

    synthetic_cfg = SystemConfig()
    synthetic_cfg.max_copies_per_expert = 1
    synthetic = build_controlled_random_hotspot_scenario_v2(synthetic_cfg)
    synthetic_results = _run_variants(synthetic_cfg, synthetic)
    synthetic_rows = _summary_rows("synthetic_hotspot_v2", synthetic_results)
    _write_csv(synthetic_rows, root / "synthetic" / "ablation_summary.csv")
    all_rows.extend(synthetic_rows)

    real_cfg = SystemConfig()
    real_cfg.max_copies_per_expert = 1
    real_world = build_real_moe_trace_scenario(
        real_cfg,
        trace_dir=trace_dir,
        uav_scenario_path=uav_scenario,
        expert_memory_scale=expert_memory_scale,
    )
    real_results = _run_variants(real_cfg, real_world)
    real_rows = _summary_rows("real_moe_uav", real_results)
    _write_csv(real_rows, root / "real_world" / "ablation_summary.csv")
    all_rows.extend(real_rows)

    _write_csv(all_rows, root / "ablation_summary_all.csv")
    _write_markdown(all_rows, root / "README.md")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AeroMoDE ablation experiments.")
    parser.add_argument("--output-dir", default="results/ablations")
    parser.add_argument("--trace-dir", default="data/real_world/moe/switch_base_8_trace")
    parser.add_argument("--uav-scenario", default="data/real_world/uav/uav_edge_scenario_4node.json")
    parser.add_argument("--expert-memory-scale", type=float, default=80.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_ablations(
        output_dir=args.output_dir,
        trace_dir=args.trace_dir,
        uav_scenario=args.uav_scenario,
        expert_memory_scale=args.expert_memory_scale,
    )
    print(f"Wrote ablation results to {args.output_dir}")


if __name__ == "__main__":
    main()
