"""Run real MoE trace experiments for fig1-fig9 style plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..config import SystemConfig
from ..evaluation import compute_metrics, run_all_methods
from ..models import Expert, Task, UAV
from .scenario_builder import build_real_moe_trace_scenario


METHOD_ORDER = [
    "Proposed",
    "Random Placement",
    "Importance-based Placement",
    "No-similarity Placement",
]


def _save_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "value",
        "method",
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


def _metrics_rows(
    experiment: str,
    value: float,
    results,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for name, result in results:
        metric = compute_metrics(name, result)
        rows.append(
            {
                "experiment": experiment,
                "value": float(value),
                "method": name,
                "D_total_ms": metric.D_total * 1e3,
                "Substitutions": metric.substitutions,
                "Deployments": metric.deployments,
                "AvgAccess_ms": metric.avg_D_access * 1e3,
                "AvgCompute_ms": metric.avg_D_compute * 1e3,
                "AvgTrans_ms": metric.avg_D_trans * 1e3,
                "AvgReturn_ms": metric.avg_D_return * 1e3,
                "Feasible": metric.feasible,
            }
        )
    return rows


def _grid_positions(num_points: int, area_size: float = 240.0) -> List[np.ndarray]:
    side = int(np.ceil(np.sqrt(num_points)))
    margin = area_size / (side + 1)
    coords = np.linspace(margin, area_size - margin, side)
    positions: List[np.ndarray] = []
    for y in coords:
        for x in coords:
            positions.append(np.array([x, y, 80.0], dtype=np.float64))
            if len(positions) == num_points:
                return positions
    return positions


def _load_base_uav_specs(path: str | Path) -> List[Dict[str, float]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["uavs"]


def _make_uavs(
    uav_count: int,
    scenario_path: str | Path,
    memory_scale: float = 1.0,
) -> List[UAV]:
    base = _load_base_uav_specs(scenario_path)
    positions = _grid_positions(uav_count)
    uavs: List[UAV] = []
    for idx in range(uav_count):
        spec = base[idx % len(base)]
        uavs.append(
            UAV(
                id=idx,
                position=positions[idx],
                C_u=float(spec["C_u"]),
                M_u=float(spec["M_u"]) * memory_scale,
                P_u=float(spec["P_u"]),
            )
        )
    return uavs


def _clone_task(task: Task, new_id: int | None = None) -> Task:
    return Task(
        id=task.id if new_id is None else new_id,
        position=np.array(task.position, dtype=np.float64),
        S_in=float(task.S_in),
        S_out=float(task.S_out),
        S_mid=[float(value) for value in task.S_mid],
        expert_sequence=[int(value) for value in task.expert_sequence],
        omega=float(task.omega),
        gating_scores=(
            {int(key): float(value) for key, value in task.gating_scores.items()}
            if task.gating_scores
            else None
        ),
    )


def _scale_mid_sizes(tasks: List[Task], scale: float) -> List[Task]:
    out = [_clone_task(task) for task in tasks]
    for task in out:
        task.S_mid = [value * scale for value in task.S_mid]
    return out


def _limit_tasks(tasks: List[Task], count: int) -> List[Task]:
    if count <= len(tasks):
        return [_clone_task(task, idx) for idx, task in enumerate(tasks[:count])]
    out: List[Task] = []
    for idx in range(count):
        task = _clone_task(tasks[idx % len(tasks)], idx)
        out.append(task)
    return out


def _limit_experts(
    experts: List[Expert],
    tasks: List[Task],
    expert_count: int,
) -> Tuple[List[Expert], List[Task]]:
    allowed = {expert.id for expert in sorted(experts, key=lambda item: item.id)[:expert_count]}
    limited_experts = [expert for expert in experts if expert.id in allowed]
    limited_tasks: List[Task] = []
    for task in tasks:
        seq = [expert_id for expert_id in task.expert_sequence if expert_id in allowed]
        if not seq:
            continue
        new_task = _clone_task(task, len(limited_tasks))
        new_task.expert_sequence = seq
        new_task.S_mid = (new_task.S_mid[: len(seq)] if new_task.S_mid else [1e5] * len(seq))
        if len(new_task.S_mid) < len(seq):
            new_task.S_mid.extend([new_task.S_mid[-1]] * (len(seq) - len(new_task.S_mid)))
        if new_task.gating_scores:
            new_task.gating_scores = {
                expert_id: score
                for expert_id, score in new_task.gating_scores.items()
                if expert_id in allowed
            }
        limited_tasks.append(new_task)
    return limited_experts, limited_tasks


def _build_base(
    cfg: SystemConfig,
    trace_dir: str | Path,
    uav_scenario: str | Path,
    expert_memory_scale: float,
    memory_scale: float,
) -> Tuple[List[UAV], List[Expert], List[Task]]:
    return build_real_moe_trace_scenario(
        cfg,
        trace_dir=trace_dir,
        uav_scenario_path=uav_scenario,
        memory_scale=memory_scale,
        expert_memory_scale=expert_memory_scale,
    )


def _run_one(
    experiment: str,
    value: float,
    trace_dir: str | Path,
    uav_scenario: str | Path,
    expert_memory_scale: float,
    memory_scale: float = 1.0,
    xi: float | None = None,
    mid_scale: float = 1.0,
    uav_count: int | None = None,
    task_count: int | None = None,
    expert_count: int | None = None,
) -> List[Dict[str, object]]:
    cfg = SystemConfig()
    cfg.max_copies_per_expert = 1
    if xi is not None:
        cfg.similarity.xi = xi

    uavs, experts, tasks = _build_base(
        cfg,
        trace_dir=trace_dir,
        uav_scenario=uav_scenario,
        expert_memory_scale=expert_memory_scale,
        memory_scale=memory_scale,
    )
    if uav_count is not None:
        uavs = _make_uavs(uav_count, uav_scenario, memory_scale=memory_scale)
    if task_count is not None:
        tasks = _limit_tasks(tasks, task_count)
    if mid_scale != 1.0:
        tasks = _scale_mid_sizes(tasks, mid_scale)
    if expert_count is not None:
        experts, tasks = _limit_experts(experts, tasks, expert_count)

    results = run_all_methods(uavs, experts, tasks, cfg)
    return _metrics_rows(experiment, value, results)


def run_all_real_world_sweeps(
    output_dir: str | Path = "results/real_world/fig1_9",
    trace_dir: str | Path = "data/real_world/moe/switch_base_8_trace",
    uav_scenario: str | Path = "data/real_world/uav/uav_edge_scenario_4node.json",
    expert_memory_scale: float = 80.0,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Fig1-3 base data.
    base_rows = _run_one(
        "base",
        1.0,
        trace_dir,
        uav_scenario,
        expert_memory_scale,
    )
    _save_csv(base_rows, out / "real_world_base.csv")

    rows: List[Dict[str, object]] = []
    for xi in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        rows.extend(
            _run_one(
                "xi",
                xi,
                trace_dir,
                uav_scenario,
                expert_memory_scale,
                xi=xi,
            )
        )
    _save_csv(rows, out / "real_world_sensitivity_xi.csv")

    rows = []
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        rows.extend(
            _run_one(
                "mid_scale",
                scale,
                trace_dir,
                uav_scenario,
                expert_memory_scale,
                mid_scale=scale,
            )
        )
    _save_csv(rows, out / "real_world_sensitivity_mid_size.csv")

    rows = []
    for scale in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]:
        rows.extend(
            _run_one(
                "memory_scale",
                scale,
                trace_dir,
                uav_scenario,
                expert_memory_scale,
                memory_scale=scale,
            )
        )
    _save_csv(rows, out / "real_world_sensitivity_memory.csv")

    rows = []
    for count in [4, 6, 8, 10, 12]:
        rows.extend(
            _run_one(
                "uav_count",
                float(count),
                trace_dir,
                uav_scenario,
                expert_memory_scale,
                uav_count=count,
            )
        )
    _save_csv(rows, out / "real_world_uav_count.csv")

    rows = []
    for count in [10, 20, 30, 40, 60, 80]:
        rows.extend(
            _run_one(
                "task_count",
                float(count),
                trace_dir,
                uav_scenario,
                expert_memory_scale,
                task_count=count,
            )
        )
    _save_csv(rows, out / "real_world_task_count.csv")

    rows = []
    for count in [16, 32, 48, 64, 80, 96]:
        rows.extend(
            _run_one(
                "expert_count",
                float(count),
                trace_dir,
                uav_scenario,
                expert_memory_scale,
                expert_count=count,
            )
        )
    _save_csv(rows, out / "real_world_expert_count.csv")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-world MoE fig1-fig9 sweeps.")
    parser.add_argument("--output-dir", default="results/real_world/fig1_9")
    parser.add_argument("--trace-dir", default="data/real_world/moe/switch_base_8_trace")
    parser.add_argument("--uav-scenario", default="data/real_world/uav/uav_edge_scenario_4node.json")
    parser.add_argument("--expert-memory-scale", type=float, default=80.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all_real_world_sweeps(
        output_dir=args.output_dir,
        trace_dir=args.trace_dir,
        uav_scenario=args.uav_scenario,
        expert_memory_scale=args.expert_memory_scale,
    )
    print(f"Wrote real-world fig1-fig9 CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()

