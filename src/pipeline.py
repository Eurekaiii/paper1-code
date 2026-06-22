"""
End-to-end AeroMoDE pipeline.

This module consumes already-built UAVs, experts, tasks, and configuration,
then performs communication precomputation, demand modeling, placement, and
scheduling.
"""

from typing import List

import numpy as np

from .communication import build_access_info, build_connectivity_matrix
from .config import SystemConfig
from .demand import (
    compute_direct_demand,
    compute_effective_demand,
    compute_required_experts,
)
from .models import Expert, SystemResult, Task, UAV
from .placement import get_deployed_experts_by_uav, greedy_placement
from .scheduling import schedule_all_tasks
from .scoring import compute_base_scores
from .similarity import (
    build_candidate_set,
    build_substitutable_sets,
    compute_similarity_matrix,
    strict_substitute_set,
)


def run_pipeline(
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Execute the complete AeroMoDE pipeline."""
    ch_cfg = cfg.channel
    dep_cfg = cfg.deployment
    sched_cfg = cfg.scheduling
    sim_cfg = cfg.similarity

    num_uavs = len(uavs)
    uav_ids = [u.id for u in uavs]
    uav_indices = {u.id: i for i, u in enumerate(uavs)}

    G, Rmat = build_connectivity_matrix(uavs, ch_cfg)
    R_vals = [
        Rmat[i, j]
        for i in range(num_uavs)
        for j in range(num_uavs)
        if i != j and G[i, j] == 1
    ]
    R_max = max(R_vals) if R_vals else 1.0

    access_uavs, access_delays = build_access_info(tasks, uavs, ch_cfg)
    demand = compute_direct_demand(tasks, uavs, access_uavs)
    required_experts = compute_required_experts(tasks)

    similarity = compute_similarity_matrix(experts)
    substitutable_sets = build_substitutable_sets(
        similarity,
        required_experts,
        sim_cfg.xi,
    )
    candidates = build_candidate_set(substitutable_sets)
    demand_eff = compute_effective_demand(
        demand,
        substitutable_sets,
        similarity,
        uav_ids,
    )

    C_map = {u.id: u.C_u for u in uavs}
    F_map = {e.id: e.F_e for e in experts}
    W_map = {e.id: e.W_e for e in experts}
    M_map = {u.id: u.M_u for u in uavs}

    base_scores = compute_base_scores(
        candidates=candidates,
        uav_ids=uav_ids,
        demand_eff=demand_eff,
        G=G,
        R=Rmat,
        R_max=R_max,
        C_map=C_map,
        F_map=F_map,
        W_map=W_map,
        uav_indices=uav_indices,
        cfg=dep_cfg,
    )

    strict_substitutes = {
        e: strict_substitute_set(e, substitutable_sets) for e in candidates
    }
    deployment = greedy_placement(
        candidates=candidates,
        uav_ids=uav_ids,
        base_scores=base_scores,
        W_map=W_map,
        M_map=M_map,
        similarity=similarity,
        strict_substitutes=strict_substitutes,
        G=G,
        uav_indices=uav_indices,
        cfg=dep_cfg,
        max_copies_per_expert=cfg.max_copies_per_expert,
        required_experts=required_experts,
        demand_eff=demand_eff,
        substitutable_sets=substitutable_sets,
    )

    deployed_by_uav = get_deployed_experts_by_uav(deployment)
    plans, D_weighted = schedule_all_tasks(
        tasks=tasks,
        access_assignment=access_uavs,
        access_delays=access_delays,
        uavs=uavs,
        deployment=deployment,
        deployed_by_uav=deployed_by_uav,
        substitutable_sets=substitutable_sets,
        similarity=similarity,
        G=G,
        R=Rmat,
        uav_indices=uav_indices,
        F_map=F_map,
        C_map=C_map,
        cfg=sched_cfg,
        comm_cfg=ch_cfg,
    )

    return SystemResult(
        deployment=deployment,
        execution_plans=plans,
        D_total=sum(p.D_total for p in plans),
        D_weighted=D_weighted,
    )


def print_report(result: SystemResult, uavs: List[UAV], experts: List[Expert]) -> None:
    """Print a human-readable summary of the system result."""
    W_map = {e.id: e.W_e for e in experts}
    M_map = {u.id: u.M_u for u in uavs}
    access_counts = {u.id: 0 for u in uavs}
    for plan in result.execution_plans:
        access_counts[plan.access_uav] = access_counts.get(plan.access_uav, 0) + 1

    print("=" * 72)
    print("  AeroMoDE Pipeline Result")
    print("=" * 72)

    print("\n--- UAV Parameters ---")
    for u in uavs:
        print(
            f"  UAV {u.id}: pos=({u.x:.1f}, {u.y:.1f}, {u.H:.1f}) m  "
            f"C={u.C_u / 1e9:.2f} GFLOP/s  "
            f"M={u.M_u / 1e9:.2f} GB  "
            f"P={u.P_u:.2f} W"
        )

    print("\n--- Default Access Task Counts ---")
    for u in uavs:
        print(f"  UAV {u.id}: {access_counts.get(u.id, 0)} tasks")

    print("\n--- Expert Deployment ---")
    deployed = [(e, u) for (e, u), m in result.deployment.items() if m == 1]
    if not deployed:
        print("  (no experts deployed)")
    else:
        for e, u in sorted(deployed, key=lambda item: (item[1], item[0])):
            w = W_map.get(e, 0) / 1e6
            print(f"  Expert E{e} -> UAV {u}   ({w:.0f} MB)")

    print("\n--- UAV Memory Usage ---")
    for u in uavs:
        used = sum(
            W_map.get(e, 0)
            for (e, u_id), m in result.deployment.items()
            if m == 1 and u_id == u.id
        )
        cap = M_map.get(u.id, 1)
        print(
            f"  UAV {u.id}:  {used / 1e6:.0f} / {cap / 1e6:.0f} MB  "
            f"({100 * used / cap:.1f}%)"
        )

    print("\n--- Task Execution Plans ---")
    for plan in result.execution_plans:
        print(f"\n  Task {plan.task_id}  (access UAV {plan.access_uav})")
        for step in plan.steps:
            sub = "SUB" if step.is_substituted else "ORG"
            print(
                f"    Layer {step.layer_idx}:  "
                f"want=E{step.original_expert} -> use=E{step.actual_expert}  "
                f"on UAV {step.uav_id}  [{sub}]"
            )
            print(
                f"      trans={step.cost_transmission * 1e3:.2f} ms  "
                f"comp={step.cost_computation * 1e3:.2f} ms  "
                f"err={step.cost_error:.4f}"
            )
        print(
            f"    TOTAL D_k = {plan.D_total * 1e3:.2f} ms  "
            f"(acc={plan.D_access * 1e3:.2f}  "
            f"cmp={plan.D_compute * 1e3:.2f}  "
            f"tx={plan.D_trans * 1e3:.2f}  "
            f"ret={plan.D_return * 1e3:.2f})"
        )

    print("\n--- Overall ---")
    print(f"  D_total (unweighted)  = {result.D_total * 1e3:.2f} ms")
    print(f"  D_total (weighted)    = {result.D_weighted * 1e3:.2f} ms")
    print("=" * 72)
