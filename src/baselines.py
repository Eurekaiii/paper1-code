"""Baseline placement methods.

All baselines share the same UAV/task/expert scenario and the same strict
single-hop greedy scheduler. They differ only in expert placement.
"""

from copy import deepcopy
from typing import Callable, Dict, List, Set, Tuple

import numpy as np

from .communication import build_access_info, build_connectivity_matrix
from .config import SystemConfig
from .demand import (
    compute_direct_demand,
    compute_effective_demand,
    compute_required_experts,
)
from .models import Expert, SystemResult, Task, UAV
from .pipeline import run_pipeline
from .placement import get_deployed_experts_by_uav
from .scheduling import schedule_all_tasks
from .similarity import (
    build_candidate_set,
    build_substitutable_sets,
    compute_similarity_matrix,
)


def _empty_deployment(
    candidates: Set[int],
    uav_ids: List[int],
) -> Dict[Tuple[int, int], int]:
    return {(e, u): 0 for e in candidates for u in uav_ids}


def _schedule_with_deployment(
    name: str,
    deployment: Dict[Tuple[int, int], int],
    substitutable_sets: Dict[int, Set[int]],
    similarity: Dict[Tuple[int, int], float],
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Run the shared strict single-hop scheduler for a fixed deployment."""
    G, Rmat = build_connectivity_matrix(uavs, cfg.channel)
    access_uavs, access_delays = build_access_info(tasks, uavs, cfg.channel)
    deployed_by_uav = get_deployed_experts_by_uav(deployment)

    uav_indices = {u.id: i for i, u in enumerate(uavs)}
    F_map = {e.id: e.F_e for e in experts}
    C_map = {u.id: u.C_u for u in uavs}

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
        cfg=cfg.scheduling,
        comm_cfg=cfg.channel,
    )

    return SystemResult(
        deployment=deployment,
        execution_plans=plans,
        D_total=sum(p.D_total for p in plans),
        D_weighted=D_weighted,
    )


def _prepare_similarity_context(
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> Tuple[
    List[int],
    Set[int],
    Dict[int, Set[int]],
    Set[int],
    Dict[Tuple[int, int], float],
    Dict[int, Dict[int, float]],
]:
    """Build common demand/similarity objects for placement baselines."""
    uav_ids = [u.id for u in uavs]
    access_uavs, _ = build_access_info(tasks, uavs, cfg.channel)
    demand = compute_direct_demand(tasks, uavs, access_uavs)
    required_experts = compute_required_experts(tasks)
    similarity = compute_similarity_matrix(experts)
    substitutable_sets = build_substitutable_sets(
        similarity,
        required_experts,
        cfg.similarity.xi,
    )
    candidates = build_candidate_set(substitutable_sets)
    demand_eff = compute_effective_demand(
        demand,
        substitutable_sets,
        similarity,
        uav_ids,
    )
    return (
        uav_ids,
        required_experts,
        substitutable_sets,
        candidates,
        similarity,
        demand_eff,
    )


def _place_required_originals(
    deployment: Dict[Tuple[int, int], int],
    required_experts: Set[int],
    uav_ids: List[int],
    W_map: Dict[int, float],
    remaining_memory: Dict[int, float],
    choose_uav: Callable[[int, List[int]], int],
) -> None:
    """Ensure every required original expert is deployed if memory permits."""
    for e in sorted(required_experts, key=lambda x: W_map.get(x, 0.0), reverse=True):
        feasible_uavs = [u for u in uav_ids if remaining_memory[u] >= W_map[e]]
        if not feasible_uavs:
            continue
        u = choose_uav(e, feasible_uavs)
        deployment[(e, u)] = 1
        remaining_memory[u] -= W_map[e]


def _copy_count(
    deployment: Dict[Tuple[int, int], int],
    expert_id: int,
) -> int:
    """Return the number of deployed copies for one expert."""
    return sum(
        deployed
        for (e, _u), deployed in deployment.items()
        if e == expert_id
    )


def run_random_placement(
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Random feasible placement followed by the shared scheduler."""
    rng = np.random.default_rng(cfg.seed + 1009)
    (
        uav_ids,
        required_experts,
        substitutable_sets,
        candidates,
        similarity,
        _,
    ) = _prepare_similarity_context(uavs, experts, tasks, cfg)

    W_map = {e.id: e.W_e for e in experts}
    remaining_memory = {u.id: u.M_u for u in uavs}
    deployment = _empty_deployment(candidates, uav_ids)

    _place_required_originals(
        deployment,
        required_experts,
        uav_ids,
        W_map,
        remaining_memory,
        choose_uav=lambda _e, feasible: int(rng.choice(feasible)),
    )

    remaining_pairs = [
        (e, u)
        for e in candidates
        for u in uav_ids
        if deployment.get((e, u), 0) == 0
    ]
    rng.shuffle(remaining_pairs)
    for e, u in remaining_pairs:
        if _copy_count(deployment, e) >= cfg.max_copies_per_expert:
            continue
        if remaining_memory[u] < W_map[e]:
            continue
        deployment[(e, u)] = 1
        remaining_memory[u] -= W_map[e]

    return _schedule_with_deployment(
        "Random Placement",
        deployment,
        substitutable_sets,
        similarity,
        uavs,
        experts,
        tasks,
        cfg,
    )


def run_importance_placement(
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Place experts by global effective demand, ignoring network scoring."""
    (
        uav_ids,
        required_experts,
        substitutable_sets,
        candidates,
        similarity,
        demand_eff,
    ) = _prepare_similarity_context(uavs, experts, tasks, cfg)

    W_map = {e.id: e.W_e for e in experts}
    remaining_memory = {u.id: u.M_u for u in uavs}
    deployment = _empty_deployment(candidates, uav_ids)

    C_map = {u.id: u.C_u for u in uavs}

    def strongest_compute_uav(_e: int, feasible_uavs: List[int]) -> int:
        return max(feasible_uavs, key=lambda u: C_map.get(u, 0.0))

    _place_required_originals(
        deployment,
        required_experts,
        uav_ids,
        W_map,
        remaining_memory,
        choose_uav=strongest_compute_uav,
    )

    ranked_pairs = sorted(
        [
            (e, u)
            for e in candidates
            for u in uav_ids
            if deployment.get((e, u), 0) == 0
        ],
        key=lambda pair: (
            demand_eff.get(pair[0], {}).get(pair[1], 0.0),
            sum(demand_eff.get(pair[0], {}).values()),
            C_map.get(pair[1], 0.0),
            -W_map.get(pair[0], 0.0),
        ),
        reverse=True,
    )
    for e, u in ranked_pairs:
        if _copy_count(deployment, e) >= cfg.max_copies_per_expert:
            continue
        if remaining_memory[u] < W_map[e]:
            continue
        deployment[(e, u)] = 1
        remaining_memory[u] -= W_map[e]

    return _schedule_with_deployment(
        "Importance-based Placement",
        deployment,
        substitutable_sets,
        similarity,
        uavs,
        experts,
        tasks,
        cfg,
    )


def run_no_similarity_placement(
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Run Proposed placement/scheduling with similarity disabled."""
    cfg_no_sim = deepcopy(cfg)
    cfg_no_sim.similarity.xi = 1.0
    return run_pipeline(uavs, experts, tasks, cfg_no_sim)
