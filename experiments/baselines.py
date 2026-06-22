"""
Baseline Algorithms
=====================
Comparison baselines for AeroMoDE.

Baselines:
  - LocalOnly:     No collaboration; each task runs on its access UAV only.
  - NoSubstitute:  AeroMoDE deployment but disables expert substitution at runtime
                   (only original experts used).
  - RandomPlace:   Random expert placement + normal scheduling.
  - NoRedundancy:  Greedy placement without dynamic redundancy (ν = 0).
  - RoundRobin:    Place experts in round-robin order, ignoring scores.
"""

from __future__ import annotations
import copy
import numpy as np
from typing import Dict, List, Tuple, Set

from src.config import SystemConfig
from src.models import UAV, Task, Expert, SystemResult
from src.main import run_pipeline
from src.placement import get_deployed_experts_by_uav
from src.similarity import strict_substitute_set


# ======================================================================
# Baseline 1: Local Only
# ======================================================================
def baseline_local_only(
    uavs: List[UAV], experts: List[Expert], tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Each task executes ALL its expert steps on its access UAV.

    No cross-UAV communication. If an expert is not deployed on the access
    UAV, the task fails (its delay is set to a large penalty).
    """
    # Force all experts to deploy on every UAV (idealised for comparison)
    # In practice, we deploy all experts that fit on each UAV
    cfg2 = copy.deepcopy(cfg)
    cfg2.max_copies_per_expert = len(uavs)  # allow copies on every UAV

    # Set memory large enough to hold all experts
    total_expert_size = sum(e.W_e for e in experts)
    uavs_big = []
    for u in uavs:
        uavs_big.append(UAV(id=u.id, position=u.position.copy(),
                            C_u=u.C_u, M_u=total_expert_size * 2))

    return run_pipeline(uavs_big, experts, tasks, cfg2)


# ======================================================================
# Baseline 2: No Expert Substitution
# ======================================================================
def baseline_no_substitution(
    uavs: List[UAV], experts: List[Expert], tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Disable similarity-based substitution: set ξ = 1.0 (only identical
    experts are substitutable, i.e. effectively no substitution)."""
    cfg2 = copy.deepcopy(cfg)
    cfg2.similarity.xi = 1.0
    return run_pipeline(uavs, experts, tasks, cfg2)


# ======================================================================
# Baseline 3: Random Placement
# ======================================================================
def baseline_random_placement(
    uavs: List[UAV], experts: List[Expert], tasks: List[Task],
    cfg: SystemConfig, rng: np.random.Generator = None,
) -> SystemResult:
    """Replace the greedy placement with random feasible placement."""
    if rng is None:
        rng = np.random.default_rng(42)

    uav_ids = [u.id for u in uavs]
    E_cand = set(e.id for e in experts)
    W_map = {e.id: e.W_e for e in experts}
    M_map = {u.id: u.M_u for u in uavs}

    # Identify required experts from tasks
    from src.demand import compute_required_experts
    E_req = compute_required_experts(tasks)

    remaining = {u: M_map[u] for u in uav_ids}

    deployment: Dict[Tuple[int, int], int] = {}
    for e in E_cand:
        for u in uav_ids:
            deployment[(e, u)] = 0

    # Phase 0: pre-deploy required experts first (guarantee feasibility)
    req_sorted = sorted(E_req, key=lambda e: W_map.get(e, 0.0), reverse=True)
    for e in req_sorted:
        if e not in E_cand:
            continue
        feasible_uavs = [u for u in uav_ids if remaining[u] >= W_map[e]]
        if feasible_uavs:
            u = rng.choice(feasible_uavs)
            deployment[(e, u)] = 1
            remaining[u] -= W_map[e]

    # Phase 1: random placement for remaining experts
    remaining_experts = [e for e in E_cand
                         if all(deployment[(e, u)] == 0 for u in uav_ids)]
    rng.shuffle(remaining_experts)

    for e in remaining_experts:
        W_e = W_map[e]
        feasible_uavs = [u for u in uav_ids if remaining[u] >= W_e]
        if not feasible_uavs:
            continue
        u = rng.choice(feasible_uavs)
        deployment[(e, u)] = 1
        remaining[u] -= W_e

    deployed_by_uav = get_deployed_experts_by_uav(deployment)

    # Re-run the scheduling part with this random deployment
    from src.communication import build_connectivity_matrix, build_access_info
    from src.demand import compute_direct_demand, compute_required_experts, \
        compute_effective_demand
    from src.similarity import compute_similarity_matrix, build_substitutable_sets, \
        build_candidate_set
    from src.scheduling import schedule_all_tasks

    G, Rmat = build_connectivity_matrix(uavs, cfg.channel)
    R_vals = [Rmat[i, j] for i in range(len(uavs)) for j in range(len(uavs))
              if i != j and G[i, j] == 1]
    R_max = max(R_vals) if R_vals else 1.0

    access_uavs, access_delays = build_access_info(tasks, uavs, cfg.channel)
    demand = compute_direct_demand(tasks, uavs, access_uavs)
    E_req = compute_required_experts(tasks)
    sim = compute_similarity_matrix(experts)
    A_sets = build_substitutable_sets(sim, E_req, cfg.similarity.xi)
    demand_eff = compute_effective_demand(demand, A_sets, sim, uav_ids)

    uav_indices = {u.id: i for i, u in enumerate(uavs)}
    C_map = {u.id: u.C_u for u in uavs}
    F_map = {e.id: e.F_e for e in experts}

    return_rate = cfg.channel.B * np.log2(1.0 + 10.0)

    plans, D_weighted = schedule_all_tasks(
        tasks=tasks, access_assignment=access_uavs,
        access_delays=access_delays, uavs=uavs,
        deployment=deployment, deployed_by_uav=deployed_by_uav,
        substitutable_sets=A_sets, similarity=sim,
        G=G, R=Rmat, uav_indices=uav_indices,
        F_map=F_map, C_map=C_map,
        return_rate=return_rate, cfg=cfg.scheduling,
        comm_cfg=cfg.channel,
    )

    return SystemResult(
        deployment=deployment, execution_plans=plans,
        D_total=sum(p.D_total for p in plans), D_weighted=D_weighted,
    )


# ======================================================================
# Baseline 4: No Redundancy (ν = 0)
# ======================================================================
def baseline_no_redundancy(
    uavs: List[UAV], experts: List[Expert], tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Greedy placement without dynamic redundancy penalty."""
    cfg2 = copy.deepcopy(cfg)
    cfg2.deployment.nu = 0.0
    return run_pipeline(uavs, experts, tasks, cfg2)


# ======================================================================
# Baseline 5: Round-Robin Placement
# ======================================================================
def baseline_round_robin(
    uavs: List[UAV], experts: List[Expert], tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Round-robin expert distribution: expert e goes to UAV (e mod U)."""
    uav_ids = [u.id for u in uavs]
    E_cand = set(e.id for e in experts)
    W_map = {e.id: e.W_e for e in experts}
    M_map = {u.id: u.M_u for u in uavs}

    # Identify required experts
    from src.demand import compute_required_experts
    E_req = compute_required_experts(tasks)

    deployment: Dict[Tuple[int, int], float] = {}
    for e in E_cand:
        for u in uav_ids:
            deployment[(e, u)] = 0

    remaining = {u: M_map[u] for u in uav_ids}

    # Phase 0: pre-deploy required experts with round-robin
    req_sorted = sorted(E_req, key=lambda e: W_map.get(e, 0.0), reverse=True)
    for e in req_sorted:
        if e not in E_cand:
            continue
        u_target = uav_ids[e % len(uav_ids)]
        W_e = W_map[e]
        if remaining[u_target] >= W_e:
            deployment[(e, u_target)] = 1
            remaining[u_target] -= W_e
        else:
            for u in uav_ids:
                if remaining[u] >= W_e:
                    deployment[(e, u)] = 1
                    remaining[u] -= W_e
                    break

    # Phase 1: round-robin remaining experts
    remaining_experts = sorted(
        [e for e in E_cand if all(deployment[(e, u)] == 0 for u in uav_ids)],
        key=lambda e: W_map[e], reverse=True,
    )

    for e in remaining_experts:
        u_target = uav_ids[e % len(uav_ids)]
        W_e = W_map[e]
        if remaining[u_target] >= W_e:
            deployment[(e, u_target)] = 1
            remaining[u_target] -= W_e
        else:
            # Try other UAVs
            for u in uav_ids:
                if remaining[u] >= W_e:
                    deployment[(e, u)] = 1
                    remaining[u] -= W_e
                    break

    deployed_by_uav = get_deployed_experts_by_uav(deployment)

    # Scheduling
    from src.communication import build_connectivity_matrix, build_access_info
    from src.demand import compute_direct_demand, compute_required_experts, \
        compute_effective_demand
    from src.similarity import compute_similarity_matrix, build_substitutable_sets
    from src.scheduling import schedule_all_tasks

    G, Rmat = build_connectivity_matrix(uavs, cfg.channel)
    access_uavs, access_delays = build_access_info(tasks, uavs, cfg.channel)
    demand = compute_direct_demand(tasks, uavs, access_uavs)
    E_req = compute_required_experts(tasks)
    sim = compute_similarity_matrix(experts)
    A_sets = build_substitutable_sets(sim, E_req, cfg.similarity.xi)
    demand_eff = compute_effective_demand(demand, A_sets, sim, uav_ids)

    uav_indices = {u.id: i for i, u in enumerate(uavs)}
    C_map = {u.id: u.C_u for u in uavs}
    F_map = {e.id: e.F_e for e in experts}
    return_rate = cfg.channel.B * np.log2(1.0 + 10.0)

    plans, D_weighted = schedule_all_tasks(
        tasks=tasks, access_assignment=access_uavs,
        access_delays=access_delays, uavs=uavs,
        deployment=deployment, deployed_by_uav=deployed_by_uav,
        substitutable_sets=A_sets, similarity=sim,
        G=G, R=Rmat, uav_indices=uav_indices,
        F_map=F_map, C_map=C_map,
        return_rate=return_rate, cfg=cfg.scheduling,
        comm_cfg=cfg.channel,
    )

    return SystemResult(
        deployment=deployment, execution_plans=plans,
        D_total=sum(p.D_total for p in plans), D_weighted=D_weighted,
    )


# ======================================================================
# Baseline registry
# ======================================================================
BASELINES = {
    "AeroMoDE":           run_pipeline,
    "NoSubstitute":       baseline_no_substitution,
    "NoRedundancy":       baseline_no_redundancy,
    "RandomPlacement":    baseline_random_placement,
    "RoundRobin":         baseline_round_robin,
}
