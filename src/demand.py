"""
AeroMoDE Demand Modeling
==========================
Task access → direct expert demand → effective demand (with substitutes).

Reference: aeromde_main.tex  Section IV-B  (Demand modeling),
                              Section IV-C  (effective demand).
"""

from typing import List, Dict, Set, Tuple
import numpy as np

from .models import Task, UAV


# ---------------------------------------------------------------------------
# Direct demand  Demand(r, u)  — Eq. (8)：：求所有无人机u自身对用到的所有专家r的直接需求，（无周围无人机对专家的需求）
# ---------------------------------------------------------------------------
def compute_direct_demand(
    tasks: List[Task],
    uavs: List[UAV],
    access_assignment: List[int],
) -> Dict[int, Dict[int, float]]:
    """Demand(r, u) = Σ_{k: u_k^a = u}  ω_k · q_{k,r}.

    Parameters
    ----------
    tasks : List[Task]
    uavs : List[UAV]
    access_assignment : List[int]
        access_assignment[k] = u_k^a  (UAV id that task k accesses).

    Returns
    -------
    demand : dict   demand[r][u] = float
    """
    demand: Dict[int, Dict[int, float]] = {}

    for task, u_id in zip(tasks, access_assignment):
        omega = task.omega
        for r in set(task.expert_sequence):
            q_kr = task.get_demand_strength(r)
            if q_kr <= 0:
                continue
            demand.setdefault(r, {})
            demand[r][u_id] = demand[r].get(u_id, 0.0) + omega * q_kr

    # Fill zeros for UAVs with no demand
    for r in demand:
        for uav in uavs:
            demand[r].setdefault(uav.id, 0.0)

    return demand


# ---------------------------------------------------------------------------
# Required expert set  ℰ_req  — Eq. (10)
# ---------------------------------------------------------------------------
def compute_required_experts(tasks: List[Task]) -> Set[int]:
    """ℰ_req = { r | ∃ k, r ∈ E_k }."""
    req: Set[int] = set()
    for task in tasks:
        req.update(task.expert_sequence)
    return req


# ---------------------------------------------------------------------------
# Effective demand  Demand^{eff}(e, u)  — Eq. (11-12)求所有无人机u自身对用到的所有专家r（包含类似）的直接需求，（无周围无人机对专家的需求）
# ---------------------------------------------------------------------------
def compute_effective_demand(
    demand: Dict[int, Dict[int, float]],
    substitutable_sets: Dict[int, Set[int]],   # A(r) — experts that can replace r
    similarity: Dict[Tuple[int, int], float],   # Sim(r, e)
    uav_ids: List[int],
) -> Dict[int, Dict[int, float]]:
    """Demand^{eff}(e, u) = Σ_{r: e ∈ A(r)}  Demand(r, u) · θ(r, e).

    θ(r, e) = 1 if e == r else Sim(r, e).

    Parameters
    ----------
    demand : dict   demand[r][u] = float  (direct demand)
    substitutable_sets : dict   A[r] = {e1, e2, ...}
    similarity : dict   (r, e) → Sim(r, e)
    uav_ids : List[int]

    Returns
    -------
    demand_eff : dict   demand_eff[e][u] = float
    """
    demand_eff: Dict[int, Dict[int, float]] = {}

    for r, r_demand in demand.items():
        A_r = substitutable_sets.get(r, {r})
        for e in A_r:
            # Discount factor
            theta = 1.0 if e == r else similarity.get((r, e), 0.0)
            if theta <= 0:
                continue

            demand_eff.setdefault(e, {})
            for u_id in uav_ids:
                val = r_demand.get(u_id, 0.0) * theta
                demand_eff[e][u_id] = demand_eff[e].get(u_id, 0.0) + val

    # Fill zeros for all (e, u) pairs without demand
    for e in demand_eff:
        for u_id in uav_ids:
            demand_eff[e].setdefault(u_id, 0.0)

    return demand_eff
