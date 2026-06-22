"""
AeroMoDE Dynamic Greedy Expert Placement
==========================================
Algorithm 1 from the paper: iteratively selects the (expert, UAV) pair
with the highest unit-memory value under dynamic redundancy penalties.

Reference: aeromde_main.tex  Section IV-E, Algorithm 1.
"""

from typing import Dict, List, Tuple, Set
import numpy as np

from .config import DeploymentConfig
from .scoring import compute_redundancy, compute_dynamic_score


def _filter_rankings_by_memory(
    expert_rankings: Dict[int, List[Tuple[int, float]]],
    remaining_memory: Dict[int, float],
    W_map: Dict[int, float],
) -> None:
    """Drop required-expert rankings that no longer fit current memory."""
    for e in list(expert_rankings.keys()):
        expert_rankings[e] = [
            (u, score)
            for u, score in expert_rankings[e]
            if remaining_memory[u] >= W_map.get(e, float("inf"))
        ]
        if not expert_rankings[e]:
            del expert_rankings[e]


def greedy_placement(
    candidates: Set[int],
    uav_ids: List[int],
    base_scores: Dict[Tuple[int, int], float],
    W_map: Dict[int, float],                        # expert → W_e
    M_map: Dict[int, float],                         # uav → M_u
    similarity: Dict[Tuple[int, int], float],
    strict_substitutes: Dict[int, Set[int]],          # Ā(e)
    G: np.ndarray,
    uav_indices: Dict[int, int],
    cfg: DeploymentConfig,
    max_copies_per_expert: int = 1,
    required_experts: Set[int] | None = None,
    demand_eff: Dict[int, Dict[int, float]] | None = None,
) -> Dict[Tuple[int, int], int]:
    """Dynamic greedy expert placement (Algorithm 1).

    Parameters
    ----------
    candidates : set   ℰ_cand
    uav_ids : list     UAV IDs
    base_scores : dict   (e, u) → BaseScore  (static, pre-computed)
    W_map : dict   expert → memory size
    M_map : dict   uav → memory capacity
    similarity : dict   (a, b) → Sim(a, b)
    strict_substitutes : dict   e → Ā(e)
    G : np.ndarray   connectivity matrix (U×U)
    uav_indices : dict   uav_id → matrix index
    cfg : DeploymentConfig
    max_copies_per_expert : int
    required_experts : set | None
        Experts that MUST be deployed somewhere (feasibility guarantee).
    demand_eff : dict | None
        demand_eff[e][u] = float. Used in pre-deployment to place required
        experts on the UAV with highest LOCAL demand (avoiding cross-UAV hops).

    Returns
    -------
    deployment : dict   (e, u) → m_{e,u} ∈ {0, 1}
    """
    deployment: Dict[Tuple[int, int], int] = {}
    for e in candidates:
        for u in uav_ids:
            deployment[(e, u)] = 0

    remaining_memory = {u: M_map.get(u, 0.0) for u in uav_ids}
    expert_copies = {e: 0 for e in candidates}

    # ================================================================
    # Phase 0: pre-deploy required experts (feasibility guarantee).
    # "Regret-based" assignment: for each required expert, compute the
    # gap between its best and second-best UAV by A(e,u).  Allocate
    # experts with the LARGEST regret first — they have the strongest
    # preference and suffer most if displaced.  This avoids the bin-
    # packing pathology where later experts get pushed to unreachable UAVs.
    # ================================================================
    if required_experts:
        # Build per-expert UAV rankings by A(e,u)
        expert_rankings: Dict[int, List[Tuple[int, float]]] = {}
        for e in required_experts:
            if e not in candidates:
                continue
            rankings: List[Tuple[int, float]] = []
            for u in uav_ids:
                if remaining_memory[u] < W_map.get(e, float("inf")):
                    continue
                local = demand_eff.get(e, {}).get(u, 0.0) if demand_eff else 0.0
                remote = 0.0
                if demand_eff:
                    for v in uav_ids:
                        if v == u:
                            continue
                        if uav_indices[v] < len(G) and uav_indices[u] < len(G):
                            if G[uav_indices[v], uav_indices[u]] == 1:
                                remote += demand_eff.get(e, {}).get(v, 0.0)
                A_val = local + cfg.eta * remote
                rankings.append((u, A_val))
            rankings.sort(key=lambda x: x[1], reverse=True)
            if rankings:
                expert_rankings[e] = rankings

        # Assign by descending regret (gap between best and second-best)
        while expert_rankings:
            _filter_rankings_by_memory(expert_rankings, remaining_memory, W_map)
            if not expert_rankings:
                break

            # Find expert with largest regret
            best_e = -1
            best_regret = -1.0
            for e, rankings in expert_rankings.items():
                if len(rankings) >= 2:
                    regret = rankings[0][1] - rankings[1][1]
                elif len(rankings) == 1:
                    regret = float("inf")  # only one viable UAV — must go there
                else:
                    regret = -1.0
                if regret > best_regret:
                    best_regret = regret
                    best_e = e

            if best_e < 0:
                break

            rankings = expert_rankings.pop(best_e)
            best_u, _ = rankings[0]
            if remaining_memory[best_u] < W_map.get(best_e, float("inf")):
                continue
            deployment[(best_e, best_u)] = 1
            remaining_memory[best_u] -= W_map.get(best_e, 0.0)
            expert_copies[best_e] = 1

            # Remove this UAV from other experts' rankings if memory exhausted
            if remaining_memory[best_u] < min(W_map.values()):
                _filter_rankings_by_memory(expert_rankings, remaining_memory, W_map)
            # If no UAV can fit this required expert, we proceed anyway;
            # the scheduling phase will report the infeasibility clearly.

    # ================================================================
    # Phase 1: main greedy loop (Algorithm 1)
    # ================================================================
    while True:
        # --- build feasible set  F_t ---
        feasible: List[Tuple[int, int]] = []
        for e in candidates:
            if expert_copies[e] >= max_copies_per_expert:
                continue
            W_e = W_map.get(e, float("inf"))
            for u in uav_ids:
                if deployment[(e, u)] == 1:
                    continue
                if remaining_memory[u] >= W_e:
                    feasible.append((e, u))

        if not feasible:
            break

        # --- compute Value_t for all feasible ---
        best_value = float("-inf")
        best_pair: Tuple[int, int] = (-1, -1)

        for e, u in feasible:
            W_e = W_map.get(e, 1.0)
            red = compute_redundancy(e, u, deployment, similarity,
                                     strict_substitutes, G, uav_indices,
                                     cfg.eta_red)
            base = base_scores.get((e, u), 0.0)
            _, value = compute_dynamic_score(base, red, cfg.nu, W_e)

            if value > best_value:
                best_value = value
                best_pair = (e, u)

        # --- deploy best ---
        e_star, u_star = best_pair
        if e_star < 0:
            break

        deployment[(e_star, u_star)] = 1
        remaining_memory[u_star] -= W_map.get(e_star, 0.0)
        expert_copies[e_star] += 1

    return deployment


def get_deployed_experts_by_uav(
    deployment: Dict[Tuple[int, int], int],
) -> Dict[int, List[int]]:
    """Group deployed experts by UAV for easier lookup during scheduling.

    Returns dict  uav_id → [expert_id, ...].
    """
    by_uav: Dict[int, List[int]] = {}
    for (e, u), m in deployment.items():
        if m == 1:
            by_uav.setdefault(u, []).append(e)
    return by_uav
