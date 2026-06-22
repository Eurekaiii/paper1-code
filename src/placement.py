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
    substitutable_sets: Dict[int, Set[int]] | None = None,
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
        demand_eff[e][u] = float. Used as a fallback in pre-deployment if
        a required expert's precomputed BaseScore is unavailable.
    substitutable_sets : dict | None
        substitutable_sets[r] gives experts that can cover required expert r.

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
    # Phase 0: pre-deploy coverage for required experts. A required expert
    # can be covered by itself or by one of its substitutable lightweight
    # experts, which avoids filling memory with only large original experts.
    # ================================================================
    if required_experts:
        uncovered_required = set(required_experts)

        def is_covered(required: int) -> bool:
            cover_set = substitutable_sets.get(required, {required}) if substitutable_sets else {required}
            cover_set = cover_set | {required}
            return any(
                deployed == 1 and e in cover_set
                for (e, _u), deployed in deployment.items()
            )

        while True:
            uncovered_required = {
                r for r in uncovered_required if not is_covered(r)
            }
            if not uncovered_required:
                break

            best_required = -1
            best_pair: Tuple[int, int] = (-1, -1)
            best_value = float("-inf")

            for required in sorted(uncovered_required):
                cover_set = (
                    substitutable_sets.get(required, {required})
                    if substitutable_sets
                    else {required}
                )
                cover_set = (cover_set | {required}) & candidates
                for e in sorted(cover_set):
                    if expert_copies.get(e, 0) >= max_copies_per_expert:
                        continue
                    W_e = W_map.get(e, float("inf"))
                    if W_e <= 0:
                        continue
                    for u in uav_ids:
                        if deployment.get((e, u), 0) == 1:
                            continue
                        if remaining_memory[u] < W_e:
                            continue
                        score = base_scores.get((e, u))
                        if score is None:
                            local = demand_eff.get(e, {}).get(u, 0.0) if demand_eff else 0.0
                            remote = 0.0
                            if demand_eff:
                                for v in uav_ids:
                                    if v == u:
                                        continue
                                    if uav_indices[v] < len(G) and uav_indices[u] < len(G):
                                        if G[uav_indices[v], uav_indices[u]] == 1:
                                            remote += demand_eff.get(e, {}).get(v, 0.0)
                            score = local + cfg.eta * remote
                        value = score / W_e
                        if value > best_value:
                            best_value = value
                            best_required = required
                            best_pair = (e, u)

            if best_required < 0:
                break

            best_e, best_u = best_pair
            deployment[(best_e, best_u)] = 1
            remaining_memory[best_u] -= W_map.get(best_e, 0.0)
            expert_copies[best_e] = expert_copies.get(best_e, 0) + 1
            uncovered_required.discard(best_required)
            # If another required expert is also covered by this substitute,
            # the next loop iteration will remove it through is_covered().

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
