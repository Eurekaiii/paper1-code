"""
AeroMoDE Communication-Aware Expert Placement Scoring
=======================================================
Compute A(e,u), B(e,u), C(e,u) → BaseScore(e,u) → Score_t(e,u) → Value_t(e,u).

Reference: aeromde_main.tex  Section IV-D  (placement scoring),
                              Section IV-E  (redundancy & dynamic score).
"""

from typing import Dict, List, Tuple, Set
import numpy as np

from .config import DeploymentConfig


# ======================================================================
#  A(e, u) — Communication-aware task demand benefit  (Eq. 13)
# ======================================================================
def compute_A(
    e: int,
    u: int,
    demand_eff: Dict[int, Dict[int, float]],
    G: np.ndarray,              # connectivity  (U×U)
    R: np.ndarray,               # rates         (U×U)
    R_max: float,
    eta: float,
    uav_indices: Dict[int, int],  # uav_id → matrix column index
) -> float:
    """A(e,u) = Demand^{eff}(e,u) + η·Σ_{v≠u} Demand^{eff}(e,v)·g_{v,u}·(R_{v,u}/R_max)."""
    local = demand_eff.get(e, {}).get(u, 0.0)

    remote = 0.0
    for v_id, v_idx in uav_indices.items():
        if v_id == u:
            continue
        if R_max <= 0:
            continue
        g_vu = G[v_idx, uav_indices[u]]
        if g_vu == 0:
            continue
        d_eff = demand_eff.get(e, {}).get(v_id, 0.0)
        r_vu = R[v_idx, uav_indices[u]]
        remote += d_eff * float(g_vu) * (r_vu / R_max)

    return local + eta * remote


# ======================================================================
#  B(e, u) — Compute capability benefit  (Eq. 14)
# ======================================================================
def compute_B(
    C_u: float,
    F_e: float,
) -> float:
    """B(e,u) = C_u / F_e."""
    if F_e <= 0:
        return float("inf")
    return C_u / F_e


# ======================================================================
#  C(e, u) — Network position benefit  (Eq. 15)
# ======================================================================
def compute_C(
    e: int,
    u: int,
    demand_eff: Dict[int, Dict[int, float]],
    G: np.ndarray,
    uav_indices: Dict[int, int],
) -> float:
    """C(e,u) = I_e · Σ_{v≠u} g_{v,u}.

    where I_e = Σ_u Demand^{eff}(e,u)  (global importance).
    """
    # Global effective demand I_e
    I_e = sum(demand_eff.get(e, {}).values())

    # In-degree centrality: how many UAVs can reach u
    in_degree = 0
    u_idx = uav_indices[u]
    for v_id, v_idx in uav_indices.items():
        if v_id == u:
            continue
        in_degree += int(G[v_idx, u_idx])

    return I_e * in_degree


# ======================================================================
#  Min-max normalisation
# ======================================================================
def minmax_normalise(values: Dict[Tuple[int, int], float]) \
        -> Dict[Tuple[int, int], float]:
    """Normalise all values to [0, 1] via min-max.

    If all values are identical, returns 0.0 for everything.
    """
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    denom = vmax - vmin
    if denom < 1e-15:
        return {k: 0.0 for k in values}
    return {k: (v - vmin) / denom for k, v in values.items()}


# ======================================================================
#  BaseScore(e, u)  — Eq. (16-17)
# ======================================================================
def compute_base_scores(
    candidates: Set[int],
    uav_ids: List[int],
    demand_eff: Dict[int, Dict[int, float]],
    G: np.ndarray,
    R: np.ndarray,
    R_max: float,
    C_map: Dict[int, float],          # uav_id → C_u
    F_map: Dict[int, float],          # expert_id → F_e
    W_map: Dict[int, float],          # expert_id → W_e
    uav_indices: Dict[int, int],
    cfg: DeploymentConfig,
) -> Dict[Tuple[int, int], float]:
    """Compute BaseScore(e,u) for all feasible (e,u) pairs.

    Steps:
    1. Compute raw A, B, C, W for each pair.
    2. Min-max normalise each dimension independently.
    3. Weighted sum:  α·Â + β·B̂ + γ·Ĉ  −  μ·Ŵ.
    """
    raw_A: Dict[Tuple[int, int], float] = {}
    raw_B: Dict[Tuple[int, int], float] = {}
    raw_C: Dict[Tuple[int, int], float] = {}
    raw_W: Dict[Tuple[int, int], float] = {}

    for e in candidates:
        F_e = F_map.get(e, 1.0)
        W_e = W_map.get(e, 0.0)
        for u in uav_ids:
            key = (e, u)

            raw_A[key] = compute_A(e, u, demand_eff, G, R, R_max,
                                   cfg.eta, uav_indices)
            raw_B[key] = compute_B(C_map.get(u, 0.0), F_e)
            raw_C[key] = compute_C(e, u, demand_eff, G, uav_indices)
            raw_W[key] = W_e

    # Normalise
    norm_A = minmax_normalise(raw_A)
    norm_B = minmax_normalise(raw_B)
    norm_C = minmax_normalise(raw_C)
    norm_W = minmax_normalise(raw_W)

    # Weighted sum
    base: Dict[Tuple[int, int], float] = {}
    for key in raw_A:
        base[key] = (cfg.alpha * norm_A[key]
                     + cfg.beta  * norm_B[key]
                     + cfg.gamma * norm_C[key]
                     - cfg.mu    * norm_W[key])

    return base


# ======================================================================
#  Redundancy_t(e, u)  — Eq. (14)
# ======================================================================
def compute_redundancy(
    e: int,
    u: int,
    deployment: Dict[Tuple[int, int], int],   # (e', v) → m_{e',v}
    similarity: Dict[Tuple[int, int], float],
    strict_substitutes: Dict[int, Set[int]],    # Ā(·)
    G: np.ndarray,
    uav_indices: Dict[int, int],
    eta_red: float,
) -> float:
    """Redundancy_t(e,u) = Σ_{e'∈Ā(e), m_{e',u}=1} Sim(e,e')
                           + η_red · Σ_{v≠u} g_{v,u} · Σ_{e'∈Ā(e), m_{e',v}=1} Sim(e,e').

    First term: local redundancy (same UAV).
    Second term: neighbour redundancy (single-hop reachable UAVs), discounted by η_red.
    """
    A_bar_e = strict_substitutes.get(e, set())
    u_idx = uav_indices[u]

    # Local redundancy
    local_red = 0.0
    for ep in A_bar_e:
        if deployment.get((ep, u), 0) == 1:
            local_red += similarity.get((e, ep), 0.0)

    # Neighbour redundancy
    neigh_red = 0.0
    for v_id, v_idx in uav_indices.items():
        if v_id == u:
            continue
        if G[v_idx, u_idx] == 0:          # v must be able to reach u
            continue
        for ep in A_bar_e:
            if deployment.get((ep, v_id), 0) == 1:
                neigh_red += similarity.get((e, ep), 0.0)

    return local_red + eta_red * neigh_red


# ======================================================================
#  Score_t(e, u) & Value_t(e, u)  — Eqs. (15-16)
# ======================================================================
def compute_dynamic_score(
    base_score: float,
    redundancy: float,
    nu: float,
    W_e: float,
) -> Tuple[float, float]:
    """Score_t = BaseScore − ν·Redundancy_t.
       Value_t = Score_t / W_e.

    Returns (score, value).
    """
    score = base_score - nu * redundancy
    if W_e <= 0:
        return (score, float("-inf"))
    value = score / W_e
    return (score, value)
