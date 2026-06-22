"""
AeroMoDE — End-to-End Pipeline
================================
Complete run:  config → models → communication → demand → similarity
              → scoring → placement → scheduling → result.

Usage:
    python -m src.main
"""

from typing import List, Dict, Tuple
import numpy as np

from .config import SystemConfig, ChannelConfig, DeploymentConfig, \
    SchedulingConfig, SimilarityConfig
from .models import UAV, Task, Expert, SystemResult
from .communication import (
    build_connectivity_matrix,
    build_access_info,
    default_access_uav,
    access_delay,
)
from .demand import (
    compute_direct_demand,
    compute_required_experts,
    compute_effective_demand,
)
from .similarity import (
    compute_similarity_matrix,
    build_substitutable_sets,
    build_candidate_set,
    strict_substitute_set,
)
from .scoring import compute_base_scores, compute_redundancy
from .placement import greedy_placement, get_deployed_experts_by_uav
from .scheduling import schedule_all_tasks


# ======================================================================
#  Example scenario builder
# ======================================================================
def build_example_scenario(cfg: SystemConfig):
    """Construct a small but illustrative UAV + task + expert scenario.

    Scenario
    --------
    3 UAVs at different altitudes, forming a sparse topology.
    5 ground tasks with varying expert sequences.
    6 experts with weight vectors for similarity; E1≈E4, E2≈E5.
    """
    rng = np.random.default_rng(cfg.seed)

    # --- UAVs ---
    uavs = [
        UAV(id=0, position=np.array([0.0, 0.0, 80.0]),
            C_u=1e10, M_u=2e9),       # 2 GB memory, 10 GFLOP/s
        UAV(id=1, position=np.array([120.0, 0.0, 70.0]),
            C_u=5e9, M_u=1.5e9),      # 1.5 GB, 5 GFLOP/s
        UAV(id=2, position=np.array([60.0, 100.0, 90.0]),
            C_u=8e9, M_u=1.8e9),      # 1.8 GB, 8 GFLOP/s
    ]

    # --- Experts ---
    # dim = 64 for the "weight vector"; in practice this would be the
    # flattened FFN parameters.
    dim = 64
    base_vecs = {
        0: rng.normal(0, 1, dim),
        1: rng.normal(0, 1, dim),
        2: rng.normal(0, 1, dim),
        3: rng.normal(0, 1, dim),
        4: rng.normal(0, 1, dim),
        5: rng.normal(0, 1, dim),
    }

    # Make E4 similar to E1, E5 similar to E2
    base_vecs[3] = base_vecs[0] + 0.15 * rng.normal(0, 1, dim)   # E4 ≈ E1
    base_vecs[4] = base_vecs[1] + 0.15 * rng.normal(0, 1, dim)   # E5 ≈ E2

    # Normalise for cosine similarity
    for k in base_vecs:
        base_vecs[k] = base_vecs[k] / np.linalg.norm(base_vecs[k])

    experts = [
        Expert(id=0, W_e=300e6, F_e=5e8,  weight_vector=base_vecs[0]),  # 300 MB
        Expert(id=1, W_e=350e6, F_e=6e8,  weight_vector=base_vecs[1]),
        Expert(id=2, W_e=280e6, F_e=4.5e8, weight_vector=base_vecs[2]),
        Expert(id=3, W_e=320e6, F_e=5.2e8, weight_vector=base_vecs[3]),  # ≈ E1
        Expert(id=4, W_e=360e6, F_e=6.2e8, weight_vector=base_vecs[4]),  # ≈ E2
        Expert(id=5, W_e=400e6, F_e=7e8,   weight_vector=base_vecs[5]),
    ]

    # --- Tasks ---
    # Each task has a sequence of required experts E_k
    tasks = [
        Task(id=0,
             position=np.array([10.0, 10.0]),
             S_in=1e6, S_out=1e4,
             S_mid=[5e5, 5e5, 5e5],
             expert_sequence=[0, 1, 2],
             omega=1.0),
        Task(id=1,
             position=np.array([50.0, 20.0]),
             S_in=2e6, S_out=1e4,
             S_mid=[8e5, 8e5],
             expert_sequence=[1, 3],
             omega=0.8),
        Task(id=2,
             position=np.array([100.0, 10.0]),
             S_in=1.5e6, S_out=1e4,
             S_mid=[6e5, 6e5, 6e5],
             expert_sequence=[0, 4, 5],
             omega=1.2),
        Task(id=3,
             position=np.array([30.0, 80.0]),
             S_in=1e6, S_out=1e4,
             S_mid=[4e5, 4e5],
             expert_sequence=[2, 5],
             omega=0.9),
        Task(id=4,
             position=np.array([80.0, 70.0]),
             S_in=2.5e6, S_out=1e4,
             S_mid=[1e6, 1e6, 1e6, 1e6],
             expert_sequence=[0, 1, 2, 5],
             omega=1.5),
    ]

    return uavs, experts, tasks


# ======================================================================
#  Main pipeline
# ======================================================================
def run_pipeline(
    uavs: List[UAV],
    experts: List[Expert],
    tasks: List[Task],
    cfg: SystemConfig,
) -> SystemResult:
    """Execute the complete AeroMoDE pipeline."""

    # --- shorthand ---
    ch_cfg = cfg.channel
    dep_cfg = cfg.deployment
    sched_cfg = cfg.scheduling
    sim_cfg = cfg.similarity

    U = len(uavs)
    uav_ids = [u.id for u in uavs]
    uav_indices = {u.id: i for i, u in enumerate(uavs)}

    # ---- Phase 1: Communication pre-computation ----
    G, Rmat = build_connectivity_matrix(uavs, ch_cfg)

    # R_max — max rate among available UAV-UAV links
    R_vals = [Rmat[i, j] for i in range(U) for j in range(U)
              if i != j and G[i, j] == 1]
    R_max = max(R_vals) if R_vals else 1.0

    # ---- Phase 2: Task access & demand (Section IV-B) ----
    access_uavs, access_delays = build_access_info(tasks, uavs, ch_cfg)

    demand = compute_direct_demand(tasks, uavs, access_uavs)
    E_req = compute_required_experts(tasks)

    # ---- Phase 3: Similarity & candidates (Section IV-C) ----
    sim = compute_similarity_matrix(experts)
    A_sets = build_substitutable_sets(sim, E_req, sim_cfg.xi)
    E_cand = build_candidate_set(A_sets)

    demand_eff = compute_effective_demand(demand, A_sets, sim, uav_ids)

    # ---- Phase 4: Scoring (Section IV-D) ----
    C_map = {u.id: u.C_u for u in uavs}
    F_map = {e.id: e.F_e for e in experts}
    W_map = {e.id: e.W_e for e in experts}
    M_map = {u.id: u.M_u for u in uavs}

    base_scores = compute_base_scores(
        candidates=E_cand,
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

    # ---- Phase 5: Greedy placement (Section IV-E, Algorithm 1) ----
    # Build strict substitute sets Ā(e)
    strict_subs = {e: strict_substitute_set(e, A_sets) for e in E_cand}

    deployment = greedy_placement(
        candidates=E_cand,
        uav_ids=uav_ids,
        base_scores=base_scores,
        W_map=W_map,
        M_map=M_map,
        similarity=sim,
        strict_substitutes=strict_subs,
        G=G,
        uav_indices=uav_indices,
        cfg=dep_cfg,
        max_copies_per_expert=cfg.max_copies_per_expert,
        required_experts=E_req,            # guarantee all required experts placed
        demand_eff=demand_eff,             # place on UAVs with highest local demand
    )

    deployed_by_uav = get_deployed_experts_by_uav(deployment)

    # ---- Phase 6: Inference scheduling (Section IV-E) ----
    # Use a conservative return rate (last UAV → ground)
    return_rate = ch_cfg.B * np.log2(1.0 + 10.0)   # ~10 dB SNR for return

    plans, D_weighted = schedule_all_tasks(
        tasks=tasks,
        access_assignment=access_uavs,
        access_delays=access_delays,
        uavs=uavs,
        deployment=deployment,
        deployed_by_uav=deployed_by_uav,
        substitutable_sets=A_sets,
        similarity=sim,
        G=G,
        R=Rmat,
        uav_indices=uav_indices,
        F_map=F_map,
        C_map=C_map,
        return_rate=return_rate,
        cfg=sched_cfg,
        comm_cfg=ch_cfg,
    )

    return SystemResult(
        deployment=deployment,
        execution_plans=plans,
        D_total=sum(p.D_total for p in plans),
        D_weighted=D_weighted,
    )


# ======================================================================
#  Report / pretty-print
# ======================================================================
def print_report(result: SystemResult, uavs: List[UAV], experts: List[Expert]):
    """Print a human-readable summary of the system result."""

    W_map = {e.id: e.W_e for e in experts}
    M_map = {u.id: u.M_u for u in uavs}

    print("=" * 72)
    print("  AeroMoDE — Pipeline Result")
    print("=" * 72)

    # Deployment
    print("\n--- Expert Deployment ---")
    deployed = [(e, u) for (e, u), m in result.deployment.items() if m == 1]
    if not deployed:
        print("  (no experts deployed)")
    else:
        for e, u in sorted(deployed, key=lambda x: (x[1], x[0])):
            w = W_map.get(e, 0) / 1e6
            print(f"  Expert E{e}  →  UAV {u}   ({w:.0f} MB)")

    # Memory usage per UAV
    print("\n--- UAV Memory Usage ---")
    for u in uavs:
        used = sum(W_map.get(e, 0) for (e, uu), m in result.deployment.items()
                   if m == 1 and uu == u.id)
        cap = M_map.get(u.id, 1)
        print(f"  UAV {u.id}:  {used/1e6:.0f} / {cap/1e6:.0f} MB  "
              f"({100*used/cap:.1f}%)")

    # Per-task execution
    print("\n--- Task Execution Plans ---")
    for plan in result.execution_plans:
        print(f"\n  Task {plan.task_id}  (access UAV {plan.access_uav})")
        for s in plan.steps:
            sub = "SUB" if s.is_substituted else "ORG"
            print(f"    Layer {s.layer_idx}:  "
                  f"want=E{s.original_expert}  →  use=E{s.actual_expert}  "
                  f"on UAV {s.uav_id}  [{sub}]")
            print(f"      trans={s.cost_transmission*1e3:.2f} ms  "
                  f"comp={s.cost_computation*1e3:.2f} ms  "
                  f"err={s.cost_error:.4f}")
        print(f"    TOTAL D_k = {plan.D_total*1e3:.2f} ms  "
              f"(acc={plan.D_access*1e3:.2f}  "
              f"cmp={plan.D_compute*1e3:.2f}  "
              f"tx={plan.D_trans*1e3:.2f}  "
              f"ret={plan.D_return*1e3:.2f})")

    # Overall
    print(f"\n--- Overall ---")
    print(f"  D_total (unweighted)  = {result.D_total*1e3:.2f} ms")
    print(f"  D_total (ω-weighted)  = {result.D_weighted*1e3:.2f} ms")
    print("=" * 72)


# ======================================================================
#  __main__
# ======================================================================
def main():
    cfg = SystemConfig()

    print("Building example scenario ...")
    uavs, experts, tasks = build_example_scenario(cfg)
    print(f"  {len(uavs)} UAVs, {len(experts)} experts, {len(tasks)} tasks")

    print("Running AeroMoDE pipeline ...")
    result = run_pipeline(uavs, experts, tasks, cfg)

    print_report(result, uavs, experts)


if __name__ == "__main__":
    main()
