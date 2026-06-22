"""
AeroMoDE Lightweight Inference Scheduling
===========================================
Per-layer greedy scheduling: for each expert step, pick the (ê, u) that
minimises  transmission + computation + substitution error.

Reference: aeromde_main.tex  Section IV-E  (Eqs. 18–27).
"""

from typing import Dict, List, Tuple, Set, Optional
import numpy as np

from .config import SchedulingConfig
from .models import Task, UAV, ExpertStep, TaskExecutionPlan
from .communication import single_hop_delay


def schedule_task(
    task: Task,
    access_uav: int,
    uavs: List[UAV],
    uav_map: Dict[int, UAV],
    deployment: Dict[Tuple[int, int], int],
    deployed_by_uav: Dict[int, List[int]],
    substitutable_sets: Dict[int, Set[int]],        # A(r)
    similarity: Dict[Tuple[int, int], float],
    G: np.ndarray,
    R: np.ndarray,
    uav_indices: Dict[int, int],
    F_map: Dict[int, float],
    C_map: Dict[int, float],
    access_delay_val: float,
    return_rate: float,                              # R^{down} for return
    cfg: SchedulingConfig,
    comm_cfg,   # ChannelConfig — for single_hop_delay calls
) -> TaskExecutionPlan:
    """Schedule one task's expert execution path layer-by-layer.

    Algorithm sketch (Eqs. 18–27):
      cur ← u_k^a
      for l = 1 .. L_k:
        build C_{k,l} = {(ê, u) | ê ∈ A(e_{k,l}), m_{ê,u}=1, u==cur or g_{cur,u}=1}
        for each (ê, u):
          Cost = T^{single}(S_{k,l-1}^{mid}) + F_ê/C_u + λ·Err(e_{k,l}, ê)
        pick min-cost
        update cur ← u*
    """
    # Map uav_id → UAV object for delay lookups
    uav_objects: Dict[int, UAV] = {u.id: u for u in uavs}

    cur = access_uav
    steps: List[ExpertStep] = []

    for l_idx, e_orig in enumerate(task.expert_sequence):
        # Determine intermediate data size to transmit
        # S_{k,0}^{mid} = S_k^{in} (input), S_{k,l}^{mid} from task.S_mid
        if l_idx == 0:
            S_prev = task.S_in
        else:
            # S_mid[l_idx-1] is the output size after step l_idx
            # (after step l_idx-1 completes, the feature has size S_mid[l_idx-1])
            S_prev = task.S_mid[l_idx - 1] if l_idx - 1 < len(task.S_mid) else task.S_mid[-1]

        # Build candidate execution set C_{k,l} (Eq. 19)
        A_e = substitutable_sets.get(e_orig, {e_orig})
        candidates: List[Tuple[int, int]] = []  # (ê, u)
        fallback_candidates: List[Tuple[int, int]] = []  # all deployed, ignoring reachability

        for e_hat in A_e:
            for u_id, deployed_list in deployed_by_uav.items():
                if e_hat not in deployed_list:
                    continue
                fallback_candidates.append((e_hat, u_id))  # always available as fallback
                if u_id == cur:
                    candidates.append((e_hat, u_id))
                else:
                    cur_idx = uav_indices[cur]
                    u_idx = uav_indices[u_id]
                    if G[cur_idx, u_idx] == 1:
                        candidates.append((e_hat, u_id))

        # If no single-hop-reachable candidate, fall back to all deployed experts
        # (with a large transmission penalty for unreachable links)
        use_fallback = False
        if not candidates:
            if not fallback_candidates:
                raise RuntimeError(
                    f"Task {task.id} layer {l_idx}: no UAV has a substitute "
                    f"for expert {e_orig} deployed at all."
                )
            candidates = fallback_candidates
            use_fallback = True

        # Evaluate cost for each candidate (Eq. 20)
        best_cost = float("inf")
        best_e_hat = -1
        best_u = -1
        best_trans = 0.0
        best_comp = 0.0
        best_err = 0.0

        for e_hat, u_id in candidates:
            cur_uav = uav_objects[cur]
            tgt_uav = uav_objects[u_id]

            # Transmission delay
            if u_id == cur:
                T_trans = 0.0
            else:
                T_trans = single_hop_delay(cur_uav, tgt_uav, S_prev, comm_cfg)
                # If not single-hop reachable (in fallback), use a large penalty
                # proportional to data size / minimum rate
                if np.isinf(T_trans) and use_fallback:
                    # Penalty: assume multi-hop or retransmission cost
                    # Use 5x the worst single-hop delay as penalty
                    worst_rate = 1e6  # 1 Mbps fallback
                    T_trans = S_prev / worst_rate * 5.0

            if np.isinf(T_trans):
                continue

            # Computation delay
            F_e = F_map.get(e_hat, 1.0)
            C_uu = C_map.get(u_id, 1.0)
            T_comp = F_e / C_uu if C_uu > 0 else float("inf")

            # Substitution error (Eq. 21)
            if e_hat == e_orig:
                err = 0.0
            else:
                err = 1.0 - similarity.get((e_orig, e_hat), 0.0)

            cost = T_trans + T_comp + cfg.lamb * err

            if cost < best_cost:
                best_cost = cost
                best_e_hat = e_hat
                best_u = u_id
                best_trans = T_trans
                best_comp = T_comp
                best_err = cfg.lamb * err

        if best_e_hat < 0:
            raise RuntimeError(
                f"Task {task.id} layer {l_idx}: no feasible candidate "
                f"(all have infinite cost)."
            )

        steps.append(ExpertStep(
            layer_idx=l_idx,
            original_expert=e_orig,
            actual_expert=best_e_hat,
            uav_id=best_u,
            is_substituted=(best_e_hat != e_orig),
            cost_transmission=best_trans,
            cost_computation=best_comp,
            cost_error=best_err,
        ))

        # Update current UAV
        cur = best_u

    # --- Aggregate delays ---
    D_access = access_delay_val
    D_compute = sum(s.cost_computation for s in steps)
    D_trans = sum(s.cost_transmission for s in steps)

    # Return delay: last UAV → ground  (Eq. 5)
    if return_rate > 0:
        D_return = task.S_out / return_rate
    else:
        D_return = 0.0

    D_total = D_access + D_compute + D_trans + D_return

    return TaskExecutionPlan(
        task_id=task.id,
        access_uav=access_uav,
        steps=steps,
        D_access=D_access,
        D_compute=D_compute,
        D_trans=D_trans,
        D_return=D_return,
        D_total=D_total,
    )


def schedule_all_tasks(
    tasks: List[Task],
    access_assignment: List[int],
    access_delays: np.ndarray,            # (K, U)
    uavs: List[UAV],
    deployment: Dict[Tuple[int, int], int],
    deployed_by_uav: Dict[int, List[int]],
    substitutable_sets: Dict[int, Set[int]],
    similarity: Dict[Tuple[int, int], float],
    G: np.ndarray,
    R: np.ndarray,
    uav_indices: Dict[int, int],
    F_map: Dict[int, float],
    C_map: Dict[int, float],
    return_rate: float,
    cfg: SchedulingConfig,
    comm_cfg,
) -> Tuple[List[TaskExecutionPlan], float]:
    """Schedule all tasks and compute weighted total delay D_total.

    Returns (plans, D_weighted).
    """
    uav_map = {u.id: u for u in uavs}
    plans: List[TaskExecutionPlan] = []

    D_weighted = 0.0

    for ki, task in enumerate(tasks):
        u_acc = access_assignment[ki]
        acc_dly = access_delays[ki, uav_indices[u_acc]]

        plan = schedule_task(
            task=task,
            access_uav=u_acc,
            uavs=uavs,
            uav_map=uav_map,
            deployment=deployment,
            deployed_by_uav=deployed_by_uav,
            substitutable_sets=substitutable_sets,
            similarity=similarity,
            G=G,
            R=R,
            uav_indices=uav_indices,
            F_map=F_map,
            C_map=C_map,
            access_delay_val=acc_dly,
            return_rate=return_rate,
            cfg=cfg,
            comm_cfg=comm_cfg,
        )
        plans.append(plan)
        D_weighted += task.omega * plan.D_total

    return plans, D_weighted
