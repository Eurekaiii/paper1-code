"""
AeroMoDE lightweight inference scheduling.

For each task layer, the scheduler chooses a deployed original/substitute
expert on the current UAV or on a single-hop reachable UAV. Multi-hop and
unreachable transmission are not allowed.
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from .communication import downlink_delay, downlink_rate, single_hop_delay
from .config import SchedulingConfig
from .models import ExpertStep, Task, TaskExecutionPlan, UAV


def _build_compute_budgets(
    C_map: Dict[int, float],
    cfg: SchedulingConfig,
) -> Dict[int, float]:
    """Return per-UAV FLOP budgets for the current scheduling window."""
    if not cfg.enforce_compute_capacity:
        return {u: float("inf") for u in C_map}
    window_s = max(cfg.compute_window_s, 0.0)
    return {u: max(C_u, 0.0) * window_s for u, C_u in C_map.items()}


def schedule_task(
    task: Task,
    access_uav: int,
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
    remaining_compute: Dict[int, float],
    access_delay_val: float,
    cfg: SchedulingConfig,
    comm_cfg,
) -> TaskExecutionPlan:
    """Schedule one task layer-by-layer under strict single-hop reachability."""
    uav_objects: Dict[int, UAV] = {u.id: u for u in uavs}
    cur = access_uav
    steps: List[ExpertStep] = []

    for layer_idx, original_expert in enumerate(task.expert_sequence):
        is_final_layer = layer_idx == task.L - 1
        if layer_idx == 0:
            data_size = task.S_in
        else:
            prev_idx = layer_idx - 1
            data_size = task.S_mid[prev_idx] if prev_idx < len(task.S_mid) else task.S_mid[-1]

        substitutes = substitutable_sets.get(original_expert, {original_expert})
        deployed_substitutes: List[Tuple[int, int]] = []
        candidates: List[Tuple[int, int]] = []

        for actual_expert in substitutes:
            for uav_id, deployed_experts in deployed_by_uav.items():
                if actual_expert not in deployed_experts:
                    continue
                deployed_substitutes.append((actual_expert, uav_id))
                if uav_id == cur or G[uav_indices[cur], uav_indices[uav_id]] == 1:
                    if is_final_layer and downlink_rate(
                        uav_objects[uav_id], task.position, comm_cfg
                    ) <= 0.0:
                        continue
                    candidates.append((actual_expert, uav_id))

        if not candidates:
            if deployed_substitutes:
                raise RuntimeError(
                    f"Task {task.id} layer {layer_idx}: expert {original_expert} "
                    f"or its substitutes are deployed, but none are reachable "
                    f"from UAV {cur} by a single hop."
                )
            raise RuntimeError(
                f"Task {task.id} layer {layer_idx}: no deployed expert can serve "
                f"expert {original_expert}."
            )

        best_cost = float("inf")
        best_expert = -1
        best_uav = -1
        best_trans = 0.0
        best_comp = 0.0
        best_err = 0.0
        capacity_feasible = False

        for actual_expert, uav_id in candidates:
            F_e = F_map.get(actual_expert, 1.0)
            if remaining_compute.get(uav_id, 0.0) < F_e:
                continue
            capacity_feasible = True

            if uav_id == cur:
                trans_delay = 0.0
            else:
                trans_delay = single_hop_delay(
                    uav_objects[cur],
                    uav_objects[uav_id],
                    data_size,
                    comm_cfg,
                )

            if np.isinf(trans_delay):
                continue

            C_u = C_map.get(uav_id, 1.0)
            comp_delay = F_e / C_u if C_u > 0 else float("inf")

            if actual_expert == original_expert:
                err = 0.0
            else:
                err = 1.0 - similarity.get((original_expert, actual_expert), 0.0)
            err_penalty = cfg.lamb * err

            cost = trans_delay + comp_delay + err_penalty
            if cost < best_cost:
                best_cost = cost
                best_expert = actual_expert
                best_uav = uav_id
                best_trans = trans_delay
                best_comp = comp_delay
                best_err = err_penalty

        if best_expert < 0:
            if not capacity_feasible:
                raise RuntimeError(
                    f"Task {task.id} layer {layer_idx}: reachable expert "
                    f"candidates exist, but their UAV compute budgets are exhausted."
                )
            raise RuntimeError(
                f"Task {task.id} layer {layer_idx}: all single-hop candidates "
                f"have infinite cost."
            )

        steps.append(
            ExpertStep(
                layer_idx=layer_idx,
                original_expert=original_expert,
                actual_expert=best_expert,
                uav_id=best_uav,
                is_substituted=(best_expert != original_expert),
                cost_transmission=best_trans,
                cost_computation=best_comp,
                cost_error=best_err,
            )
        )
        remaining_compute[best_uav] -= F_map.get(best_expert, 0.0)
        cur = best_uav

    D_access = access_delay_val
    D_compute = sum(step.cost_computation for step in steps)
    D_trans = sum(step.cost_transmission for step in steps)
    D_return = downlink_delay(uav_objects[cur], task, comm_cfg)
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
    access_delays: np.ndarray,
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
    cfg: SchedulingConfig,
    comm_cfg,
) -> Tuple[List[TaskExecutionPlan], float]:
    """Schedule all tasks and compute weighted total delay."""
    plans: List[TaskExecutionPlan] = []
    D_weighted = 0.0
    remaining_compute = _build_compute_budgets(C_map, cfg)

    for task_idx, task in enumerate(tasks):
        access_uav = access_assignment[task_idx]
        access_delay_val = access_delays[task_idx, uav_indices[access_uav]]
        plan = schedule_task(
            task=task,
            access_uav=access_uav,
            uavs=uavs,
            deployment=deployment,
            deployed_by_uav=deployed_by_uav,
            substitutable_sets=substitutable_sets,
            similarity=similarity,
            G=G,
            R=R,
            uav_indices=uav_indices,
            F_map=F_map,
            C_map=C_map,
            remaining_compute=remaining_compute,
            access_delay_val=access_delay_val,
            cfg=cfg,
            comm_cfg=comm_cfg,
        )
        plans.append(plan)
        D_weighted += task.omega * plan.D_total

    return plans, D_weighted
