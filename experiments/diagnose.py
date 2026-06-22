"""
Diagnosis: why does AeroMoDE sometimes underperform?
======================================================
Deep-dive into specific scenarios where AeroMoDE > NoSubstitute latency.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
from src.config import SystemConfig
from src.models import UAV, Task, Expert
from src.main import run_pipeline
from experiments.scenarios import build_all_scenarios
from experiments.baselines import baseline_no_substitution


def diagnose_scenario(name_filter: str = "homogeneous_4uav"):
    """Run AeroMoDE vs NoSub on a specific scenario and print detailed trace."""
    scenarios = build_all_scenarios(42)
    target = None
    for s in scenarios:
        if s.name == name_filter:
            target = s
            break

    if target is None:
        print(f"Scenario '{name_filter}' not found.")
        return

    print(f"=== Diagnosing: {target.name} ===\n")

    # ---- Run AeroMoDE ----
    print("--- AeroMoDE ---")
    result_aero = run_pipeline(target.uavs, target.experts, target.tasks, target.cfg)

    # ---- Run NoSubstitute ----
    cfg_nosub = copy.deepcopy(target.cfg)
    cfg_nosub.similarity.xi = 1.0
    print("\n--- NoSubstitute ---")
    result_nosub = run_pipeline(target.uavs, target.experts, target.tasks, cfg_nosub)

    # ---- Compare deployments ----
    print("\n\n==============================================")
    print("  DEPLOYMENT COMPARISON")
    print("==============================================")
    W_map = {e.id: e.W_e for e in target.experts}

    aero_deploy = {}
    nosub_deploy = {}
    for (e, u), m in result_aero.deployment.items():
        if m == 1:
            aero_deploy[(e, u)] = W_map.get(e, 0) / 1e6
    for (e, u), m in result_nosub.deployment.items():
        if m == 1:
            nosub_deploy[(e, u)] = W_map.get(e, 0) / 1e6

    print("\nAeroMoDE deployment:")
    for u in sorted(set(u for _, u in aero_deploy)):
        experts_here = [(e, sz) for (e, uu), sz in aero_deploy.items() if uu == u]
        total = sum(sz for _, sz in experts_here)
        cap = target.uavs[u].M_u / 1e6
        print(f"  UAV {u} ({target.uavs[u].C_u/1e9:.1f} GFLOP/s, {cap:.0f} MB): "
              f"{[(f'E{e}', f'{sz:.0f}MB') for e, sz in experts_here]} "
              f"→ {total:.0f}/{cap:.0f} MB")

    print("\nNoSubstitute deployment:")
    for u in sorted(set(u for _, u in nosub_deploy)):
        experts_here = [(e, sz) for (e, uu), sz in nosub_deploy.items() if uu == u]
        total = sum(sz for _, sz in experts_here)
        cap = target.uavs[u].M_u / 1e6
        print(f"  UAV {u} ({target.uavs[u].C_u/1e9:.1f} GFLOP/s, {cap:.0f} MB): "
              f"{[(f'E{e}', f'{sz:.0f}MB') for e, sz in experts_here]} "
              f"→ {total:.0f}/{cap:.0f} MB")

    # ---- Compare task-by-task ----
    print("\n\n==============================================")
    print("  TASK-BY-TASK COMPARISON")
    print("==============================================")

    aero_plans = {p.task_id: p for p in result_aero.execution_plans}
    nosub_plans = {p.task_id: p for p in result_nosub.execution_plans}

    total_aero_d = 0
    total_nosub_d = 0
    n_worse = 0
    n_better = 0

    for task_id in sorted(aero_plans.keys()):
        ap = aero_plans[task_id]
        np_plan = nosub_plans[task_id]

        diff = ap.D_total - np_plan.D_total
        total_aero_d += ap.D_total
        total_nosub_d += np_plan.D_total

        marker = ""
        if diff > 1e-4:
            marker = f"  ← AeroMoDE WORSE by {diff*1e3:.1f} ms"
            n_worse += 1
        elif diff < -1e-4:
            marker = f"  ← AeroMoDE BETTER by {-diff*1e3:.1f} ms"
            n_better += 1

        # Per-step trace
        aero_steps_str = " → ".join(
            f"E{s.actual_expert}@UAV{s.uav_id}"
            + ("(sub!)" if s.is_substituted else "")
            for s in ap.steps
        )
        nosub_steps_str = " → ".join(
            f"E{s.actual_expert}@UAV{s.uav_id}"
            for s in np_plan.steps
        )

        print(f"\nTask {task_id} (access UAV {ap.access_uav}):")
        print(f"  AeroMoDE:     {aero_steps_str}")
        print(f"    D={ap.D_total*1e3:.1f} ms  "
              f"(acc={ap.D_access*1e3:.1f}, cmp={ap.D_compute*1e3:.1f}, "
              f"tx={ap.D_trans*1e3:.1f})")
        print(f"  NoSubstitute: {nosub_steps_str}")
        print(f"    D={np_plan.D_total*1e3:.1f} ms  "
              f"(acc={np_plan.D_access*1e3:.1f}, cmp={np_plan.D_compute*1e3:.1f}, "
              f"tx={np_plan.D_trans*1e3:.1f})")
        print(f"  {marker}")

    print(f"\n\nSummary: AeroMoDE better on {n_better} tasks, worse on {n_worse} tasks")
    print(f"  Total AeroMoDE:    {total_aero_d*1e3:.1f} ms")
    print(f"  Total NoSubstitute: {total_nosub_d*1e3:.1f} ms")
    print(f"  Improvement: {(total_nosub_d - total_aero_d)/total_nosub_d*100:+.1f}%")


if __name__ == "__main__":
    diagnose_scenario("homogeneous_4uav")
