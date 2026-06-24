"""
Command-line entry point for the AeroMoDE demo.

Most project logic lives in specialized modules:
- scenario.py builds UAV/expert/task inputs.
- pipeline.py runs placement and scheduling.
"""

from .config import SystemConfig
from .evaluation import run_all_methods, run_multi_seed_evaluation
from .reporting import (
    print_comparison_table,
    print_deployment_details,
    print_multi_seed_diagnostic_table,
    print_multi_seed_table,
    print_multi_seed_winner_table,
    print_substitution_pair_details,
)
from .pipeline import print_report, run_pipeline
from .scenario import (
    build_controlled_random_hotspot_scenario,
    build_controlled_random_hotspot_scenario_v2,
    build_example_scenario,
    build_hotspot_similarity_scenario,
    build_random_scenario,
    build_scalable_hotspot_scenario,
)

__all__ = [
    "build_controlled_random_hotspot_scenario",
    "build_controlled_random_hotspot_scenario_v2",
    "build_example_scenario",
    "build_hotspot_similarity_scenario",
    "build_random_scenario",
    "build_scalable_hotspot_scenario",
    "print_comparison_table",
    "print_deployment_details",
    "print_multi_seed_diagnostic_table",
    "print_multi_seed_table",
    "print_multi_seed_winner_table",
    "print_report",
    "print_substitution_pair_details",
    "run_all_methods",
    "run_multi_seed_evaluation",
    "run_pipeline",
]


def main() -> None:
    cfg = SystemConfig()

    print("Building hotspot-similarity scenario ...")
    # To test randomized controlled scenarios, temporarily replace this call
    # with build_controlled_random_hotspot_scenario(cfg) or
    # build_controlled_random_hotspot_scenario_v2(cfg).
    uavs, experts, tasks = build_hotspot_similarity_scenario(cfg)
    print(f"  {len(uavs)} UAVs, {len(experts)} experts, {len(tasks)} tasks")

    print("Running Proposed and baseline methods ...")
    results = run_all_methods(uavs, experts, tasks, cfg)

    proposed_name, proposed_result = results[0]
    print_report(proposed_result, uavs, experts)
    print_comparison_table(results)
    print_deployment_details(results, uavs)
    print_substitution_pair_details(results)

    print("\nRunning multi-seed evaluation ...")
    multi_seed_metrics = run_multi_seed_evaluation(
        seeds=[0, 1, 2, 3, 4],
        scenario_builder=build_hotspot_similarity_scenario,
    )
    print_multi_seed_diagnostic_table(multi_seed_metrics, seeds=[0, 1, 2, 3, 4])
    print_multi_seed_winner_table(multi_seed_metrics, seeds=[0, 1, 2, 3, 4])
    print_multi_seed_table(multi_seed_metrics)


if __name__ == "__main__":
    main()
