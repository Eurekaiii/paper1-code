"""
Command-line entry point for the AeroMoDE demo.

Most project logic lives in specialized modules:
- scenario.py builds UAV/expert/task inputs.
- pipeline.py runs placement and scheduling.
"""

from .config import SystemConfig
from .baselines import print_comparison_table, run_all_methods
from .pipeline import print_report, run_pipeline
from .scenario import (
    build_example_scenario,
    build_hotspot_similarity_scenario,
    build_random_scenario,
)

__all__ = [
    "build_example_scenario",
    "build_hotspot_similarity_scenario",
    "build_random_scenario",
    "print_comparison_table",
    "print_report",
    "run_all_methods",
    "run_pipeline",
]


def main() -> None:
    cfg = SystemConfig()

    print("Building hotspot-similarity scenario ...")
    uavs, experts, tasks = build_hotspot_similarity_scenario(cfg)
    print(f"  {len(uavs)} UAVs, {len(experts)} experts, {len(tasks)} tasks")

    print("Running Proposed and baseline methods ...")
    results = run_all_methods(uavs, experts, tasks, cfg)

    proposed_name, proposed_result = results[0]
    print_report(proposed_result, uavs, experts)
    print_comparison_table(results)


if __name__ == "__main__":
    main()
