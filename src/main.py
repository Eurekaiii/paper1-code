"""
Command-line entry point for the AeroMoDE demo.

Most project logic lives in specialized modules:
- scenario.py builds UAV/expert/task inputs.
- pipeline.py runs placement and scheduling.
"""

from .config import SystemConfig
from .pipeline import print_report, run_pipeline
from .scenario import build_example_scenario, build_random_scenario

__all__ = [
    "build_example_scenario",
    "build_random_scenario",
    "print_report",
    "run_pipeline",
]


def main() -> None:
    cfg = SystemConfig()

    print("Building random scenario ...")
    uavs, experts, tasks = build_random_scenario(cfg)
    print(f"  {len(uavs)} UAVs, {len(experts)} experts, {len(tasks)} tasks")

    print("Running AeroMoDE pipeline ...")
    result = run_pipeline(uavs, experts, tasks, cfg)

    print_report(result, uavs, experts)


if __name__ == "__main__":
    main()
