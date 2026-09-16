"""Run the real MoE trace-driven AeroMoDE experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import SystemConfig
from ..evaluation import run_all_methods
from ..reporting import print_comparison_table, print_deployment_details
from .export_results import export_real_moe_results, summarize_results
from .scenario_builder import build_real_moe_trace_scenario


def run_real_moe_experiment(
    trace_dir: str | Path = "data/real_moe/traces/switch_base_8",
    output_dir: str | Path = "results/real_moe",
    uav_scenario: str | Path | None = None,
    seed: int = 42,
    memory_scale: float = 1.0,
    expert_memory_scale: float = 1.0,
    max_copies_per_expert: int = 1,
) -> None:
    """Build a trace-driven scenario, run all methods, and export CSV files."""

    cfg = SystemConfig()
    cfg.seed = seed
    cfg.max_copies_per_expert = max_copies_per_expert

    uavs, experts, tasks = build_real_moe_trace_scenario(
        cfg,
        trace_dir=trace_dir,
        uav_scenario_path=uav_scenario,
        memory_scale=memory_scale,
        expert_memory_scale=expert_memory_scale,
    )
    results = run_all_methods(uavs, experts, tasks, cfg)
    export_real_moe_results(results, output_dir=output_dir)

    print(summarize_results(results))
    print_comparison_table(results)
    print_deployment_details(results, uavs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real MoE trace-driven experiment.")
    parser.add_argument("--trace-dir", default="data/real_moe/traces/switch_base_8")
    parser.add_argument("--output-dir", default="results/real_moe")
    parser.add_argument("--uav-scenario", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-scale", type=float, default=1.0)
    parser.add_argument("--expert-memory-scale", type=float, default=1.0)
    parser.add_argument("--max-copies-per-expert", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_real_moe_experiment(
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        uav_scenario=args.uav_scenario,
        seed=args.seed,
        memory_scale=args.memory_scale,
        expert_memory_scale=args.expert_memory_scale,
        max_copies_per_expert=args.max_copies_per_expert,
    )


if __name__ == "__main__":
    main()
