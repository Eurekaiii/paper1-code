"""Console reporting helpers for experiment diagnostics."""

from typing import Dict, List, Tuple

import numpy as np

from .evaluation import MethodMetrics, compute_metrics, count_substitution_pairs
from .models import SystemResult, UAV


def print_comparison_table(results: List[Tuple[str, SystemResult]]) -> None:
    """Print the requested method comparison table."""
    metrics = [compute_metrics(name, result) for name, result in results]
    headers = [
        "Method",
        "Substitutions",
        "Deployments",
        "D_total(ms)",
        "AvgAccess(ms)",
        "AvgCompute(ms)",
        "AvgTrans(ms)",
        "AvgReturn(ms)",
    ]
    rows = [
        [
            m.name,
            str(m.substitutions),
            str(m.deployments),
            f"{m.D_total * 1e3:.2f}",
            f"{m.avg_D_access * 1e3:.2f}",
            f"{m.avg_D_compute * 1e3:.2f}",
            f"{m.avg_D_trans * 1e3:.2f}",
            f"{m.avg_D_return * 1e3:.2f}",
        ]
        for m in metrics
    ]

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    print("\n=== Method Comparison ===")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_deployment_details(
    results: List[Tuple[str, SystemResult]],
    uavs: List[UAV],
) -> None:
    """Print per-method expert deployments grouped by UAV."""
    uav_ids = [u.id for u in uavs]

    print("\n=== Deployment Details ===")
    for name, result in results:
        print(f"\n{name}")
        for uav_id in uav_ids:
            expert_ids = sorted(
                e
                for (e, u), deployed in result.deployment.items()
                if deployed == 1 and u == uav_id
            )
            expert_text = ", ".join(f"E{e}" for e in expert_ids) if expert_ids else "-"
            print(f"  UAV {uav_id}: {expert_text}")


def print_substitution_pair_details(results: List[Tuple[str, SystemResult]]) -> None:
    """Print per-method substitution pair counts."""
    print("\n=== Substitution Pairs ===")
    for name, result in results:
        counts = count_substitution_pairs(result)
        print(f"\n{name}")
        if not counts:
            print("  (none)")
            continue
        for (original, actual), count in sorted(counts.items()):
            print(f"  E{original} -> E{actual}: {count}")


def print_multi_seed_table(metrics_by_method: Dict[str, List[MethodMetrics]]) -> None:
    """Print mean/std diagnostics across repeated scenario seeds."""
    headers = [
        "Method",
        "mean D_total(ms)",
        "std D_total(ms)",
        "mean Substitutions",
        "mean AvgCompute(ms)",
        "mean AvgTrans(ms)",
    ]

    rows: List[List[str]] = []
    for name, metrics in metrics_by_method.items():
        D_total = np.array([m.D_total for m in metrics], dtype=float) * 1e3
        substitutions = np.array([m.substitutions for m in metrics], dtype=float)
        avg_compute = np.array([m.avg_D_compute for m in metrics], dtype=float) * 1e3
        avg_trans = np.array([m.avg_D_trans for m in metrics], dtype=float) * 1e3
        rows.append(
            [
                name,
                f"{np.mean(D_total):.2f}",
                f"{np.std(D_total):.2f}",
                f"{np.mean(substitutions):.2f}",
                f"{np.mean(avg_compute):.2f}",
                f"{np.mean(avg_trans):.2f}",
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    print("\n=== Multi-seed Evaluation ===")
    print("Seeds: [0, 1, 2, 3, 4]")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_multi_seed_diagnostic_table(
    metrics_by_method: Dict[str, List[MethodMetrics]],
    seeds: List[int],
) -> None:
    """Print per-seed method diagnostics and each seed's best method."""
    headers = [
        "Seed",
        "Method",
        "D_total(ms)",
        "Substitutions",
        "AvgCompute(ms)",
        "AvgTrans(ms)",
    ]
    method_names = list(metrics_by_method.keys())
    rows: List[List[str]] = []

    for idx, seed in enumerate(seeds):
        for method in method_names:
            metrics = metrics_by_method[method][idx]
            rows.append(
                [
                    str(seed),
                    method,
                    f"{metrics.D_total * 1e3:.2f}",
                    str(metrics.substitutions),
                    f"{metrics.avg_D_compute * 1e3:.2f}",
                    f"{metrics.avg_D_trans * 1e3:.2f}",
                ]
            )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    print("\n=== Multi-seed Per-seed Diagnostics ===")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))

    print("\n=== Multi-seed Winners ===")
    for idx, seed in enumerate(seeds):
        winner = min(method_names, key=lambda method: metrics_by_method[method][idx].D_total)
        print(f"Seed {seed} winner: {winner}")


def print_multi_seed_winner_table(
    metrics_by_method: Dict[str, List[MethodMetrics]],
    seeds: List[int],
) -> None:
    """Print compact per-seed D_total comparison with winner."""
    method_columns = [
        ("Proposed", "Proposed D_total"),
        ("Random Placement", "Random D_total"),
        ("Importance-based Placement", "Importance D_total"),
        ("No-similarity Placement", "No-similarity D_total"),
    ]
    headers = ["Seed", *(label for _, label in method_columns), "Winner"]
    rows: List[List[str]] = []

    for idx, seed in enumerate(seeds):
        available_methods = [
            method for method, _ in method_columns if method in metrics_by_method
        ]
        winner = min(
            available_methods,
            key=lambda method: metrics_by_method[method][idx].D_total,
        )
        rows.append(
            [
                str(seed),
                *[
                    f"{metrics_by_method[method][idx].D_total * 1e3:.2f}"
                    if method in metrics_by_method
                    else "-"
                    for method, _ in method_columns
                ],
                winner,
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    print("\n=== Multi-seed Winner Table ===")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
