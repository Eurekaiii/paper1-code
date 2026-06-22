"""
Scenario builders for AeroMoDE experiments.

This module owns the generation of UAVs, experts, and tasks. The pipeline
module consumes these objects but does not decide how they are created.
"""

from typing import List, Tuple

import numpy as np

from .config import SystemConfig
from .models import Expert, Task, UAV


def _unit_vector(rng: np.random.Generator, dim: int) -> np.ndarray:
    """Sample a normalized vector for cosine-similarity experiments."""
    vec = rng.normal(0.0, 1.0, dim)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return _unit_vector(rng, dim)
    return vec / norm


def _grid_positions(num_points: int, area_size: float) -> List[np.ndarray]:
    """Place UAVs on a regular grid to create separated service regions."""
    side = int(np.ceil(np.sqrt(num_points)))
    margin = area_size / (side + 1)
    coords = np.linspace(margin, area_size - margin, side)
    positions: List[np.ndarray] = []
    for y in coords:
        for x in coords:
            positions.append(np.array([x, y]))
            if len(positions) == num_points:
                return positions
    return positions


def build_random_scenario(
    cfg: SystemConfig,
    num_uavs: int = 4,
    num_experts: int = 8,
    num_tasks: int = 10,
    area_size: float = 240.0,
    vector_dim: int = 64,
) -> Tuple[List[UAV], List[Expert], List[Task]]:
    """Construct a reproducible random but feasible UAV/task/expert scenario."""
    if num_uavs <= 0:
        raise ValueError("num_uavs must be positive.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive.")
    if area_size <= 0:
        raise ValueError("area_size must be positive.")
    if vector_dim <= 0:
        raise ValueError("vector_dim must be positive.")

    rng = np.random.default_rng(cfg.seed)
    ground_positions = _grid_positions(num_uavs, area_size)

    uavs: List[UAV] = []
    for u_id in range(num_uavs):
        position = np.array(
            [
                ground_positions[u_id][0],
                ground_positions[u_id][1],
                rng.uniform(70.0, 90.0),
            ]
        )
        uavs.append(
            UAV(
                id=u_id,
                position=position,
                C_u=rng.uniform(5e9, 12e9),
                M_u=rng.uniform(1.8e9, 2.6e9),
                P_u=rng.uniform(8.0, 14.0),
            )
        )

    weight_vectors = [_unit_vector(rng, vector_dim) for _ in range(num_experts)]

    # Create near-duplicate experts so the substitution mechanism has candidates.
    for e_id in range(num_experts // 2, num_experts):
        src_id = e_id - num_experts // 2
        noisy = weight_vectors[src_id] + 0.15 * rng.normal(0.0, 1.0, vector_dim)
        weight_vectors[e_id] = noisy / np.linalg.norm(noisy)

    experts: List[Expert] = []
    for e_id in range(num_experts):
        experts.append(
            Expert(
                id=e_id,
                W_e=rng.uniform(220e6, 420e6),
                F_e=rng.uniform(4e8, 8e8),
                weight_vector=weight_vectors[e_id],
            )
        )

    tasks: List[Task] = []
    for task_id in range(num_tasks):
        anchor = uavs[task_id % num_uavs]
        offset = rng.normal(0.0, area_size / 30.0, size=2)
        position = np.clip(anchor.position[:2] + offset, 0.0, area_size)
        num_layers = int(rng.integers(2, 5))
        expert_sequence = (
            rng.choice(np.arange(num_experts), size=num_layers, replace=True)
            .astype(int)
            .tolist()
        )
        first_mid = rng.uniform(3e5, 1.0e6)
        tasks.append(
            Task(
                id=task_id,
                position=position,
                S_in=rng.uniform(0.8e6, 3.0e6),
                S_out=rng.uniform(8e3, 3e4),
                S_mid=[first_mid * (0.85**i) for i in range(num_layers)],
                expert_sequence=expert_sequence,
                omega=rng.uniform(0.6, 1.6),
            )
        )

    return uavs, experts, tasks


def build_example_scenario(
    cfg: SystemConfig,
) -> Tuple[List[UAV], List[Expert], List[Task]]:
    """Construct a small deterministic scenario useful for debugging."""
    rng = np.random.default_rng(cfg.seed)

    uavs = [
        UAV(
            id=0,
            position=np.array([0.0, 0.0, 80.0]),
            C_u=1e10,
            M_u=2e9,
            P_u=10.0,
        ),
        UAV(
            id=1,
            position=np.array([120.0, 0.0, 70.0]),
            C_u=5e9,
            M_u=1.5e9,
            P_u=8.0,
        ),
        UAV(
            id=2,
            position=np.array([60.0, 100.0, 90.0]),
            C_u=8e9,
            M_u=1.8e9,
            P_u=12.0,
        ),
    ]

    dim = 64
    base_vecs = {e_id: _unit_vector(rng, dim) for e_id in range(6)}
    base_vecs[3] = base_vecs[0] + 0.15 * rng.normal(0, 1, dim)
    base_vecs[4] = base_vecs[1] + 0.15 * rng.normal(0, 1, dim)
    base_vecs[3] = base_vecs[3] / np.linalg.norm(base_vecs[3])
    base_vecs[4] = base_vecs[4] / np.linalg.norm(base_vecs[4])

    experts = [
        Expert(id=0, W_e=300e6, F_e=5e8, weight_vector=base_vecs[0]),
        Expert(id=1, W_e=350e6, F_e=6e8, weight_vector=base_vecs[1]),
        Expert(id=2, W_e=280e6, F_e=4.5e8, weight_vector=base_vecs[2]),
        Expert(id=3, W_e=320e6, F_e=5.2e8, weight_vector=base_vecs[3]),
        Expert(id=4, W_e=360e6, F_e=6.2e8, weight_vector=base_vecs[4]),
        Expert(id=5, W_e=400e6, F_e=7e8, weight_vector=base_vecs[5]),
    ]

    tasks = [
        Task(
            id=0,
            position=np.array([10.0, 10.0]),
            S_in=1e6,
            S_out=1e4,
            S_mid=[5e5, 5e5, 5e5],
            expert_sequence=[0, 1, 2],
            omega=1.0,
        ),
        Task(
            id=1,
            position=np.array([50.0, 20.0]),
            S_in=2e6,
            S_out=1e4,
            S_mid=[8e5, 8e5],
            expert_sequence=[1, 3],
            omega=0.8,
        ),
        Task(
            id=2,
            position=np.array([100.0, 10.0]),
            S_in=1.5e6,
            S_out=1e4,
            S_mid=[6e5, 6e5, 6e5],
            expert_sequence=[0, 4, 5],
            omega=1.2,
        ),
        Task(
            id=3,
            position=np.array([30.0, 80.0]),
            S_in=1e6,
            S_out=1e4,
            S_mid=[4e5, 4e5],
            expert_sequence=[2, 5],
            omega=0.9,
        ),
        Task(
            id=4,
            position=np.array([80.0, 70.0]),
            S_in=2.5e6,
            S_out=1e4,
            S_mid=[1e6, 1e6, 1e6, 1e6],
            expert_sequence=[0, 1, 2, 5],
            omega=1.5,
        ),
    ]

    return uavs, experts, tasks
