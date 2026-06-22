"""
Experiment Scenario Definitions
=================================
Pre-built scenarios covering the paper's key experimental dimensions:
- UAV count & heterogeneity
- Bandwidth range
- Model/Expert scale
- Task load

Each function returns (uavs, experts, tasks, cfg_overrides).
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from src.config import SystemConfig, ChannelConfig, DeploymentConfig, \
    SchedulingConfig, SimilarityConfig
from src.models import UAV, Task, Expert


# ======================================================================
# Scenario descriptor
# ======================================================================
@dataclass
class Scenario:
    """A named experimental scenario."""
    name: str
    uavs: List[UAV]
    experts: List[Expert]
    tasks: List[Task]
    cfg: SystemConfig


# ======================================================================
# UAV builders
# ======================================================================
def _make_uavs_homogeneous(n: int, spacing: float = 120.0,
                           C_u: float = 1e10, M_u: float = 2e9) -> List[UAV]:
    """n identical UAVs in a line."""
    uavs = []
    for i in range(n):
        pos = np.array([i * spacing, 0.0, 80.0])
        uavs.append(UAV(id=i, position=pos, C_u=C_u, M_u=M_u))
    return uavs


def _make_uavs_heterogeneous(config: List[Tuple[float, float, float, float, float]]) \
        -> List[UAV]:
    """UAVs with specified (x, y, H, C_u, M_u)."""
    uavs = []
    for i, (x, y, H, C, M) in enumerate(config):
        uavs.append(UAV(id=i, position=np.array([x, y, H]), C_u=C, M_u=M))
    return uavs


# ======================================================================
# Expert builders
# ======================================================================
def _make_experts(n: int, dim: int = 64, W_base: float = 300e6,
                  F_base: float = 5e8, similarity_pairs: List[Tuple[int, int, float]] = None,
                  rng: np.random.Generator = None) -> List[Expert]:
    """Build n experts with optional similarity structure.

    similarity_pairs: list of (i, j, similarity) — expert i's weight vector
                      is set close to expert j's with the given cosine sim.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    vecs = {}
    for i in range(n):
        vecs[i] = rng.normal(0, 1, dim)
        vecs[i] = vecs[i] / np.linalg.norm(vecs[i])

    # Override for similarity pairs
    if similarity_pairs:
        for i, j, sim in similarity_pairs:
            # vec_i = sim * vec_j + sqrt(1-sim^2) * orthogonal_noise
            base = vecs[j].copy()
            noise = rng.normal(0, 1, dim)
            noise -= np.dot(noise, base) * base  # orthogonalise
            noise_norm = np.linalg.norm(noise)
            if noise_norm > 1e-12:
                noise = noise / noise_norm
            vecs[i] = sim * base + np.sqrt(max(0, 1 - sim**2)) * noise
            vecs[i] = vecs[i] / np.linalg.norm(vecs[i])

    experts = []
    for i in range(n):
        # Vary W_e and F_e by ±20%
        w_scale = 0.8 + 0.4 * rng.random()
        f_scale = 0.8 + 0.4 * rng.random()
        experts.append(Expert(
            id=i, W_e=W_base * w_scale, F_e=F_base * f_scale,
            weight_vector=vecs[i],
        ))
    return experts


# ======================================================================
# Task builders
# ======================================================================
def _make_tasks_random(n: int, uavs: List[UAV], expert_ids: List[int],
                       max_seq_len: int = 4, rng: np.random.Generator = None) \
                       -> List[Task]:
    """Generate n tasks scattered under the UAV footprint."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Bounding box from UAV positions
    xs = [u.x for u in uavs]
    ys = [u.y for u in uavs]
    x_min, x_max = min(xs) - 50, max(xs) + 50
    y_min, y_max = min(ys) - 50, max(ys) + 50

    tasks = []
    for k in range(n):
        pos = np.array([rng.uniform(x_min, x_max),
                         rng.uniform(y_min, y_max)])
        L = rng.integers(2, max_seq_len + 1)
        seq = list(rng.choice(expert_ids, size=L, replace=True))

        S_in = rng.uniform(0.5e6, 3e6)    # 0.5–3 Mb input
        S_out = rng.uniform(0.5e4, 2e4)   # small output
        S_mid = [rng.uniform(3e5, 1.2e6) for _ in range(L)]

        tasks.append(Task(
            id=k, position=pos, S_in=S_in, S_out=S_out,
            S_mid=S_mid, expert_sequence=seq, omega=1.0,
        ))
    return tasks


# ======================================================================
# Pre-built scenarios
# ======================================================================
def build_all_scenarios(rng_seed: int = 42) -> List[Scenario]:
    """Return all experiment scenarios."""
    rng = np.random.default_rng(rng_seed)
    scenarios: List[Scenario] = []

    # --- Shared experts (8 experts, with similarity structure) ---
    # E0≈E4 (sim 0.85), E1≈E5 (sim 0.80), E2≈E6 (sim 0.82)
    sim_pairs = [(4, 0, 0.85), (5, 1, 0.80), (6, 2, 0.82)]
    experts_8 = _make_experts(8, dim=64, similarity_pairs=sim_pairs, rng=rng)

    # ---- 1. UAV count: 2, 3, 4 (homogeneous) ----
    for n_uav in [2, 3, 4]:
        uavs = _make_uavs_homogeneous(n_uav, spacing=120.0)
        tasks = _make_tasks_random(10, uavs, list(range(8)),
                                   max_seq_len=4, rng=rng)
        cfg = SystemConfig()
        cfg.channel.B = 20e6  # 125 Mbps equivalent in Shannon sense
        scenarios.append(Scenario(
            f"homogeneous_{n_uav}uav", uavs, experts_8, tasks, cfg,
        ))

    # ---- 2. Bandwidth sweep: 5, 10, 20, 40 MHz (on 3-UAV homogeneous) ----
    uavs3 = _make_uavs_homogeneous(3, spacing=120.0)
    tasks_3uav = _make_tasks_random(10, uavs3, list(range(8)),
                                    max_seq_len=4, rng=rng)
    for B_val in [5e6, 10e6, 20e6, 40e6]:
        cfg = SystemConfig()
        cfg.channel.B = B_val
        scenarios.append(Scenario(
            f"bandwidth_{B_val/1e6:.0f}MHz", uavs3, experts_8, tasks_3uav, cfg,
        ))

    # ---- 3. Heterogeneous environments (paper Table 3: D, E, F) ----
    # D: Nano-L + Nano-M
    uavs_D = _make_uavs_heterogeneous([
        (0, 0, 80, 1.2e10, 2.0e9),    # strong
        (120, 0, 70, 5e9, 1.2e9),     # weak
    ])
    # E: Nano-L + Nano-S
    uavs_E = _make_uavs_heterogeneous([
        (0, 0, 80, 1.2e10, 2.0e9),    # strong
        (120, 0, 70, 2e9, 0.7e9),     # very weak
    ])
    # F: Nano-L + Nano-M + Nano-S
    uavs_F = _make_uavs_heterogeneous([
        (0, 0, 80, 1.2e10, 2.0e9),
        (120, 0, 70, 5e9, 1.2e9),
        (240, 0, 65, 2e9, 0.7e9),
    ])
    for name, uavs in [("hetero_D", uavs_D), ("hetero_E", uavs_E),
                        ("hetero_F", uavs_F)]:
        tasks = _make_tasks_random(10, uavs, list(range(8)),
                                   max_seq_len=4, rng=rng)
        scenarios.append(Scenario(name, uavs, experts_8, tasks, SystemConfig()))

    # ---- 4. Task load: light (5) / medium (10) / heavy (20) ----
    uavs3b = _make_uavs_homogeneous(3, spacing=120.0)
    for n_tasks, label in [(5, "light"), (10, "medium"), (20, "heavy")]:
        tasks = _make_tasks_random(n_tasks, uavs3b, list(range(8)),
                                   max_seq_len=4, rng=rng)
        scenarios.append(Scenario(
            f"taskload_{label}", uavs3b, experts_8, tasks, SystemConfig(),
        ))

    # ---- 5. Expert scale: 4, 8, 12 experts ----
    uavs3c = _make_uavs_homogeneous(3, spacing=120.0)
    for n_exp in [4, 8, 12]:
        exp_list = _make_experts(n_exp, dim=64, rng=rng)
        tasks = _make_tasks_random(10, uavs3c, list(range(n_exp)),
                                   max_seq_len=4, rng=rng)
        scenarios.append(Scenario(
            f"expertscale_{n_exp}", uavs3c, exp_list, tasks, SystemConfig(),
        ))

    return scenarios


# ======================================================================
# Sensitivity-analysis scenarios
# ======================================================================
def build_sensitivity_scenarios(rng_seed: int = 42) -> Dict[str, List[Scenario]]:
    """Scenarios for parameter sensitivity analysis.

    Returns dict keyed by parameter name.
    """
    rng = np.random.default_rng(rng_seed)
    uavs = _make_uavs_homogeneous(3, spacing=120.0)
    sim_pairs = [(4, 0, 0.85), (5, 1, 0.80), (6, 2, 0.82)]
    experts = _make_experts(8, dim=64, similarity_pairs=sim_pairs, rng=rng)
    tasks = _make_tasks_random(10, uavs, list(range(8)),
                               max_seq_len=4, rng=rng)

    result: Dict[str, List[Scenario]] = {}

    # ξ (similarity threshold)
    xi_scenarios = []
    for xi in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        cfg = SystemConfig()
        cfg.similarity.xi = xi
        xi_scenarios.append(Scenario(f"xi_{xi}", uavs, experts, tasks, cfg))
    result["xi"] = xi_scenarios

    # λ (error penalty)
    lamb_scenarios = []
    for lam in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        cfg = SystemConfig()
        cfg.scheduling.lamb = lam
        lamb_scenarios.append(Scenario(f"lambda_{lam}", uavs, experts, tasks, cfg))
    result["lambda"] = lamb_scenarios

    # η (neighbour demand weight)
    eta_scenarios = []
    for eta in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cfg = SystemConfig()
        cfg.deployment.eta = eta
        eta_scenarios.append(Scenario(f"eta_{eta}", uavs, experts, tasks, cfg))
    result["eta"] = eta_scenarios

    # ν (redundancy penalty)
    nu_scenarios = []
    for nu in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        cfg = SystemConfig()
        cfg.deployment.nu = nu
        nu_scenarios.append(Scenario(f"nu_{nu}", uavs, experts, tasks, cfg))
    result["nu"] = nu_scenarios

    return result
