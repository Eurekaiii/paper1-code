"""
AeroMoDE Data Models
=====================
UAV, Task, and Expert data structures used throughout the pipeline.

Reference: aeromde_main.tex Sections III & IV.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# UAV
# ---------------------------------------------------------------------------
@dataclass
class UAV:
    """A single UAV node in the quasi-static aerial edge network.

    Attributes
    ----------
    id : int
        UAV index u ∈ 𝒰 = {1, ..., U}.
    position : np.ndarray
        Spatial location l_u = (x_u, y_u, H_u) in metres.
    C_u : float
        Available compute capability (FLOP/s).
    M_u : float
        Available memory capacity for hosting expert parameters (bytes).
    """
    id: int
    position: np.ndarray          # shape (3,)  → (x, y, H)
    C_u: float                    # FLOP/s
    M_u: float                    # bytes

    @property
    def x(self) -> float: return self.position[0]
    @property
    def y(self) -> float: return self.position[1]
    @property
    def H(self) -> float: return self.position[2]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
@dataclass
class Task:
    """A ground inference task.

    Attributes
    ----------
    id : int
        Task index k ∈ 𝒦.
    position : np.ndarray
        Ground location q_k = (x_k, y_k).  H = 0 (ground level).
    S_in : float
        Input data size (bits).
    S_out : float
        Output result size (bits).
    S_mid : List[float]
        Intermediate feature sizes after each expert step l
        (S_{k,l}^{mid} for l = 1, ..., L_k).  Length = L_k.
    expert_sequence : List[int]
        Ordered expert indices E_k = [e_1, e_2, ..., e_{L_k}]
        that the task's gating network selected.
    omega : float
        Task weight / priority ω_k.
    gating_scores : Optional[Dict[int, float]]
        Per-expert gating scores q_{k,r} (summed across tokens & layers).
        If None, demand is computed by occurrence count in expert_sequence.
    """
    id: int
    position: np.ndarray                    # shape (2,)  → (x_k, y_k)
    S_in: float                             # bits
    S_out: float                            # bits
    S_mid: List[float]                      # [S_{k,1}^{mid}, ..., S_{k,L_k}^{mid}]
    expert_sequence: List[int]              # E_k = [e1, e2, ...]
    omega: float = 1.0
    gating_scores: Optional[Dict[int, float]] = None

    @property
    def x(self) -> float: return self.position[0]

    @property
    def y(self) -> float: return self.position[1]

    @property
    def L(self) -> int:
        """Number of expert steps for this task."""
        return len(self.expert_sequence)

    def get_demand_strength(self, expert_id: int) -> float:
        """Return q_{k,r}: demand strength of task k for expert r."""
        if self.gating_scores is not None:
            return self.gating_scores.get(expert_id, 0.0)
        # Fallback: count occurrences in expert sequence
        return float(self.expert_sequence.count(expert_id))


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------
@dataclass
class Expert:
    """A candidate expert for deployment.

    Attributes
    ----------
    id : int
        Expert index e ∈ ℰ.
    W_e : float
        Memory footprint of expert parameters (bytes).
    F_e : float
        Computational cost of one forward pass (FLOPs).
    weight_vector : Optional[np.ndarray]
        Flattened FFN weight vector vec(W_e), used for cosine similarity.
        If None, similarity must be provided externally.
    """
    id: int
    W_e: float                            # bytes
    F_e: float                            # FLOPs
    weight_vector: Optional[np.ndarray] = None   # vec(W_e) for similarity


# ---------------------------------------------------------------------------
# Deployment & Execution result types
# ---------------------------------------------------------------------------
@dataclass
class ExpertStep:
    """One step in a task's execution plan."""
    layer_idx: int                        # l (0-indexed)
    original_expert: int                  # e_{k,l} — what gating wanted
    actual_expert: int                    # ê*_{k,l} — what was actually used
    uav_id: int                           # u*_{k,l} — where it ran
    is_substituted: bool                  # True if actual ≠ original
    cost_transmission: float              # transmission delay (s)
    cost_computation: float               # computation delay (s)
    cost_error: float                     # λ · Err penalty


@dataclass
class TaskExecutionPlan:
    """Full execution trace for one task."""
    task_id: int
    access_uav: int                       # u_k^a
    steps: List[ExpertStep]
    D_access: float
    D_compute: float
    D_trans: float
    D_return: float
    D_total: float


@dataclass
class SystemResult:
    """Complete system output."""
    deployment: Dict[Tuple[int, int], int]   # (e, u) → m_{e,u}
    execution_plans: List[TaskExecutionPlan]
    D_total: float
    D_weighted: float                        # Σ ω_k D_k
