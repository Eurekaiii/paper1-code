"""
AeroMoDE Communication Model
==============================
Ground-to-UAV access links and UAV-to-UAV single-hop links.

Reference: aeromde_main.tex  Section III-B  (wireless links),
                              Section IV-B   (single-hop simplification).
"""

from typing import Dict, Tuple, List
import numpy as np

from .config import ChannelConfig
from .models import UAV, Task


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------
def _ground_to_uav_distance(task_pos: np.ndarray, uav: UAV) -> float:
    """d_{k,u} = sqrt((x_k-x_u)^2 + (y_k-y_u)^2 + H_u^2)."""
    dx = task_pos[0] - uav.x
    dy = task_pos[1] - uav.y
    return np.sqrt(dx * dx + dy * dy + uav.H * uav.H)


def _uav_to_uav_distance(uav_a: UAV, uav_b: UAV) -> float:
    """d_{u,v} = sqrt((x_u-x_v)^2 + (y_u-y_v)^2 + (H_u-H_v)^2)."""
    dx = uav_a.x - uav_b.x
    dy = uav_a.y - uav_b.y
    dz = uav_a.H - uav_b.H
    return np.sqrt(dx * dx + dy * dy + dz * dz)


# ---------------------------------------------------------------------------
# Channel gain & SNR
# ---------------------------------------------------------------------------
def channel_gain(distance: float, cfg: ChannelConfig) -> float:
    """h = β_0 · d^{-α}."""
    return cfg.beta_0 * (distance ** (-cfg.alpha))


# ---------------------------------------------------------------------------
# Ground → UAV access
# ---------------------------------------------------------------------------
def access_snr(task_pos: np.ndarray, uav: UAV, cfg: ChannelConfig) -> float:
    """ρ_{k,u}^{access} = P_k · h_{k,u} / (N_0 · B)."""
    d = _ground_to_uav_distance(task_pos, uav)
    h = channel_gain(d, cfg)
    return (cfg.P_ground * h) / (cfg.N0 * cfg.B)


def access_rate(task: Task, uav: UAV, cfg: ChannelConfig) -> float:
    """R_{k,u}^{access} = g_{k,u}^{access} · B · log₂(1 + ρ).

    Returns 0 if the link is unavailable.
    """
    rho = access_snr(task.position, uav, cfg)
    if rho < cfg.rho_th:
        return 0.0
    return cfg.B * np.log2(1.0 + rho)


def access_delay(task: Task, uav: UAV, cfg: ChannelConfig) -> float:
    """D_{k,u}^{access} = S_k^{in} / R_{k,u}^{access}.

    Returns +∞ if the link is unavailable.
    """
    rate = access_rate(task, uav, cfg)
    if rate <= 0.0:
        return float("inf")
    return task.S_in / rate


def default_access_uav(task: Task, uavs: List[UAV], cfg: ChannelConfig) -> int:
    """u_k^a = argmin_u  S_k^{in} / R_{k,u}^{access}  (Eq. 7)."""
    best_uav = -1
    best_delay = float("inf")
    for uav in uavs:
        dly = access_delay(task, uav, cfg)
        if dly < best_delay:
            best_delay = dly
            best_uav = uav.id
    if best_uav < 0:
        raise RuntimeError(f"Task {task.id}: no reachable UAV for access.")
    return best_uav


# ---------------------------------------------------------------------------
# UAV ↔ UAV single-hop
# ---------------------------------------------------------------------------
def uav_snr(uav_tx: UAV, uav_rx: UAV, cfg: ChannelConfig) -> float:
    """ρ_{u,v} = P_u · h_{u,v} / (N_0 · B)."""
    d = _uav_to_uav_distance(uav_tx, uav_rx)
    h = channel_gain(d, cfg)
    return (cfg.P_uav * h) / (cfg.N0 * cfg.B)


def uav_link_available(uav_tx: UAV, uav_rx: UAV, cfg: ChannelConfig) -> bool:
    """g_{u,v} = 1 if ρ ≥ ρ_th else 0."""
    return uav_snr(uav_tx, uav_rx, cfg) >= cfg.rho_th


def uav_rate(uav_tx: UAV, uav_rx: UAV, cfg: ChannelConfig) -> float:
    """R_{u,v} = g_{u,v} · B · log₂(1 + ρ_{u,v})."""
    if not uav_link_available(uav_tx, uav_rx, cfg):
        return 0.0
    rho = uav_snr(uav_tx, uav_rx, cfg)
    return cfg.B * np.log2(1.0 + rho)


def single_hop_delay(uav_tx: UAV, uav_rx: UAV, data_size: float,
                     cfg: ChannelConfig) -> float:
    """T_{tx,rx}^{single}(S) — single-hop transmission delay.

    0 if tx == rx, S/R if link available, +∞ otherwise.
    """
    if uav_tx.id == uav_rx.id:
        return 0.0
    rate = uav_rate(uav_tx, uav_rx, cfg)
    if rate <= 0.0:
        return float("inf")
    return data_size / rate


# ---------------------------------------------------------------------------
# Bulk pre-computation
# ---------------------------------------------------------------------------
def build_connectivity_matrix(uavs: List[UAV], cfg: ChannelConfig) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Pre-compute g_{u,v} and R_{u,v} for all UAV pairs.

    Returns
    -------
    G : np.ndarray   shape (U, U)   G[u,v] = g_{u,v} ∈ {0, 1}
    R : np.ndarray   shape (U, U)   R[u,v] = R_{u,v} in bps
    """
    U = len(uavs)
    G = np.zeros((U, U), dtype=np.int32)
    Rmat = np.zeros((U, U), dtype=np.float64)

    for i, u in enumerate(uavs):
        for j, v in enumerate(uavs):
            if i == j:
                G[i, j] = 1
                Rmat[i, j] = float("inf")   # local → "infinite" rate
            elif uav_link_available(u, v, cfg):
                G[i, j] = 1
                Rmat[i, j] = uav_rate(u, v, cfg)

    return G, Rmat


def build_access_info(tasks: List[Task], uavs: List[UAV], cfg: ChannelConfig) \
        -> Tuple[List[int], np.ndarray]:
    """Pre-compute default access UAV and access delays for all tasks.

    Returns
    -------
    access_uav : List[int]     length K  — u_k^a for each task
    access_delays : np.ndarray shape (K, U) — D_{k,u}^{access}
    """
    K, U = len(tasks), len(uavs)
    access_delays = np.full((K, U), np.inf, dtype=np.float64)
    access_uav = []

    for ki, task in enumerate(tasks):
        best_delay = float("inf")
        best_u = -1
        for ui, uav in enumerate(uavs):
            dly = access_delay(task, uav, cfg)
            access_delays[ki, ui] = dly
            if dly < best_delay:
                best_delay = dly
                best_u = uav.id
        access_uav.append(best_u)

    return access_uav, access_delays
