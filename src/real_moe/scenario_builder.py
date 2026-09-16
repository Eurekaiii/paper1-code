"""Build AeroMoDE scenarios from real MoE trace bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..config import SystemConfig
from ..models import Expert, Task, UAV
from .trace_schema import ExpertTrace, TraceBundle, read_trace_bundle, validate_trace_bundle


def _layer_aware_vector(
    expert: ExpertTrace,
    max_layer: int,
    layer_marker_scale: float,
) -> np.ndarray:
    """Append a layer marker so substitutions mostly stay within a MoE layer."""

    base = np.asarray(expert.weight_vector, dtype=np.float64)
    if layer_marker_scale <= 0:
        return base
    marker = np.zeros(max_layer + 1, dtype=np.float64)
    if 0 <= expert.layer <= max_layer:
        marker[expert.layer] = layer_marker_scale
    combined = np.concatenate([base, marker])
    norm = np.linalg.norm(combined)
    return combined / norm if norm > 1e-12 else combined


def _default_uavs(cfg: SystemConfig, memory_scale: float) -> List[UAV]:
    """Use a controlled four-UAV hotspot layout for trace-driven experiments."""

    rng = np.random.default_rng(cfg.seed)
    base_positions = [
        np.array([60.0, 60.0]),
        np.array([180.0, 60.0]),
        np.array([60.0, 180.0]),
        np.array([180.0, 180.0]),
    ]
    uavs: List[UAV] = []
    for uav_id, base_position in enumerate(base_positions):
        xy = np.clip(base_position + rng.uniform(-8.0, 8.0, size=2), 0.0, 240.0)
        uavs.append(
            UAV(
                id=uav_id,
                position=np.array([xy[0], xy[1], rng.uniform(70.0, 90.0)]),
                C_u=rng.uniform(8e9, 12e9),
                M_u=rng.uniform(1.0e9, 1.3e9) * memory_scale,
                P_u=rng.uniform(10.0, 14.0),
            )
        )
    return uavs


def _load_uavs_from_scenario(path: str | Path, memory_scale: float) -> List[UAV]:
    """Load simulator-ready UAV nodes from a JSON scenario file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    uavs: List[UAV] = []
    for item in payload.get("uavs", []):
        uavs.append(
            UAV(
                id=int(item["id"]),
                position=np.asarray(item["position_m"], dtype=np.float64),
                C_u=float(item["C_u"]),
                M_u=float(item["M_u"]) * memory_scale,
                P_u=float(item["P_u"]),
            )
        )
    if not uavs:
        raise ValueError(f"No UAVs found in scenario file: {path}")
    return uavs


def _task_position(task_id: int, uavs: List[UAV], seed: int) -> np.ndarray:
    """Place real text tasks around UAV hotspots for communication simulation."""

    rng = np.random.default_rng(seed + 31_337 + task_id)
    anchor = uavs[task_id % len(uavs)]
    return np.clip(anchor.position[:2] + rng.normal(0.0, 12.0, size=2), 0.0, 240.0)


def build_real_moe_trace_scenario(
    cfg: SystemConfig,
    trace_dir: str | Path = "data/real_moe/traces/switch_base_8",
    uav_scenario_path: str | Path | None = None,
    memory_scale: float = 1.0,
    expert_memory_scale: float = 1.0,
    reference_compute: float = 1.0e10,
    layer_marker_scale: float = 2.0,
) -> Tuple[List[UAV], List[Expert], List[Task]]:
    """Convert a real MoE trace bundle into UAV, Expert, and Task objects.

    Parameters
    ----------
    cfg:
        Existing system configuration.
    trace_dir:
        Directory containing metadata.json, experts.json, and tasks.json.
    memory_scale:
        Multiplier for UAV memory.  This lets the trace experiment control
        memory pressure without rewriting the trace itself.
    expert_memory_scale:
        Multiplier for real expert parameter memory.  This supports scaled MoE
        studies where router traces come from a small open model but the
        deployment stress mimics larger experts.
    reference_compute:
        Reference FLOP/s used to convert measured standalone expert latency
        into an equivalent ``F_e`` when latency is available.
    layer_marker_scale:
        Appended one-hot layer marker strength for expert vectors.  A positive
        value discourages cross-layer substitutions while still using real
        expert weights within each layer.
    """

    bundle: TraceBundle = read_trace_bundle(trace_dir)
    validate_trace_bundle(bundle)

    max_layer = max(expert.layer for expert in bundle.experts)
    experts = [
        Expert(
            id=trace.id,
            W_e=float(trace.parameter_bytes) * expert_memory_scale,
            F_e=(
                float(trace.latency_s) * reference_compute
                if trace.latency_s > 0
                else float(trace.flops)
            ),
            weight_vector=_layer_aware_vector(
                trace,
                max_layer=max_layer,
                layer_marker_scale=layer_marker_scale,
            ),
        )
        for trace in bundle.experts
    ]

    if uav_scenario_path is None:
        uavs = _default_uavs(cfg, memory_scale=memory_scale)
    else:
        uavs = _load_uavs_from_scenario(uav_scenario_path, memory_scale=memory_scale)
    tasks = [
        Task(
            id=trace.id,
            position=_task_position(trace.id, uavs, cfg.seed),
            S_in=float(trace.S_in),
            S_out=float(trace.S_out),
            S_mid=[float(value) for value in trace.S_mid],
            expert_sequence=[int(value) for value in trace.expert_sequence],
            omega=1.0,
            gating_scores={int(key): float(value) for key, value in trace.gating_scores.items()},
        )
        for trace in bundle.tasks
    ]
    return uavs, experts, tasks
