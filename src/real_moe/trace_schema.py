"""Trace schemas and JSON helpers for real MoE experiments.

The trace files are intentionally model-agnostic.  The extractor writes these
records after observing a real MoE model, and the scenario builder converts
them into the existing UAV/Expert/Task objects used by the AeroMoDE pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ExpertTrace:
    """Real-model metadata for one expert."""

    id: int
    layer: int
    local_expert: int
    parameter_bytes: float
    flops: float
    latency_s: float
    weight_vector: List[float]


@dataclass
class TaskTrace:
    """Router-derived workload record for one text input."""

    id: int
    text: str
    token_count: int
    S_in: float
    S_out: float
    S_mid: List[float]
    expert_sequence: List[int]
    gating_scores: Dict[int, float]
    layer_experts: List[int] = field(default_factory=list)


@dataclass
class TraceMetadata:
    """Provenance for a real MoE trace bundle."""

    model_name: str
    device: str
    num_tasks: int
    num_experts: int
    notes: str = ""


@dataclass
class TraceBundle:
    """Complete on-disk trace bundle."""

    metadata: TraceMetadata
    experts: List[ExpertTrace]
    tasks: List[TaskTrace]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_trace_bundle(bundle: TraceBundle, trace_dir: str | Path) -> None:
    """Write a trace bundle as metadata.json, experts.json, and tasks.json."""

    out = Path(trace_dir)
    _write_json(out / "metadata.json", asdict(bundle.metadata))
    _write_json(out / "experts.json", [asdict(expert) for expert in bundle.experts])
    _write_json(out / "tasks.json", [asdict(task) for task in bundle.tasks])


def read_trace_bundle(trace_dir: str | Path) -> TraceBundle:
    """Read a trace bundle from a directory."""

    root = Path(trace_dir)
    metadata_raw = _read_json(root / "metadata.json")
    experts_raw = _read_json(root / "experts.json")
    tasks_raw = _read_json(root / "tasks.json")

    metadata = TraceMetadata(**metadata_raw)
    experts = [ExpertTrace(**item) for item in experts_raw]
    tasks = [
        TaskTrace(
            id=int(item["id"]),
            text=str(item["text"]),
            token_count=int(item["token_count"]),
            S_in=float(item["S_in"]),
            S_out=float(item["S_out"]),
            S_mid=[float(value) for value in item["S_mid"]],
            expert_sequence=[int(value) for value in item["expert_sequence"]],
            gating_scores={
                int(key): float(value)
                for key, value in item.get("gating_scores", {}).items()
            },
            layer_experts=[int(value) for value in item.get("layer_experts", [])],
        )
        for item in tasks_raw
    ]
    return TraceBundle(metadata=metadata, experts=experts, tasks=tasks)


def validate_trace_bundle(bundle: TraceBundle) -> None:
    """Raise ValueError if a trace bundle is not usable by the scenario builder."""

    if not bundle.experts:
        raise ValueError("Trace bundle has no experts.")
    if not bundle.tasks:
        raise ValueError("Trace bundle has no tasks.")

    expert_ids = {expert.id for expert in bundle.experts}
    if len(expert_ids) != len(bundle.experts):
        raise ValueError("Trace bundle contains duplicate expert ids.")

    for expert in bundle.experts:
        if expert.parameter_bytes <= 0:
            raise ValueError(f"Expert {expert.id} has non-positive parameter_bytes.")
        if expert.flops <= 0:
            raise ValueError(f"Expert {expert.id} has non-positive flops.")
        if not expert.weight_vector:
            raise ValueError(f"Expert {expert.id} has an empty weight_vector.")

    for task in bundle.tasks:
        if task.token_count <= 0:
            raise ValueError(f"Task {task.id} has non-positive token_count.")
        if not task.expert_sequence:
            raise ValueError(f"Task {task.id} has an empty expert_sequence.")
        missing = set(task.expert_sequence) - expert_ids
        if missing:
            raise ValueError(
                f"Task {task.id} references unknown experts: {sorted(missing)}."
            )
