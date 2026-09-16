"""Extract real MoE traces from an open-source model.

The default target is ``google/switch-base-8`` because it is small enough for a
first trace-driven experiment and exposes router logits through Hugging Face
Transformers.  The extractor records real router activations, expert weights,
and per-expert latency proxies, then writes a TraceBundle consumed by
``scenario_builder.py``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .trace_schema import (
    ExpertTrace,
    TaskTrace,
    TraceBundle,
    TraceMetadata,
    write_trace_bundle,
)


DEFAULT_TEXTS: List[str] = [
    "Summarize the main risk of deploying language models on resource limited drones.",
    "Classify the sentiment of this message: the rescue team arrived faster than expected.",
    "Answer briefly: why does wireless bandwidth matter for edge inference?",
    "Rewrite this sentence in a formal tone: the UAV network is overloaded today.",
    "Explain one advantage of using sparse experts in a neural network.",
    "Generate a short emergency response instruction for a flooded road.",
    "Identify whether this request is about computing, communication, or storage.",
    "Give a concise description of a mobile edge computing scenario.",
    "Translate to English: 无人机边缘网络需要低时延推理。",
    "List two constraints that affect collaborative inference on UAVs.",
]


def _load_texts(path: str | Path | None, limit: int | None) -> List[str]:
    if path is None:
        texts = list(DEFAULT_TEXTS)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            texts = [line.strip() for line in handle if line.strip()]
    if limit is not None:
        texts = texts[:limit]
    if not texts:
        raise ValueError("No input texts are available for trace extraction.")
    return texts


def _project_vector(values: np.ndarray, dim: int, seed: int) -> List[float]:
    """Return a deterministic compact projection for a high-dimensional vector."""

    flat = values.reshape(-1).astype(np.float32)
    if flat.size <= dim:
        if flat.size < dim:
            flat = np.pad(flat, (0, dim - flat.size))
        norm = np.linalg.norm(flat)
        return (flat / norm).tolist() if norm > 1e-12 else flat.tolist()

    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(flat.size, size=dim, replace=False)
    projected = flat[sample_idx]
    norm = np.linalg.norm(projected)
    if norm > 1e-12:
        projected = projected / norm
    return projected.astype(float).tolist()


def _bytes_from_parameters(parameters: Iterable[object]) -> float:
    total = 0
    for param in parameters:
        total += int(param.numel()) * int(param.element_size())
    return float(total)


def _estimate_flops(parameters: Iterable[object]) -> float:
    """Use 2 multiply-add operations per parameter as a stable FLOP proxy."""

    return float(sum(int(param.numel()) for param in parameters) * 2)


def _find_switch_experts(model: object) -> List[Tuple[int, int, object]]:
    """Find Switch Transformer experts as (layer, local_expert, module)."""

    raw: List[Tuple[Tuple[str, int], int, object]] = []
    for module_name, module in model.named_modules():
        last_part = module_name.split(".")[-1]
        if not last_part.startswith("expert_"):
            continue
        parts = module_name.split(".")
        stack = parts[0] if parts else "model"
        block = -1
        local_expert = -1
        for idx, part in enumerate(parts):
            if part == "block" and idx + 1 < len(parts):
                try:
                    block = int(parts[idx + 1])
                except ValueError:
                    pass
            if part.startswith("expert_"):
                try:
                    local_expert = int(part.split("_", maxsplit=1)[1])
                except ValueError:
                    pass
        params = list(module.parameters(recurse=True))
        if block >= 0 and local_expert >= 0 and params:
            raw.append(((stack, block), local_expert, module))

    stack_order = {"encoder": 0, "decoder": 1}
    raw.sort(key=lambda item: (stack_order.get(item[0][0], 99), item[0][1], item[1]))
    group_to_layer: Dict[Tuple[str, int], int] = {}
    experts: List[Tuple[int, int, object]] = []
    for group, local_expert, module in raw:
        if group not in group_to_layer:
            group_to_layer[group] = len(group_to_layer)
        experts.append((group_to_layer[group], local_expert, module))
    return experts


def _extract_experts(model: object, projection_dim: int) -> List[ExpertTrace]:
    experts = _find_switch_experts(model)
    if not experts:
        raise RuntimeError(
            "Could not locate Switch-style expert modules. "
            "Use a Switch Transformer model or extend _find_switch_experts()."
        )

    traces: List[ExpertTrace] = []
    for expert_id, (layer, local_expert, module) in enumerate(experts):
        params = list(module.parameters(recurse=True))
        vector_chunks = [
            param.detach().float().cpu().numpy().reshape(-1)
            for param in params
            if param.numel() > 0
        ]
        raw_vector = np.concatenate(vector_chunks) if vector_chunks else np.zeros(1)
        traces.append(
            ExpertTrace(
                id=expert_id,
                layer=layer,
                local_expert=local_expert,
                parameter_bytes=_bytes_from_parameters(params),
                flops=_estimate_flops(params),
                latency_s=0.0,
                weight_vector=_project_vector(
                    raw_vector,
                    dim=projection_dim,
                    seed=17_000 + expert_id,
                ),
            )
        )
    return traces


def _measure_expert_latencies(
    model: object,
    expert_traces: List[ExpertTrace],
    hidden_size: int,
    device: str,
    repeats: int,
) -> Dict[int, float]:
    """Measure average standalone expert forward latency when possible."""

    import torch

    expert_modules = _find_switch_experts(model)
    latencies: Dict[int, float] = {}
    if not expert_modules:
        return latencies

    for expert_id, (_layer, _local, module) in enumerate(expert_modules):
        module.eval()
        first_param = next(module.parameters(), None)
        dtype = first_param.dtype if first_param is not None else torch.float32
        x = torch.randn(16, hidden_size, device=device, dtype=dtype)
        with torch.no_grad():
            for _ in range(2):
                _ = module(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(max(repeats, 1)):
                _ = module(x)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        latencies[expert_traces[expert_id].id] = elapsed / max(repeats, 1)
    return latencies


def _router_outputs_from_forward(outputs: object) -> Sequence[object]:
    router = getattr(outputs, "router_logits", None)
    if router is None:
        return []
    if isinstance(router, tuple):
        return router
    return [router]


def _register_router_classifier_hooks(model: object) -> Tuple[List[object], List[object]]:
    """Capture router classifier logits in module execution order."""

    captured: List[object] = []
    handles: List[object] = []

    def hook(_module: object, _inputs: object, output: object) -> None:
        captured.append(output.detach())

    for name, module in model.named_modules():
        if name.endswith(".router.classifier"):
            handles.append(module.register_forward_hook(hook))
    if not handles:
        raise RuntimeError("Could not find router classifier modules to hook.")
    return captured, handles


def _task_from_router_outputs(
    task_id: int,
    text: str,
    token_count: int,
    router_outputs: Sequence[object],
    layer_local_to_global: Dict[Tuple[int, int], int],
    bits_per_token: float,
    mid_bits_per_token: float,
) -> TaskTrace:
    import torch

    layer_experts: List[int] = []
    gating_scores: Dict[int, float] = {}
    for layer_idx, router_tensor in enumerate(router_outputs):
        if isinstance(router_tensor, tuple):
            router_tensor = router_tensor[0]
        if router_tensor is None:
            continue
        logits = router_tensor.detach()
        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])
        elif logits.ndim == 2:
            logits = logits.reshape(-1, logits.shape[-1])
        else:
            continue

        chosen = torch.argmax(logits, dim=-1).cpu().numpy().astype(int)
        if chosen.size == 0:
            continue
        local_expert = int(np.bincount(chosen).argmax())
        global_expert = layer_local_to_global.get((layer_idx, local_expert))
        if global_expert is None:
            continue
        layer_experts.append(global_expert)
        for local_id in chosen.tolist():
            mapped = layer_local_to_global.get((layer_idx, int(local_id)))
            if mapped is not None:
                gating_scores[mapped] = gating_scores.get(mapped, 0.0) + 1.0

    if not layer_experts:
        raise RuntimeError(
            "No router expert choices were extracted. "
            "Check that the model returns router logits."
        )

    S_in = max(token_count * bits_per_token, bits_per_token)
    S_mid = [max(token_count * mid_bits_per_token, mid_bits_per_token)] * len(layer_experts)
    S_out = max(math.ceil(token_count * 0.2), 1) * bits_per_token
    return TaskTrace(
        id=task_id,
        text=text,
        token_count=token_count,
        S_in=float(S_in),
        S_out=float(S_out),
        S_mid=[float(value) for value in S_mid],
        expert_sequence=layer_experts,
        gating_scores=gating_scores,
        layer_experts=layer_experts,
    )


def extract_switch_trace(
    model_name: str,
    output_dir: str | Path,
    texts: Sequence[str],
    device: str = "cpu",
    projection_dim: int = 128,
    latency_repeats: int = 10,
    bits_per_token: float = 16_384.0,
    mid_bits_per_token: float = 8_192.0,
) -> TraceBundle:
    """Extract a trace bundle from a Switch Transformer model."""

    import torch
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, "add_router_probs"):
        config.add_router_probs = True
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    model.to(device)
    model.eval()

    experts = _extract_experts(model, projection_dim=projection_dim)
    hidden_size = int(getattr(config, "d_model", getattr(config, "hidden_size", 768)))
    latencies = _measure_expert_latencies(
        model,
        experts,
        hidden_size=hidden_size,
        device=device,
        repeats=latency_repeats,
    )
    for expert in experts:
        latency = latencies.get(expert.id)
        if latency is not None and latency > 0:
            expert.latency_s = latency

    layer_local_to_global = {
        (expert.layer, expert.local_expert): expert.id
        for expert in experts
    }

    tasks: List[TaskTrace] = []
    with torch.no_grad():
        for task_id, text in enumerate(texts):
            encoded = tokenizer(text, return_tensors="pt", truncation=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            token_count = int(encoded["input_ids"].shape[-1])
            decoder_start = getattr(config, "decoder_start_token_id", None)
            if decoder_start is None:
                decoder_start = tokenizer.pad_token_id
            decoder_input_ids = torch.tensor([[decoder_start]], device=device)
            captured_router_logits, hook_handles = _register_router_classifier_hooks(model)
            try:
                outputs = model(
                    **encoded,
                    decoder_input_ids=decoder_input_ids,
                    return_dict=True,
                )
            finally:
                for handle in hook_handles:
                    handle.remove()
            router_outputs = captured_router_logits or _router_outputs_from_forward(outputs)
            tasks.append(
                _task_from_router_outputs(
                    task_id=task_id,
                    text=text,
                    token_count=token_count,
                    router_outputs=router_outputs,
                    layer_local_to_global=layer_local_to_global,
                    bits_per_token=bits_per_token,
                    mid_bits_per_token=mid_bits_per_token,
                )
            )

    bundle = TraceBundle(
        metadata=TraceMetadata(
            model_name=model_name,
            device=device,
            num_tasks=len(tasks),
            num_experts=len(experts),
            notes="Real Switch Transformer router trace for UAV MoE simulation.",
        ),
        experts=experts,
        tasks=tasks,
    )
    write_trace_bundle(bundle, output_dir)
    return bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a real MoE trace bundle.")
    parser.add_argument("--model-name", default="google/switch-base-8")
    parser.add_argument("--output-dir", default="data/real_moe/traces/switch_base_8")
    parser.add_argument("--texts", default=None, help="Optional UTF-8 text file, one prompt per line.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--latency-repeats", type=int, default=10)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    texts = _load_texts(args.texts, args.limit)
    bundle = extract_switch_trace(
        model_name=args.model_name,
        output_dir=args.output_dir,
        texts=texts,
        device=args.device,
        projection_dim=args.projection_dim,
        latency_repeats=args.latency_repeats,
    )
    print(
        f"Wrote real MoE trace: {len(bundle.experts)} experts, "
        f"{len(bundle.tasks)} tasks -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
