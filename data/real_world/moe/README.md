# Real MoE Trace Data

The current real MoE trace bundle is generated from `google/switch-base-8` and
stored under:

```text
data/real_moe/traces/switch_base_8/
```

Use `src/real_moe/trace_extractor.py` to regenerate it with a larger real text
workload:

```powershell
python -m src.real_moe.trace_extractor --model-name google/switch-base-8 --texts data/real_world/tasks/prompts.txt --device cpu --latency-repeats 1
```

