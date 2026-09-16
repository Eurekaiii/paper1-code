# Real MoE Trace-Driven Experiment

This folder adds a real-model workload path without changing the existing
synthetic AeroMoDE pipeline.

## Files

- `trace_schema.py`: shared JSON dataclasses for real MoE traces.
- `trace_extractor.py`: loads an open-source MoE model, runs text prompts, and
  writes real expert/router traces.
- `scenario_builder.py`: converts trace JSON into the existing `UAV`,
  `Expert`, and `Task` objects.
- `experiment.py`: runs Proposed and baseline methods on a trace-driven
  scenario.
- `export_results.py`: writes CSV outputs under `results/real_moe`.

## Typical workflow

1. Extract a trace from a real MoE model:

   ```powershell
   python -m src.real_moe.trace_extractor --model-name google/switch-base-8 --limit 10
   ```

2. Run the AeroMoDE experiment with that trace:

   ```powershell
   python -m src.real_moe.experiment --trace-dir data/real_moe/traces/switch_base_8
   ```

The first command may require network access and the `torch` and `transformers`
packages.  The second command only needs the saved trace files.

