# Real Text Task Workload

This folder stores public text workloads used to drive the real MoE model.

## Files

- `prompts.jsonl`: structured task records with source dataset, task type, and
  prompt text.
- `prompts.txt`: one prompt per line, suitable for
  `python -m src.real_moe.trace_extractor --texts ...`.
- `task_manifest.json`: source datasets and sample counts.

## Current Sources

- AG News via `fancyzhx/ag_news` on Hugging Face: news classification style
  inputs.
- SQuAD via `rajpurkar/squad` on Hugging Face: question answering inputs.

These are real public NLP workload samples.  They are not UAV-specific by
themselves; the UAV relevance comes from using them as text inference requests
served by the UAV-edge MoE deployment simulator.

