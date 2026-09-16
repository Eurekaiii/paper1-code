# Real-World Data Layer

This folder separates real/realistic workload inputs from the original
synthetic scenario builders.

## Contents

- `uav/`: realistic UAV-edge compute platform specifications and a reusable
  four-UAV scenario.
- `tasks/`: real public NLP task prompts sampled from AG News and SQuAD.
- `moe/`: real MoE traces extracted from `google/switch-base-8` using the
  public task prompts.

## Current Data Status

- Real MoE expert data: available in
  `moe/switch_base_8_trace/experts.json`.
- Real router activation/task trace: available in
  `moe/switch_base_8_trace/tasks.json`.
- Realistic UAV platform data: available in
  `uav/uav_platform_specs.csv`.
- Simulator-ready UAV scenario: available in
  `uav/uav_edge_scenario_4node.json`.

The next step is not more data collection, but experiment calibration: deciding
how to map platform memory, expert size, communication profile, and task
placement so the evaluation fairly stresses UAV-edge constraints.

