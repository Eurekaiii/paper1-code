# UAV Platform Data

This folder records realistic UAV-edge compute platform inputs for the
trace-driven experiments.  The rows are not physical UAV airframes; they are
payload compute modules commonly used in robotics and UAV/edge-AI prototypes.

## Files

- `uav_platform_specs.csv`: edge compute platform specifications.
- `uav_network_profiles.json`: repeatable UAV communication scenario profiles
  used to parameterize the simulator.

## Source Notes

- NVIDIA reports Jetson Orin Nano Super at 67 INT8 TOPS, 8 GB LPDDR5 memory,
  102 GB/s bandwidth, and 7W-25W power.
- NVIDIA reports Jetson Xavier NX with 8 GB LPDDR4x memory and 10W/15W modes
  reaching 14/21 INT8 TOPS.
- NVIDIA reports Jetson Nano Developer Kit with 4 GB LPDDR4 memory and
  25.6 GB/s bandwidth; the common 472 GFLOPS figure is stored as a low-end
  TOPS-equivalent only for relative simulation ranking.
- NVIDIA reports Orin NX/AGX Orin ranges for higher-end edge robotics modules.

The simulator currently maps `ai_perf_tops_int8` to an equivalent compute
capacity.  This is a proxy for comparative edge inference capability, not a
claim that INT8 TOPS equals FP FLOP/s for every workload.

