# AeroMoDE Ablation Summary

| Scenario | Variant | Total delay (ms) | Delta vs full | Substitutions | Deployments | Feasible |
|---|---:|---:|---:|---:|---:|---:|
| synthetic_hotspot_v2 | Full AeroMoE | 8464.99 | 0.00% | 109 | 6 | True |
| synthetic_hotspot_v2 | w/o Similarity | 17231.35 | 103.56% | 0 | 4 | True |
| synthetic_hotspot_v2 | w/o Communication | 8464.99 | 0.00% | 109 | 6 | True |
| synthetic_hotspot_v2 | w/o Redundancy | 8503.45 | 0.45% | 107 | 6 | True |
| real_moe_uav | Full AeroMoE | 3810.90 | 0.00% | 384 | 49 | True |
| real_moe_uav | w/o Similarity | 10156.35 | 166.51% | 0 | 35 | True |
| real_moe_uav | w/o Communication | 3794.23 | -0.44% | 373 | 49 | True |
| real_moe_uav | w/o Redundancy | 4075.84 | 6.95% | 308 | 49 | True |
