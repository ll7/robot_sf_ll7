<!-- AI-GENERATED (robot_sf#4366, 2026-07-04) - NEEDS-REVIEW -->
# Issue 4366 Manuscript-Asserted Number Verification

Schema: `manuscript-asserted-number-verification-report.v1`

Claim boundary: verification aid only. No manuscript edits, no claim changes, no full benchmark campaign, and no Slurm/GPU submission.

Declarations: `configs/validation/issue_4366_manuscript_asserted_numbers.yaml`

Selection assumption: The tracked issue thread names number families, but it does not include the dissertation-side audit table. These ten declarations seed the most likely high-risk implementation-number assertions from those named families. Ambiguous heatmap source records are explicitly marked not_verifiable.

## Summary

- Overall status: `pass`
- Matches: `9`
- Mismatches: `0`
- Not verifiable: `1`
- Blocked: `0`

## Results

| id | status | manuscript locator | expected | actual | source | reason |
| --- | --- | --- | --- | --- | --- | --- |
| snqi_camera_ready_v3_w_success | `match` | audit finding F / SNQI weight vector / camera_ready_v3 / w_success | `0.19045845847432735` | `0.19045845847432735` | `configs/benchmarks/snqi_weights_camera_ready_v3.json#w_success` |  |
| snqi_camera_ready_v3_w_near | `match` | audit finding F / SNQI weight vector / camera_ready_v3 / w_near | `0.30825830332144416` | `0.30825830332144416` | `configs/benchmarks/snqi_weights_camera_ready_v3.json#w_near` |  |
| snqi_canonical_weight_set_id | `match` | audit finding F / SNQI canonical set identifier | `camera_ready_v3` | `camera_ready_v3` | `configs/benchmarks/snqi_weight_sets_camera_ready.yaml#canonical_set` |  |
| scenario_density_tier_labels | `match` | audit finding F / scenario-catalog density tiers | `["low", "medium", "high"]` | `["low", "medium", "high"]` | `configs/scenarios/archetypes/classic_density_tier_index.yaml#density_tiers` |  |
| classic_crossing_high_ped_density | `match` | audit finding F / scenario-catalog density / classic_crossing_high | `0.08` | `0.08` | `configs/scenarios/archetypes/classic_crossing.yaml#scenarios[name=classic_crossing_high].simulation_config.ped_density` |  |
| classic_bottleneck_high_ped_density | `match` | audit finding F / scenario-catalog density / classic_bottleneck_high | `0.0` | `0.0` | `configs/scenarios/archetypes/classic_bottleneck.yaml#scenarios[name=classic_bottleneck_high].simulation_config.ped_density` |  |
| ppo_br06_v3_total_timesteps | `match` | audit finding F / reinforcement learning training budget | `15000000` | `15000000` | `configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml#total_timesteps` |  |
| ppo_br06_v3_grid_resolution_m | `match` | audit finding F / reinforcement learning observation grid resolution | `0.2` | `0.2` | `configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml#env_overrides.grid_config.resolution` |  |
| ppo_br06_v3_max_peds_per_group | `match` | audit finding F / pedestrian-module group-size cap | `3` | `3` | `configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml#env_overrides.sim_config.max_peds_per_group` |  |
| heatmap_per_family_means_source | `not_verifiable` | audit finding F / heatmap per-family means | `not_verifiable` |  |  | The issue names heatmap per-family means but does not identify one canonical source-of-record table or locator. This declaration records the gap instead of guessing among generated evidence packets. |

<!-- /AI-GENERATED -->
