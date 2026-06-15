# Forecast Transferability Stress Matrix

- Report id: issue_2866_transferability_matrix_rows
- Decision: stop
- Claim status: blocked
- Matrix rows: 5
- Limitation rows: 2

| dimension | coverage_status | observed_values | unavailable_reports |
| --- | --- | --- | ---: |
| observation_tier | full | deployable_tracked, oracle_full_state | 0 |
| observation_noise | full | none, tracked_noise_low, tracked_noise_unknown | 0 |
| latency | full | 0_steps, 1_step, unknown | 0 |
| dropout | full | none, synthetic_dropout_0.1, unknown | 0 |
| occlusion | full | clear, occluded_emergence, unknown | 0 |
| scenario_family | partial | corridor_fixture, motion_rich_occluded_emergence | 1 |
| map_family | partial | corridor_fixture, motion_rich_occluded_emergence | 1 |
| density | full | low, medium, unknown | 0 |
| pedestrian_model_family | full | fast_actor_fixture, social_force, unknown | 0 |
| actor_type | partial | bicycle, pedestrian | 1 |
| semantic_metadata_present | full | absent, present, unknown | 0 |

| metric | horizon_s | observation_tier | actor_type | denominator | status | value | evidence |
| --- | ---: | --- | --- | ---: | --- | ---: | --- |
| mean_ade | 0.5 | oracle_full_state | pedestrian | 20 | ok | 0.18 | diagnostic-only |
| mean_ade | 1 | oracle_full_state | pedestrian | 20 | ok | 0.34 | diagnostic-only |
| mean_ade | 0.5 | deployable_tracked | bicycle | 8 | ok | 0.42 | benchmark-eligible |
| mean_ade | 1 | deployable_tracked | bicycle | 8 | ok | 0.91 | benchmark-eligible |
| mean_ade | 1 | deployable_tracked | None | 0 | unavailable | NA | blocked |

## Limitations

- scenario_family: dimension metadata unavailable in ForecastMetrics.v1 report (cv-deployable-unavailable-cell-fixture, missing_metadata_fixture, deployable_tracked)
- map_family: dimension metadata unavailable in ForecastMetrics.v1 report (cv-deployable-unavailable-cell-fixture, missing_metadata_fixture, deployable_tracked)

Forecast transferability stress rows are benchmark-eligible only when each cell has all required dimensions, non-empty deployable metric denominators, and non-oracle observation tiers.
