# H600 SNQI Recalibration Bundle

- Evidence status: `diagnostic-only`.
- Status: `ok`.
- Canonical camera-ready SNQI weights and baselines are not overwritten.
- H500 comparison source status: `available_s10_h500_not_s20`.

| planner_key | original_h600_rank | recalibrated_h600_rank | h500_rank | rank_delta | stability | status |
| --- | --- | --- | --- | --- | --- | --- |
| goal | 4 | 4 | 10 | -6 | rank_flip | ok |
| hybrid_rule_v3_fast_progress | None | None | 5 | None | not_evaluable | not_evaluable |
| hybrid_rule_v3_fast_progress_static_escape | None | None | 4 | None | not_evaluable | not_evaluable |
| hybrid_rule_v3_fast_progress_static_escape_continuous | None | None | 1 | None | not_evaluable | not_evaluable |
| orca | 3 | 3 | 9 | -6 | rank_flip | ok |
| ppo | 1 | 1 | 8 | -7 | rank_flip | ok |
| prediction_planner | 7 | 7 | 7 | 0 | stable | ok |
| sacadrl | 2 | 2 | 11 | -9 | rank_flip | ok |
| scenario_adaptive_hybrid_orca_v1 | None | None | 2 | None | not_evaluable | not_evaluable |
| scenario_adaptive_hybrid_orca_v2_collision_guard | None | None | 3 | None | not_evaluable | not_evaluable |
| social_force | 6 | 6 | 12 | -6 | rank_flip | ok |
| socnav_sampling | 5 | 5 | 6 | -1 | stable | ok |
