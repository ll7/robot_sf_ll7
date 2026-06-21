# Issue #3300 Observation-Noise Live Smoke

## References

- Related issue: [#3300](https://github.com/ll7/robot_sf_ll7/issues/3300)
- Related PR: [#3333](https://github.com/ll7/robot_sf_ll7/pull/3333)
- Evidence index: [docs/context/evidence/README.md](../README.md)
- Bundle README: [docs/context/evidence/issue_3300_false_positive_actor_injection/README.md](README.md)
- Companion report: [docs/context/evidence/issue_3300_false_positive_actor_injection/false_positive_vs_clean.md](false_positive_vs_clean.md)
- This report summary: [docs/context/evidence/issue_3300_false_positive_actor_injection/missed_detection_vs_clean_summary.json](missed_detection_vs_clean_summary.json)

## Claim Boundary

Diagnostic same-seed local smoke for issue #3300 false-positive actor-injection. This is executable planner/environment replay evidence for one scenario, seed, planner, and five-step horizon; it is not benchmark-strength, hardware-calibrated sensor, or paper-facing evidence.

## Inputs

- Clean trace: `worktree-local ignored artifact summarized in this report (benchmarks/issue_3300_false_positive_actor_injection/clean_h5/trace.json)`
- Perturbed trace: `worktree-local ignored artifact summarized in this report (benchmarks/issue_3300_false_positive_actor_injection/missed_detection_h5/trace.json)`
- Same scenario: `True`
- Same seed: `True`

## Classification

- Label: `non_null_behavior_delta`
- Rationale: Same-seed clean and perturbed live traces differ in selected commands or progress/risk summary fields.

## Near-Field Target

- Clean trace closest robot-pedestrian distance: `1.7872738570026998` m (target <= `2.0` m, satisfied: `True`).

## Command Summary

- Sequence changed: `True`
- Clean first/last: `[0.3, 0.0]` / `[0.49999999999999994, -0.30000000000000004]`
- Perturbed first/last: `[1.2, 0.4431303704642192]` / `[2.0, 0.0]`

## Observation Summary

- Changed: `True`
- Clean: `{'missed_actor_observations_total': 0, 'occluded_actor_observations_total': 0, 'false_positive_actor_observations_total': 0, 'min_observed_actor_count': 1, 'max_observed_actor_count': 1, 'noise_profiles': ['none'], 'evidence_classes': ['ideal_state']}`
- Perturbed: `{'missed_actor_observations_total': 4, 'occluded_actor_observations_total': 0, 'false_positive_actor_observations_total': 0, 'min_observed_actor_count': 0, 'max_observed_actor_count': 0, 'noise_profiles': ['missed_detection'], 'evidence_classes': ['perception_limited']}`

## Progress Deltas

| Field | Clean | Perturbed | Delta | Changed |
|---|---:|---:|---:|---|
| `steps_observed` | `5` | `4` | `-1.0` | `True` |
| `initial_goal_distance` | `2.000001` | `2.000001` | `0.0` | `False` |
| `final_goal_distance` | `14.916698691335359` | `14.500850439914199` | `-0.4158482514211599` | `True` |
| `best_goal_distance` | `2.000001` | `2.000001` | `0.0` | `False` |
| `net_goal_progress` | `-12.916697691335358` | `-12.500849439914198` | `0.4158482514211599` | `True` |
| `best_goal_progress` | `0.0` | `0.0` | `0.0` | `False` |
| `progress_step_count` | `4` | `3` | `-1.0` | `True` |
| `regression_step_count` | `1` | `1` | `0.0` | `False` |
| `stagnant_step_count` | `0` | `0` | `0.0` | `False` |
| `longest_stagnant_run` | `0` | `0` | `0.0` | `False` |
| `closest_robot_ped_distance` | `1.7872738570026998` | `1.382334554845844` | `-0.4049393021568557` | `True` |
| `closest_robot_ped_step` | `4` | `3` | `-1.0` | `True` |
| `collision_flag_counts` | `{'pedestrian': 0, 'obstacle': 0, 'robot': 0}` | `{'pedestrian': 1, 'obstacle': 0, 'robot': 0}` | `None` | `True` |

## Planner Execution Mode

- Clean planner mode: `command_adapter`
- Perturbed planner mode: `command_adapter`
- Observed modes source: `hybrid_rule_v0_minimal` uses `HybridRuleLocalPlannerAdapter` through `scripts/validation/run_policy_search_step_diagnostics.py`, which records per-step adapter execution as `command_adapter`.
- Fallback/degraded caveat status: `not_reported`; the tracked summary does not report fallback or degraded planner diagnostics, and this row remains diagnostic-only evidence rather than benchmark-strength success evidence.

## Caveats

- One scenario, one seed, one horizon; diagnostic smoke only.
- Planner execution is reported as `command_adapter`; no fallback/degraded planner success claim is made from this diagnostic smoke.
- Clean trace closest robot-pedestrian distance: `1.7872738570026998` m (target <= `2.0` m, satisfied: `True`).
- Uses non-calibrated observation perturbations and makes no hardware sensor claim.
