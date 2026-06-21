# Issue #3300 Observation-Noise Live Smoke

## Claim Boundary

Diagnostic same-seed local smoke for issue #3300 false-positive actor-injection. This is executable planner/environment replay evidence for one scenario, seed, planner, and five-step horizon; it is not benchmark-strength, hardware-calibrated sensor, or paper-facing evidence.

## Inputs

- Clean trace: `worktree-local ignored artifact summarized in this report (benchmarks/issue_3300_false_positive_actor_injection/clean_h5/trace.json)`
- Perturbed trace: `worktree-local ignored artifact summarized in this report (benchmarks/issue_3300_false_positive_actor_injection/false_positive_h5/trace.json)`
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
- Perturbed first/last: `[0.0, 0.0]` / `[0.0, 0.0]`

## Observation Summary

- Changed: `True`
- Clean: `{'missed_actor_observations_total': 0, 'occluded_actor_observations_total': 0, 'false_positive_actor_observations_total': 0, 'min_observed_actor_count': 1, 'max_observed_actor_count': 1, 'noise_profiles': ['none'], 'evidence_classes': ['ideal_state']}`
- Perturbed: `{'missed_actor_observations_total': 0, 'occluded_actor_observations_total': 0, 'false_positive_actor_observations_total': 5, 'min_observed_actor_count': 2, 'max_observed_actor_count': 2, 'noise_profiles': ['false_positive_actor_injection'], 'evidence_classes': ['perception_limited']}`

## Progress Deltas

| Field | Clean | Perturbed | Delta | Changed |
|---|---:|---:|---:|---|
| `steps_observed` | `5` | `5` | `0.0` | `False` |
| `initial_goal_distance` | `2.000001` | `2.000001` | `0.0` | `False` |
| `final_goal_distance` | `14.916698691335359` | `2.000001` | `-12.916697691335358` | `True` |
| `best_goal_distance` | `2.000001` | `2.000001` | `0.0` | `False` |
| `net_goal_progress` | `-12.916697691335358` | `0.0` | `12.916697691335358` | `True` |
| `best_goal_progress` | `0.0` | `0.0` | `0.0` | `False` |
| `progress_step_count` | `4` | `0` | `-4.0` | `True` |
| `regression_step_count` | `1` | `0` | `-1.0` | `True` |
| `stagnant_step_count` | `0` | `5` | `5.0` | `True` |
| `longest_stagnant_run` | `0` | `5` | `5.0` | `True` |
| `closest_robot_ped_distance` | `1.7872738570026998` | `1.9121684738868852` | `0.12489461688418535` | `True` |
| `closest_robot_ped_step` | `4` | `0` | `-4.0` | `True` |
| `collision_flag_counts` | `{'pedestrian': 0, 'obstacle': 0, 'robot': 0}` | `{'pedestrian': 0, 'obstacle': 0, 'robot': 0}` | `None` | `False` |

## Caveats

- One scenario, one seed, one horizon; diagnostic smoke only.
- Clean trace closest robot-pedestrian distance: `1.7872738570026998` m (target <= `2.0` m, satisfied: `True`).
- Uses non-calibrated observation perturbations and makes no hardware sensor claim.