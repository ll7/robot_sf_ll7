# Issue #3201 Observation-Noise Live Smoke

## Claim Boundary

Diagnostic same-seed local smoke only. This retests the issue #2749 observation-noise null with a pedestrian-dominated candidate surface; it is not benchmark-strength or hardware-calibrated sensor evidence.

## Inputs

- Clean trace: `worktree-local ignored artifact summarized in this report (validation/issue-3233-near-field/clean_step/trace.json)`
- Perturbed trace: `worktree-local ignored artifact summarized in this report (validation/issue-3233-near-field/perturbed_step/trace.json)`
- Same scenario: `True`
- Same seed: `True`

## Classification

- Label: `non_null_behavior_delta`
- Rationale: Same-seed clean and perturbed live traces differ in selected commands or progress/risk summary fields.

## Near-Field Target

- Clean trace closest robot-pedestrian distance: `1.4515750296008105` m (target <= `2.0` m, satisfied: `True`).

## Command Summary

- Sequence changed: `True`
- Clean first/last: `[0.3, 0.0]` / `[0.6, 0.0]`
- Perturbed first/last: `[1.2, 0.4431303704642192]` / `[0.0, 0.0]`

## Observation Summary

- Changed: `True`
- Clean: `{'missed_actor_observations_total': 0, 'occluded_actor_observations_total': 0, 'min_observed_actor_count': 1, 'max_observed_actor_count': 1, 'noise_profiles': ['none'], 'evidence_classes': ['ideal_state']}`
- Perturbed: `{'missed_actor_observations_total': 3, 'occluded_actor_observations_total': 0, 'min_observed_actor_count': 0, 'max_observed_actor_count': 1, 'noise_profiles': ['bounded_gaussian'], 'evidence_classes': ['perception_limited']}`

## Progress Deltas

| Field | Clean | Perturbed | Delta | Changed |
|---|---:|---:|---:|---|
| `steps_observed` | `50` | `6` | `-44.0` | `True` |
| `initial_goal_distance` | `2.000001` | `2.000001` | `0.0` | `False` |
| `final_goal_distance` | `12.468645089252165` | `14.480990556613484` | `2.0123454673613193` | `True` |
| `best_goal_distance` | `2.000001` | `2.000001` | `0.0` | `False` |
| `net_goal_progress` | `-10.468644089252166` | `-12.480989556613483` | `-2.0123454673613175` | `True` |
| `best_goal_progress` | `0.0` | `0.0` | `0.0` | `False` |
| `progress_step_count` | `49` | `5` | `-44.0` | `True` |
| `regression_step_count` | `1` | `1` | `0.0` | `False` |
| `stagnant_step_count` | `0` | `0` | `0.0` | `False` |
| `longest_stagnant_run` | `0` | `0` | `0.0` | `False` |
| `closest_robot_ped_distance` | `1.4515750296008105` | `1.395304852291646` | `-0.056270177309164504` | `True` |
| `closest_robot_ped_step` | `25` | `5` | `-20.0` | `True` |
| `collision_flag_counts` | `{'pedestrian': 0, 'obstacle': 0, 'robot': 0}` | `{'pedestrian': 1, 'obstacle': 0, 'robot': 0}` | `None` | `True` |

## Caveats

- One scenario, one seed, one horizon; diagnostic smoke only.
- Clean trace closest robot-pedestrian distance: `1.4515750296008105` m (target <= `2.0` m, satisfied: `True`).
- Uses non-calibrated observation perturbations and makes no hardware sensor claim.