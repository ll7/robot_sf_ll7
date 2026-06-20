# Issue #3201 Observation-Noise Live Smoke

## Claim Boundary

Diagnostic same-seed local smoke only. This retests the issue #2749 observation-noise null with a pedestrian-dominated candidate surface; it is not benchmark-strength or hardware-calibrated sensor evidence.

## Inputs

- Clean trace: `worktree-local ignored artifact summarized in this report (benchmarks/issue_3201_observation_noise_live_smoke/clean_step/trace.json)`
- Perturbed trace: `worktree-local ignored artifact summarized in this report (benchmarks/issue_3201_observation_noise_live_smoke/perturbed_step/trace.json)`
- Same scenario: `True`
- Same seed: `True`

## Classification

- Label: `observation_only_scenario_too_weak`
- Rationale: Perturbation changed planner-input pedestrian observations, but commands and progress/risk summaries were identical. The closest live robot-pedestrian distance was 7.00 m, above the 2 m near-field target.

## Command Summary

- Sequence changed: `False`
- Clean first/last: `[1.2, 0.30000000000000004]` / `[2.0, 0.0]`
- Perturbed first/last: `[1.2, 0.30000000000000004]` / `[2.0, 0.0]`

## Observation Summary

- Changed: `True`
- Clean: `{'missed_actor_observations_total': 0, 'occluded_actor_observations_total': 0, 'min_observed_actor_count': 17, 'max_observed_actor_count': 17, 'noise_profiles': ['none'], 'evidence_classes': ['ideal_state']}`
- Perturbed: `{'missed_actor_observations_total': 179, 'occluded_actor_observations_total': 0, 'min_observed_actor_count': 5, 'max_observed_actor_count': 12, 'noise_profiles': ['bounded_gaussian'], 'evidence_classes': ['perception_limited']}`

## Progress Deltas

| Field | Clean | Perturbed | Delta | Changed |
|---|---:|---:|---:|---|
| `steps_observed` | `20` | `20` | `0.0` | `False` |
| `initial_goal_distance` | `3.5209472023792885` | `3.5209472023792885` | `0.0` | `False` |
| `final_goal_distance` | `18.630464317750842` | `18.630464317750842` | `0.0` | `False` |
| `best_goal_distance` | `2.0466035266377416` | `2.0466035266377416` | `0.0` | `False` |
| `net_goal_progress` | `-15.109517115371554` | `-15.109517115371554` | `0.0` | `False` |
| `best_goal_progress` | `1.474343675741547` | `1.474343675741547` | `0.0` | `False` |
| `progress_step_count` | `19` | `19` | `0.0` | `False` |
| `regression_step_count` | `1` | `1` | `0.0` | `False` |
| `stagnant_step_count` | `0` | `0` | `0.0` | `False` |
| `longest_stagnant_run` | `0` | `0` | `0.0` | `False` |
| `closest_robot_ped_distance` | `6.999430016273784` | `6.999430016273784` | `0.0` | `False` |
| `closest_robot_ped_step` | `19` | `19` | `0.0` | `False` |
| `collision_flag_counts` | `{'pedestrian': 0, 'obstacle': 0, 'robot': 0}` | `{'pedestrian': 0, 'obstacle': 0, 'robot': 0}` | `None` | `False` |

## Caveats

- One scenario, one seed, one horizon; diagnostic smoke only.
- The live scenario candidate remained outside the intended 2 m near-field threshold.
- Uses non-calibrated observation perturbations and makes no hardware sensor claim.