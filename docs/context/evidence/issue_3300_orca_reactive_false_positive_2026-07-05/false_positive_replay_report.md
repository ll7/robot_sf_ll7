# Issue #3300 False-Positive Actor-Injection Replay

## Claim Boundary
CPU-local observation-quality replay smoke for false-positive actor injection. Diagnostic only; not a full benchmark campaign, hardware sensor model, or paper-facing claim.

## Classification
- Label: `observed`
- Reason: false-positive injection changed at least one predeclared replay outcome
- Replay mode: `executable`

## Pairing
- Paired rows: `6`
- Unmatched nominal rows: `0`
- Unmatched perturbed rows: `0`

## Injection Summary
- Pedestrians added: `101`
- Steps with noise: `101`
- Perturbation profiles: `issue_3300_false_positive_orca_close_v1`
- Perturbation hashes: `1c77dd478a6b`

## Episode Deltas

| planner | scenario | seed | pedestrians added | changed fields |
|---|---|---:|---:|---|
| orca | francis2023_intersection_wait | 0 | 20 | avg_speed, clearing_distance_min, curvature_mean, jerk_mean, min_clearance, min_distance, socnavbench_path_irregularity |
| orca | francis2023_intersection_wait | 3300 | 20 | avg_speed, clearing_distance_min, curvature_mean, jerk_mean, min_clearance, min_distance, socnavbench_path_irregularity |
| orca | issue_3233_near_field_observation_noise | 0 | 12 | avg_speed, clearing_distance_min, curvature_mean, jerk_mean, min_clearance, min_distance, near_misses, socnavbench_path_irregularity |
| orca | issue_3233_near_field_observation_noise | 3300 | 9 | avg_speed, clearing_distance_min, curvature_mean, jerk_mean, min_clearance, min_distance, near_misses, socnavbench_path_irregularity |
| orca | single_ped_crossing_orthogonal | 0 | 20 | avg_speed, clearing_distance_min, curvature_mean, jerk_mean, min_clearance, min_distance, socnavbench_path_irregularity |
| orca | single_ped_crossing_orthogonal | 3300 | 20 | avg_speed, clearing_distance_min, curvature_mean, jerk_mean, min_clearance, min_distance, socnavbench_path_irregularity |

## Caveats
- CPU replay smoke only.
- False-positive effects are reported separately from other observation noise.
- No full benchmark campaign, Slurm/GPU submission, or paper-facing claim.
