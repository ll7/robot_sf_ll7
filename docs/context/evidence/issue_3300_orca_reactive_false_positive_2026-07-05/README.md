# Issue #3300 — ORCA Reactive False-Positive Actor-Injection Replay Evidence

## Summary

CPU-local executable replay with ORCA planner and close-range false-positive actor injection.
Classification: **observed** — false-positive injection changed predeclared replay outcomes.

## What Changed vs Prior Slices

| Prior slice | Classification | Planner | Why |
|---|---|---|---|
| PR #4390 | blocked_unavailable | goal | Flat-pedestrian buffer full; no actors injected |
| PR #4431 | scenario_too_weak | goal | Goal planner ignores ghost pedestrians |
| PR #4439 | pre-registration only | goal | Matrix configs only; no replay |
| PR #4486 | scenario_too_weak | goal | Stronger matrix still with goal planner |
| **This slice** | **observed** | **orca** | ORCA reacts to nearby ghost pedestrians |

## Key Difference

ORCA (Optimal Reciprocal Collision Avoidance) is a reactive local planner that explicitly
avoids nearby agents. When a false-positive ghost pedestrian is injected within 1.5m,
ORCA adjusts its velocity command, producing measurable behavioral changes:

- `avg_speed` changes (up to -0.24 m/s, 24% reduction in one episode)
- `curvature_mean` changes (path shape altered to avoid ghost)
- `jerk_mean` changes (acceleration profile altered)
- `near_misses` changes (up to -11 in one episode)
- `socnavbench_path_irregularity` changes (path more irregular)

## Configs

- Noise profile: `configs/benchmarks/observation_noise/issue_3300_false_positive_orca_close_v1.yaml`
  - `pedestrian_false_positive_prob: 1.0`
  - `pedestrian_false_positive_radius_m: 1.5` (close range)
  - `pedestrian_false_positive_radius: 0.5` (larger ghost)
- Nominal: `configs/benchmarks/issue_3300_orca_fp_nominal_smoke.yaml`
- Perturbed: `configs/benchmarks/issue_3300_orca_fp_perturbed_smoke.yaml`
- Scenario matrix: `configs/scenarios/sets/issue_3300_false_positive_stronger_structured_matrix.yaml`
- Planner: ORCA with `observation_mode: socnav_state`
- Seeds: [0, 3300]
- Horizon: 20 steps

## Injection Summary

- Total pedestrians added: 101 across 6 episodes
- Steps with noise: 101
- Profile: `issue_3300_false_positive_orca_close_v1`

## Classification Details

- Label: `observed`
- Reason: false-positive injection changed at least one predeclared replay outcome
- Changed fields per episode:
  - `single_ped_crossing_orthogonal` / seed=0: avg_speed, curvature_mean, jerk_mean, min_distance, min_clearance, clearing_distance_min, socnavbench_path_irregularity
  - `single_ped_crossing_orthogonal` / seed=3300: avg_speed, curvature_mean, jerk_mean, min_distance, min_clearance, clearing_distance_min, socnavbench_path_irregularity
  - `issue_3233_near_field_observation_noise` / seed=0: avg_speed, curvature_mean, jerk_mean, min_distance, min_clearance, clearing_distance_min, near_misses, socnavbench_path_irregularity
  - `issue_3233_near_field_observation_noise` / seed=3300: avg_speed, curvature_mean, jerk_mean, min_distance, min_clearance, clearing_distance_min, near_misses, socnavbench_path_irregularity
  - `francis2023_intersection_wait` / seed=0: avg_speed, curvature_mean, jerk_mean, min_distance, min_clearance, clearing_distance_min, socnavbench_path_irregularity
  - `francis2023_intersection_wait` / seed=3300: avg_speed, curvature_mean, jerk_mean, min_distance, min_clearance, clearing_distance_min, socnavbench_path_irregularity

## Caveats

- CPU replay smoke only; not a full benchmark campaign.
- False-positive effects are reported separately from other observation noise.
- No full benchmark campaign, Slurm/GPU submission, or paper-facing claim.
- Metric name fix: corrected predeclared metric names to match actual episode JSONL field names.

## Reproduction

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3300_orca_fp_nominal_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_3300_orca \
  --campaign-id issue_3300_orca_nominal \
  --skip-publication-bundle

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3300_orca_fp_perturbed_smoke.yaml \
  --mode run \
  --output-root output/benchmarks/issue_3300_orca \
  --campaign-id issue_3300_orca_perturbed \
  --skip-publication-bundle
```
