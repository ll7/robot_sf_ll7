# Benchmark Post-Fix Report (2026-02-20)

## Scope

This report documents the benchmark follow-up after fixing predictive planner
checkpoint compatibility and tightening fail-fast campaign behavior.

## Promoted Baseline Run

- campaign id:
  - `camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848`
- campaign root:
  - `output/benchmarks/camera_ready/camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848`
- summary:
  - `runs=8`, `successful_runs=8`, `episodes=1080`, runtime `386.54s`

## Quality Validation (Prediction Planner)

Base campaign (pre-fix):

- `camera_ready_all_planners_prediction_first_prediction_first_stop_on_failure_20260220_195738`
- `prediction_planner`: `status=partial-failure`, `episodes=0`, `failed_jobs=135`

Candidate campaign (post-fix):

- `camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848`
- `prediction_planner`: `status=ok`, `episodes=135`, `failed_jobs=0`
- key metrics:
  - `success_mean=0.9778`
  - `collisions_mean=0.0000`
  - `near_misses_mean=0.0296`
  - `snqi_mean=-1.8677`

Comparison artifacts:

- `output/benchmarks/camera_ready/camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848/reports/campaign_comparison.json`
- `output/benchmarks/camera_ready/camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848/reports/campaign_comparison.md`

## Runtime Hotspot Diagnostics

Updated analysis now includes planner and scenario runtime hotspots from episode
`wall_time_sec` values.

Artifacts:

- `output/benchmarks/camera_ready/camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848/reports/campaign_analysis.json`
- `output/benchmarks/camera_ready/camera_ready_all_planners_prediction_first_prediction_first_stop_verify_20260220_201848/reports/campaign_analysis.md`

Observed top runtime hotspot in this run:

- `socnav_bench` remains the dominant runtime contributor.

## Operational Policy Changes

1. Fail fast on degraded failures:
   - `stop_on_failure` now aborts campaigns on both `failed` and `partial-failure`.
2. Canonical config hardening:
   - all camera-ready benchmark presets now set `stop_on_failure: true`.
   - all-planners presets run `prediction_planner` first for early compatibility checks.
3. Log-noise reduction:
   - predictive checkpoint compatibility with extra keys (for example `value_head.*`)
     is handled without repeated user-facing warnings in worker output.

## Files Changed for This Follow-Up

- `robot_sf/benchmark/camera_ready_campaign.py`
- `robot_sf/planner/predictive_model.py`
- `scripts/tools/analyze_camera_ready_campaign.py`
- `scripts/tools/compare_camera_ready_campaigns.py`
- `configs/benchmarks/camera_ready_baseline_safe.yaml`
- `configs/benchmarks/camera_ready_smoke_all_planners.yaml`
- `configs/benchmarks/camera_ready_all_planners.yaml`
- `configs/benchmarks/camera_ready_all_planners_strict_socnav.yaml`
- `docs/benchmark_camera_ready.md`
- `tests/test_predictive_model.py`
- `tests/benchmark/test_camera_ready_campaign.py`
- `tests/tools/test_compare_camera_ready_campaigns.py`

