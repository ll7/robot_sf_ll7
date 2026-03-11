# Issue 503 Execution Notes: Experimental Pedestrian-Impact Metrics

Issue: <https://github.com/ll7/robot_sf_ll7/issues/503>

## What Was Implemented

`robot_sf/benchmark/metrics.py` now supports an optional experimental metric group behind
`compute_all_metrics(..., experimental_ped_impact=True)`.

Added outputs (`ped_impact_*`):
- configuration echo: `ped_impact_radius_m`, `ped_impact_window_steps`
- coverage counters: `ped_impact_ped_count`, `ped_impact_near_samples`,
  `ped_impact_far_samples`, `ped_impact_near_sample_frac`
- acceleration disturbance: near/far means and per-ped delta aggregates
- heading turn-rate disturbance: near/far means and per-ped delta aggregates
- validity counters: `ped_impact_accel_delta_valid`, `ped_impact_turn_rate_delta_valid`

## Semantics

- Near/far split: robot-pedestrian distance `<= radius` is near, `> radius` is far.
- Time-window: acceleration and turn-rate signals are smoothed with a trailing rolling mean over
  `window_steps`.
- Aggregation: per-ped `(near_mean - far_mean)` is computed first; then mean/median across valid
  pedestrians is reported.

This per-ped aggregation reduces density bias compared to direct pooled-sample aggregation.

## Robustness Choices

- Low-speed heading noise is suppressed by marking turn-rate samples invalid when consecutive
  pedestrian speeds are too small.
- Invalid samples are represented as `NaN` and filtered by NaN-aware aggregations.
- Degenerate cases (no pedestrians or missing near/far support) are represented via validity
  counters rather than forcing synthetic values.

## Tests Added

In `tests/test_metrics.py`:
- opt-in behavior test (`ped_impact_*` keys absent by default)
- crafted near-vs-far disturbance scenario test (positive delta checks)
- empty-crowd stability test (zero counters, validity false)

## Follow-Ups (Deferred)

- CLI exposure for map/benchmark runners (kept out-of-scope in this patch to avoid touching
  low-coverage execution plumbing files).
- Threshold profiling for `radius_m` and `window_steps` across canonical scenario suites.
