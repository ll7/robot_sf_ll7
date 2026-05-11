# Issue 1065 Route-Clearance Warning Evidence

Date: 2026-05-09

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1065>

Follow-up repair/certification issue: <https://github.com/ll7/robot_sf_ll7/issues/1105>

## Source Commands

Current fixed paper-matrix preflight:

```bash
rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1.yaml \
  --mode preflight \
  --label issue1065_paper \
  --log-level WARNING
```

Current h500 scenario-horizon preflight:

```bash
rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --mode preflight \
  --label issue1065_h500 \
  --log-level WARNING
```

Generated local outputs were left under `output/benchmarks/camera_ready/` and are not durable
dependencies. This directory keeps the compact, reviewable warning table used by the issue #1065
context note.

## Result

Both preflights reported the same `route_clearance_warning_count`: `18`.

The retained table is:

- [route_clearance_warning_classification.csv](route_clearance_warning_classification.csv)

It records each scenario, map, warning margin, warning scope, issue #1065 classification, and
planner-attribution boundary.

## Interpretation

Negative-clearance rows are benchmark-blocking for planner-failure attribution until map/route
geometry is repaired or the scenario is explicitly certified as stress-only/excluded.

Zero-margin and low-positive-margin rows are not automatically invalid, but planner-failure claims
must carry the geometry caveat and should not be used for positive static-clearance claims without
additional trace or certification evidence.
