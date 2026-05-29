# Issue #1636 Benchmark Metric Semantics

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1636>

## Decision

Benchmark-facing success now uses one collision contract:

- `success()` delegates to `success_rate()`.
- `success_rate()` requires goal completion before the horizon and zero `collision_count()`.
- `collision_count()` is the sum of wall, other-agent, and pedestrian collisions.
- Pedestrian collisions default to robot-pedestrian footprint overlap using `EpisodeData.robot_radius`
  and `EpisodeData.ped_radius`.

The legacy `collisions()` helper remains pedestrian-only for compatibility, but it now shares the
same default pedestrian predicate as `human_collisions()`. Benchmark reports should use
`compute_all_metrics()["collisions"]` or `collision_count()` for total collisions.

## Radius Provenance

Map-runner episodes already populate `EpisodeData` radii from environment config. The synthetic
benchmark runner now resolves radii from scenario metadata for both planner observations and metric
`EpisodeData`; when metadata is absent, it uses the same defaults for both surfaces
(`robot=0.3m`, `pedestrian=0.35m`).

## Validation Boundary

The regression tests cover:

- wall-collision success divergence (`success()` vs `success_rate()`),
- pedestrian footprint overlap that old center-distance defaults missed,
- positive-clearance near misses that must not become collisions,
- synthetic runner observation and metric radii staying in sync.

Validated commands:

- `uv run ruff check robot_sf/benchmark/metrics.py robot_sf/benchmark/runner.py robot_sf/benchmark/thresholds.py tests/test_metrics.py tests/test_runner_smoke.py`
- `uv run pytest tests/test_metrics.py tests/test_metrics_success_rate.py tests/unit/test_metrics_edge_cases.py tests/test_runner_smoke.py tests/test_threshold_metadata.py -q`

## Follow-Up Boundary

This change does not reinterpret historical benchmark artifacts. Existing tracked evidence keeps its
recorded threshold profile and should be treated as historical output unless regenerated through a
separate benchmark-evidence issue.
