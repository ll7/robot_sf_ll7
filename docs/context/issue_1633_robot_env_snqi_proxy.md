# Issue #1633 RobotEnv SNQI Proxy Extraction

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1633>

## Goal

Decompose one cohesive `RobotEnv` runtime concern without changing Gymnasium behavior. The first
extraction target is step-level SNQI proxy metadata because it is stateful, simulator-backed, and
testable without constructing the full environment.

## Refactor Plan

`RobotEnv` should remain the public environment facade. Runtime concerns can move behind small
collaborators while preserving thin compatibility methods where tests or downstream code already
reach into private helpers.

First extraction implemented here:

- `robot_sf/gym_env/snqi_proxy.py`
  - owns `StepSNQIProxy` and `StepSNQIProxyState`,
  - resolves SNQI thresholds lazily,
  - extracts robot pose and pedestrian/force rows from simulator-shaped payloads,
  - computes `near_misses`, `force_exceed_events`, `comfort_exposure`, and running `jerk_mean`.
- `RobotEnv` now primes and calls `StepSNQIProxy` during reset/step reward metadata assembly.

Candidate follow-up collaborators identified during the scan:

- `OccupancyGridRuntime`: grid enablement, static obstacle cache, grid generation, and observation
  injection currently spread across setup, reset, step, and obstacle normalization helpers.
- `ObservationRuntime`: SocNav/default observation setup, flattening, grid-space insertion, and
  asymmetric critic observation augmentation.
- `TelemetryVisualizationRuntime`: telemetry session creation/emission plus visualizable state
  capture used by rendering and recording.

## Acceptance Tests

The extraction is pinned by:

- `tests/gym_env/test_snqi_step_proxy.py` for the direct collaborator contract.
- `tests/gym_env/test_robot_env_snqi_step_metadata.py` for RobotEnv metadata plumbing.
- `uv run pytest tests -k "robot_env or environment_factory" -q` for the requested environment
  factory slice.

## Boundaries

This refactor does not change benchmark metrics, reward semantics, simulator stepping, observation
space shape, or Gymnasium return values. It only moves SNQI proxy state and computation out of
`robot_env.py`.
