# Issue #4141 Pedestrian Uncertainty Envelope

Status: bounded first-slice implementation.

## What landed

- `robot_sf/nav/uncertainty_envelope.py` — new predictor-agnostic module:
  - `PedestrianUncertaintyEnvelope` (frozen dataclass: `position`, `base_radius`,
    `spatial_inflation`; `effective_radius(horizon_step)`).
  - `linear_inflation_policy(alpha, dt)` — default policy, `inflation(0) == 0`.
  - `envelope_from_position(...)` convenience factory.
  - `effective_pedestrian_radius(...)` — planner-agnostic radius substitution helper.
  - `envelope_diagnostics(...)` — provenance payload with schema version and claim boundary.
  - `ConformalInflationPolicy` — stub protocol documenting the future conformal seam (#4138).
- Prediction-MPC integration: `PredictionMPCConfig` gains opt-in
  `pedestrian_uncertainty_envelope_enabled` / `pedestrian_uncertainty_alpha_mps` (default off,
  parsed by `build_prediction_mpc_config`, validated non-negative). The hard per-horizon-step
  pedestrian clearance constraint substitutes `effective_pedestrian_radius(...)` for the fixed
  radius when a horizon step is available; `diagnostics()` records the envelope provenance.
- Example config `configs/algos/prediction_mpc_cv_uncertainty_envelope.yaml`.
- Tests: `tests/nav/test_uncertainty_envelope.py` (inflation geometry, envelope validation, helper,
  diagnostics, stub protocol) and prediction-MPC regression/inflation tests in
  `tests/planner/test_prediction_mpc.py`.

## Boundary

- Linear scalar-radius inflation only.
- No conformal calibration.
- No distance-field representation.
- No pedestrian dynamics changes.
- No simulator collision semantics changes (`ContinuousOccupancy` untouched).
- No benchmark clearance-metric redefinition (`min_clearance` / `mean_clearance` unchanged).
- No safety-performance or calibration claim. `alpha` is a heuristic conservatism knob, not a
  certified coverage bound.

## Deferred to a successor slice (residual risk)

The authoritative maintainer plan (issue #4141 comment, 2026-07-02) also proposed items intentionally
deferred here to keep this PR a small, low-risk first slice off shared simulator/identity paths and
because the launcher forbids benchmark-campaign execution:

- Scenario-level config threading through `SimulationSettings` / `sim_config.py` /
  `build_robot_config_from_scenario` with the same `pedestrian_uncertainty_*` field names, plus a
  resume-identity regression (distinct `episode_id` / `config_hash`). This slice configures the
  planner through the existing algorithm YAML mechanism instead.
- Secondary integration into `NMPCSocialPlannerAdapter` soft pedestrian-clearance cost.
- Benchmark runtime integration test (`tests/benchmark/test_uncertainty_envelope_runtime.py`)
  comparing `alpha=0.0` vs `alpha=0.1` geometric clearance on a crafted fixed-seed scenario.

## Future seam

A future calibrated policy may replace `linear_inflation_policy(alpha, dt)` with a conformal policy
such as `r_conf(horizon_step)` (issue #4138) without changing the `PedestrianUncertaintyEnvelope`
dataclass, the `effective_pedestrian_radius(...)` helper, or the planner clearance-query call sites.
