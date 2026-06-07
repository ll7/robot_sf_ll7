# Issue #2526 Cyclist-Like VRU Smoke (2026-06-07)

Status: current diagnostic cyclist-like VRU fixture and map-runner trace metadata smoke; not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2526
- Parent scope note: [issue_2473_cyclist_interaction_benchmark.md](issue_2473_cyclist_interaction_benchmark.md)
- Scenario fixture: `configs/scenarios/single/issue_2526_cyclist_vru_smoke.yaml`
- Trace metadata implementation: `robot_sf/benchmark/map_runner.py`
- Focused tests: `tests/benchmark/test_issue_2526_cyclist_vru_smoke.py`

## Result

Issue #2526 adds the first executable diagnostic surface for the #2473 cyclist interaction direction.
It reuses the Francis 2023 single-pedestrian map actor `h1` as a fast-moving, cyclist-like VRU
proxy with authored `metadata.cyclist_like_vru` fields for configured speed, acceleration, actor
radius, interaction role, and claim boundary.

When a scenario opts into `metadata.cyclist_like_vru`, map-runner now records:

- `algorithm_metadata.cyclist_like_vru` with `status: diagnostic_metadata_only`;
- optional per-pedestrian `actor_type`, `interaction_role`, `claim_boundary`, and
  `cyclist_like_vru` diagnostics under
  `algorithm_metadata.simulation_step_trace.steps[].pedestrians[]`;
- diagnostic fields for configured speed, measured speed, acceleration, distance to robot,
  relative closing speed, time-to-conflict-zone proxy, clearance, and pass/overtake state when
  robot position is available.

Scenarios without cyclist-like VRU metadata keep their existing trace shape.

## Claim Boundary

This is authored proxy metadata and trace plumbing only. It does not prove cyclist behavior
realism, add a cyclist simulator backend, validate fast VRU physics fidelity, or support planner
ranking. The fixture is useful as a smoke surface for trace export and as a next-step blocker
reduction for #2473.

## Validation

Targeted validation:

```bash
uv run pytest tests/benchmark/test_issue_2526_cyclist_vru_smoke.py -q
uv run pytest tests/benchmark/test_issue_2527_waiting_crossing_fixture.py tests/benchmark/test_cyclist_interaction_benchmark_manifest.py tests/benchmark/test_issue_2526_cyclist_vru_smoke.py -q
uv run ruff check robot_sf/benchmark/map_runner.py tests/benchmark/test_issue_2526_cyclist_vru_smoke.py
uv run ruff format --check robot_sf/benchmark/map_runner.py tests/benchmark/test_issue_2526_cyclist_vru_smoke.py
uv run python scripts/tools/validate_scenario.py configs/scenarios/single/issue_2526_cyclist_vru_smoke.yaml
```

Observed result on 2026-06-07: all targeted commands passed.

## Follow-Up Direction

The next useful proof is an end-to-end map-runner episode artifact that shows the cyclist-like VRU
trace fields in exported episode metadata. A benchmark-strength cyclist interaction claim still
needs native or validated adapter cyclist dynamics, calibrated geometry/radius semantics, and
planner-comparison evidence separate from this proxy smoke.
