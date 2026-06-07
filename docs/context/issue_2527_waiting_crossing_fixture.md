# Issue #2527 Waiting-Then-Crossing Fixture (2026-06-07)

Status: current diagnostic fixture and map-runner trace metadata smoke; not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2527
- Parent scope note: [issue_2472_intent_conditioned_behavior.md](issue_2472_intent_conditioned_behavior.md)
- Scenario fixture: `configs/scenarios/single/issue_2527_waiting_then_crossing.yaml`
- Trace metadata implementation: `robot_sf/benchmark/map_runner.py`
- Focused tests: `tests/benchmark/test_issue_2527_waiting_crossing_fixture.py`
- Signal-state proxy smoke:
  [issue_2564_signal_state_proxy_smoke.md](issue_2564_signal_state_proxy_smoke.md)

## Result

Issue #2527 adds a deterministic single-pedestrian fixture that encodes an authored
waiting-then-crossing behavior and exposes that authored intent through optional map-runner
simulation-step trace metadata.

The fixture is intentionally diagnostic-only. It uses existing single-pedestrian controls
(`trajectory`, `wait_at`, and `speed_m_s`) plus explicit metadata describing
`waiting_then_crossing` phases. Issue #2564 adds trace-only proxy `signal_state` metadata that maps
the authored `waiting` phase to `robot_green_pedestrian_dont_walk` and the authored `crossing`
phase to `pedestrian_walk_robot_red`. When a scenario or single pedestrian opts into
`metadata.intent_conditioned_behavior`, map-runner adds:

- `algorithm_metadata.intent_conditioned_behavior` with `status: diagnostic_metadata_only`;
- optional per-pedestrian `intent_label`, `intent_phase`, `intent_source`,
  `claim_boundary`, and `behavior_parameters` fields under
  `algorithm_metadata.simulation_step_trace.steps[].pedestrians[]`.
- optional per-pedestrian `signal_state` fields when scenario metadata declares the #2564
  proxy signal-state wrapper.

Scenarios without explicit intent-conditioned metadata keep their existing trace shape.

## Claim Boundary

This is authored scenario metadata only. It does not prove realistic human intent, replace Social
Force behavior, certify the fixture for benchmark comparisons, prove signalized-crossing legality,
or support planner-ranking claims.
The metadata is useful for smoke-testing the trace surface requested by #2472 and for selecting the
next evidence step.

## Validation

Targeted validation:

```bash
uv run pytest tests/benchmark/test_issue_2527_waiting_crossing_fixture.py -q
uv run pytest tests/benchmark/test_map_runner_utils.py::test_run_map_episode_records_synthetic_actuation_metrics -q
uv run ruff check robot_sf/benchmark/map_runner.py tests/benchmark/test_issue_2527_waiting_crossing_fixture.py
uv run ruff format --check robot_sf/benchmark/map_runner.py tests/benchmark/test_issue_2527_waiting_crossing_fixture.py
uv run python scripts/tools/validate_scenario.py configs/scenarios/single/issue_2527_waiting_then_crossing.yaml
```

Observed result on 2026-06-07: all targeted commands passed.

## Follow-Up Direction

The next useful proof is either a broader trace/replay pipeline integration for intent-conditioned
and signal-state fields or a runtime snapshot that records release timing and role provenance
outside map-runner's analysis-only metadata. Data-grounded intent realism, signalized-crossing
legality, planner observability, and planner-ranking claims should remain separate issues with
their own evidence targets.
