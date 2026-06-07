# Issue #2564 Signal-State Proxy Smoke (2026-06-07)

Status: current diagnostic proxy smoke; not benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2564
- Signalized crossing scope note: [issue_2474_signalized_crossing_benchmark.md](issue_2474_signalized_crossing_benchmark.md)
- Waiting/crossing fixture note: [issue_2527_waiting_crossing_fixture.md](issue_2527_waiting_crossing_fixture.md)
- Scenario fixture: `configs/scenarios/single/issue_2527_waiting_then_crossing.yaml`
- Benchmark manifest: `configs/benchmarks/signalized_pedestrian_crossing_issue_2474.yaml`
- Trace implementation: `robot_sf/benchmark/map_runner.py`
- Compact smoke evidence: `docs/context/evidence/issue_2564_signal_state_proxy_smoke/summary.json`

## Result

Issue #2564 adds a proxy `signal_state` wrapper around the existing
`issue_2527_waiting_then_crossing` fixture. The fixture now records a two-phase diagnostic timeline:

- `robot_green_pedestrian_dont_walk` mapped to the authored `waiting` intent phase;
- `pedestrian_walk_robot_red` mapped to the authored `crossing` intent phase.

Map-runner propagates the proxy state into
`algorithm_metadata.simulation_step_trace.steps[].pedestrians[].signal_state` whenever the scenario
opts into authored intent metadata. The existing intent summary also records the signal-state proxy
surface and its trace fields.

## Smoke Evidence

One local `goal` planner smoke ran the fixture with simulation-step trace recording enabled. The
compact tracked summary reports:

- 1 episode, 1 successful job, 0 failed jobs;
- 80 trace steps and 80 signal-state pedestrian frames;
- both proxy phases emitted in the trace;
- first frame: waiting under `robot_green_pedestrian_dont_walk`;
- last frame: crossing under `pedestrian_walk_robot_red`.

Raw local episode JSONL remains under ignored `output/benchmark/issue2564_signal_state_smoke/` and
is not durable. The compact summary JSON above is the reviewable evidence pointer.

## Claim Boundary

This is proxy diagnostic evidence only. It does not prove traffic-signal realism,
crossing-legality compliance, forced-waiting reasoning, planner observability, or planner-ranking
improvement. The signal state is `trace_metadata_only` and `planner_observable: false`.

## Validation

Targeted validation:

```bash
uv run pytest tests/benchmark/test_issue_2527_waiting_crossing_fixture.py
uv run pytest tests/benchmark/test_signalized_crossing_benchmark_manifest.py
uv run ruff check robot_sf/benchmark/map_runner.py tests/benchmark/test_issue_2527_waiting_crossing_fixture.py tests/benchmark/test_signalized_crossing_benchmark_manifest.py
uv run ruff format --check robot_sf/benchmark/map_runner.py tests/benchmark/test_issue_2527_waiting_crossing_fixture.py tests/benchmark/test_signalized_crossing_benchmark_manifest.py
```

Runtime smoke command shape:

```bash
uv run python - <<'PY'
# Run one goal-planner episode from configs/scenarios/single/issue_2527_waiting_then_crossing.yaml
# with record_simulation_step_trace=True and write a compact summary JSON.
PY
```

## Follow-Up Direction

The next stronger proof would make signal state an explicit runtime/observation contract rather
than trace-only metadata, then define whether each benchmark row is planner-observable, hidden, or
motion-only before any planner comparison.
