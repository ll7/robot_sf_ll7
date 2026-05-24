# Issue #1467 CARLA Replay Metrics

Issue: [#1467](https://github.com/ll7/robot_sf_ll7/issues/1467)

Parent: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)

Evidence:
[`docs/context/evidence/issue_1467_carla_replay_metrics_2026-05-24/`](evidence/issue_1467_carla_replay_metrics_2026-05-24/)

## Outcome

T1 live replay now emits conservative metrics for native CARLA oracle replay
outputs. The metrics are derived from the scripted replay path and include the
fields that can be defended from the current T0/T1 contract:

- `success`
- `collision`
- `min_distance_m` when pedestrians or supported static obstacles are present
- `intervention_rate`

The parity adapter now also reads metrics from the nested Docker-runtime
`replay.metrics` field, so output from `robot-sf-carla-docker-runtime
live-replay --json` can be passed directly to
`scripts/carla_bridge/compare_oracle_replay_metrics.py`.

## Validation

CARLA-free tests:

```bash
uv run pytest -q \
  tests/carla_bridge/test_t1_live_replay.py \
  tests/carla_bridge/test_parity.py \
  tests/carla_bridge/test_parity_cli.py \
  --no-cov
```

Result: `22 passed in 1.63s`.

CARLA runtime proof:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest output/issue_1467_carla_metrics/native_metric_probe/manifest.json \
  --json
```

Result:

```text
status: oracle-replay
mode: oracle-replay
robot_spawn.adapted: false
metrics: success=true, collision=false, intervention_rate=0.0
```

Parity comparison:

```bash
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf output/issue_1467_carla_metrics/robot_sf_metrics_reference.json \
  --carla output/issue_1467_carla_metrics/live_replay_native_metric_probe.json \
  --output output/issue_1467_carla_metrics/parity_report_native_metric_probe.json
```

Result:

```text
status: comparable
success: match
collision: match
intervention_rate: comparable, delta=0.0
```

## Claim Boundary

This closes the metric-emission blocker identified by #1442 / PR #1466 for a
generated native CARLA-aligned smoke probe. It does not turn adapted replay into
parity evidence, and it does not by itself establish a broad simulator-transfer
claim for #872. The remaining parent-epic strengthening path is a durable
certified native/aligned scenario fixture with richer metric coverage.
