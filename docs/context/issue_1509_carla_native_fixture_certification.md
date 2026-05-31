# Issue #1509 CARLA Native Fixture Certification

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1509>

Parent: <https://github.com/ll7/robot_sf_ll7/issues/1491>

Evidence:
[`docs/context/evidence/issue_1509_carla_native_fixture_2026-05-31/`](evidence/issue_1509_carla_native_fixture_2026-05-31/)

## Outcome

Issue #1509 produced a fail-closed fixture certification result on `auxme-imech036`.

The certified `pr_promoted_planner_smoke` scenario exported to CARLA T0 and replayed through the
live CARLA Docker path, but the replay was not native/aligned under the Issue #1444 coordinate
contract:

- scenario certificate: `valid`, `eligible`;
- CARLA Docker preflight: `available`, image `carlasim/carla:0.9.16`;
- live replay status: `oracle-replay`;
- live replay mode: `oracle-replay-adapted`;
- coordinate alignment replay mode: `adapted`;
- robot spawn projection: `18.190854674447806 m`;
- parity report: `unavailable`.

This is durable evidence that the fixture remains non-eligible for metric parity today. It is not a
benchmark-strength CARLA/native parity success.

## Implementation Change

The live replay output now emits explicit Issue #1444-style `coordinate_alignment` metadata inside
the replay result, including:

- `replay_mode`;
- `projection_meters`;
- `projection_rationale`;
- CARLA map and server version;
- Robot-SF commit;
- scenario certificate id/source;
- replay start/end timestamps;
- `eligible_for_metric_parity`.

The parity adapter now also checks this coordinate-alignment metadata, including nested Docker
runtime replay output, so an adapted replay cannot be counted as comparable merely because legacy
mode/status fields look permissive.

## Commands

```bash
uv run python scripts/tools/certify_scenarios.py \
  configs/scenarios/single/pr_promoted_planner_smoke.yaml \
  --scenario-id pr_promoted_planner_smoke \
  --output output/issue_1509_carla_native_fixture/scenario_cert_pr_promoted.json \
  --fail-on-excluded
```

```bash
uv run robot-sf-export-carla-t0 \
  --scenario-file configs/scenarios/single/pr_promoted_planner_smoke.yaml \
  --output-dir output/issue_1509_carla_native_fixture/pr_promoted_export \
  --robot-sf-commit 3cdc075a
```

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime preflight --json
```

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest output/issue_1509_carla_native_fixture/pr_promoted_export/manifest.json \
  --scenario-id pr_promoted_planner_smoke \
  --max-steps 10 \
  --json
```

```bash
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf docs/context/evidence/issue_1467_carla_replay_metrics_2026-05-24/robot_sf_metrics_reference.json \
  --carla output/issue_1509_carla_native_fixture/live_replay_pr_promoted.json \
  --output output/issue_1509_carla_native_fixture/parity_report_pr_promoted.json
```

## Validation

```bash
uv run pytest -q \
  tests/carla_bridge/test_t1_live_replay.py \
  tests/carla_bridge/test_parity.py \
  tests/carla_bridge/test_parity_cli.py \
  tests/carla_bridge/test_docker_runtime.py \
  --no-cov
```

Result: `40 passed in 1.64s`.

## Claim Boundary

This closes the #1509 stage as a conservative fail-closed result. The replay path runs and records
the coordinate-alignment reason for exclusion, but native/aligned parity is still blocked until a
certified scenario fixture spawns without projection or carries a pre-declared reversible alignment
whose threshold is satisfied.
