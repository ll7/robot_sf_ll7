# Issue #1467 CARLA Replay Metrics Evidence

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1467>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/872>

This bundle records the 2026-05-24 native CARLA replay metric probe on
`auxme-imech036`.

## Outcome

- CARLA Docker runtime preflight passed with `carla==0.9.16`.
- The generated CARLA-aligned native metric probe reached `oracle-replay` with
  exact robot spawn and no adaptations.
- T1 live replay emitted a top-level replay `metrics` mapping:
  `success=true`, `collision=false`, `intervention_rate=0.0`.
- `compare_oracle_replay_metrics.py` produced `status=comparable`:
  `success` and `collision` matched, and `intervention_rate` was comparable with
  `delta=0.0`.

## Commands

```bash
uv run pytest -q \
  tests/carla_bridge/test_t1_live_replay.py \
  tests/carla_bridge/test_parity.py \
  tests/carla_bridge/test_parity_cli.py \
  --no-cov
```

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime preflight --json
```

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest output/issue_1467_carla_metrics/native_metric_probe/manifest.json \
  --json
```

```bash
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf output/issue_1467_carla_metrics/robot_sf_metrics_reference.json \
  --carla output/issue_1467_carla_metrics/live_replay_native_metric_probe.json \
  --output output/issue_1467_carla_metrics/parity_report_native_metric_probe.json
```

## Files

- `preflight_with_carla.json`: CARLA Docker/runtime prerequisite report.
- `native_metric_probe/`: generated T0 payload and manifest for the short native
  metric probe.
- `live_replay_native_metric_probe.json`: live replay result with emitted
  replay metrics.
- `robot_sf_metrics_reference.json`: Robot-SF-side reference metrics for the
  generated smoke probe.
- `parity_report_native_metric_probe.json`: conservative parity adapter output.
- `SHA256SUMS`: checksums for the evidence files.

## Claim Boundary

This is a native metric-emission smoke, not a broad CARLA transfer campaign. It
proves the replay/adapter path can now produce comparable rows for a generated
CARLA-aligned native probe. Broader or paper-facing CARLA claims still need a
durable certified scenario/export contract and stronger metric coverage.
