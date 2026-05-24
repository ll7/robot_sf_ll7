# Issue #1442 CARLA Native Spawn Probe Evidence

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1442>

This bundle records the 2026-05-24 CARLA parity-gate attempt on `auxme-imech036`.

## Outcome

- CARLA Docker runtime preflight passed when run with `carla==0.9.16`.
- The existing certified #1111 payload still ran only as `oracle-replay-adapted`.
  The robot spawn was projected by `18.190854674447806 m`, so it remains
  non-parity diagnostic evidence under the #1444 coordinate-alignment contract.
- A generated native-spawn probe payload, using the inverse Robot-SF coordinates
  of the previously projected CARLA spawn, reached `oracle-replay` with exact
  robot placement and no adaptations.
- The native-spawn probe does not yet provide trajectory-level CARLA metrics.
  The parity report is therefore `unavailable` with the narrower blocker:
  native replay can run, but the current T1 replay output lacks comparable
  metric fields.

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
  --manifest docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/manifest.json \
  --max-steps 10 \
  --json
```

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest output/issue_1442_carla_parity_2026-05-24/native_probe/manifest.json \
  --max-steps 10 \
  --json
```

```bash
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/robot_sf_metrics_reference.json \
  --carla docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/live_replay_native_probe.json \
  --output docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24/parity_report_native_probe.json
```

## Files

- `preflight_with_carla.json`: CARLA Docker/runtime prerequisite report.
- `live_replay_certified_payload_adapted.json`: the existing certified #1111
  payload replay, still adapted by CARLA map projection.
- `native_probe/`: generated T0 payload and manifest for the exact-spawn probe.
- `live_replay_native_probe.json`: live CARLA replay result for the native probe.
- `robot_sf_metrics_reference.json`: placeholder Robot-SF metric reference.
- `parity_report_native_probe.json`: conservative parity adapter output.
- `SHA256SUMS`: checksums for the evidence files.

## Artifact Decision

The JSON files are small and reviewable. They are tracked to preserve the runtime
boundary and the newly discovered native-spawn route. The CARLA Docker image,
coverage output, and raw ignored `output/` copies remain local machine state.
