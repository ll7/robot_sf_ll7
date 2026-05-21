# Issue #1430 CARLA Live Replay Parity Evidence 2026-05-21

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1430>
Parent epic: <https://github.com/ll7/robot_sf_ll7/issues/872>

## Summary

This bundle records the post-#1329 live CARLA replay attempt for the certified #1111 payload.
The preferred `imech036` route was not usable non-interactively because SSH stopped at host-key
verification. The fallback host `imech156-u` was used.

The fallback host satisfied the Docker/NVIDIA/CARLA server prerequisites and connected a CARLA
0.9.16 Python client to a CARLA 0.9.16 server on `Carla/Maps/Town10HD_Opt`. The run then failed
closed before metric-producing oracle replay:

```text
status: failed
mode: failed
reason: CARLA failed to spawn robot
```

The previous #1169 blocker, `T0 payload static obstacle replay is not implemented`, was not the
observed failure after the #1329 static-geometry support. Metric parity remains unavailable because
the run did not reach `oracle-replay`.

Follow-up #1437 tracks diagnosis of the robot actor-spawn failure before the parent epic can claim
positive metric parity.

## Commands

Host route check for the preferred host:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 imech036 \
  'hostname; command -v docker || true; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true; pwd'
```

Result: `Host key verification failed.`

Fallback preflight:

```bash
uv run robot-sf-carla-docker-runtime preflight --skip-api-check --json
```

Fallback live replay:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/manifest.json \
  --max-steps 10 \
  --pull \
  --json
```

Parity unavailable report:

```bash
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf docs/context/evidence/issue_1430_carla_live_parity_2026-05-21/robot_sf_metrics_reference.json \
  --carla docs/context/evidence/issue_1430_carla_live_parity_2026-05-21/live_replay_imech156.json \
  --output docs/context/evidence/issue_1430_carla_live_parity_2026-05-21/parity_report.json
```

## Files

- `live_replay_imech156.json`: CARLA Docker runtime and live replay result.
- `robot_sf_metrics_reference.json`: placeholder Robot-SF record documenting that no metric
  comparison was attempted because CARLA replay failed before metrics existed.
- `parity_report.json`: conservative #1110 parity report; all rows are `unavailable` because
  CARLA status/mode is `failed`.
- `SHA256SUMS`: checksums for the JSON evidence files.

## Artifact Decision

The JSON files in this directory are small, reviewable evidence and should be tracked. The pulled
`carlasim/carla:0.9.16` Docker image is local machine state and must not be committed. No raw CARLA
videos, logs, or bulky replay outputs were added.
