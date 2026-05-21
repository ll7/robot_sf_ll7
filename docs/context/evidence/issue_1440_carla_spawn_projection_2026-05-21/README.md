# Issue #1440 CARLA Spawn Projection Evidence 2026-05-21

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1440>
Parent epic: <https://github.com/ll7/robot_sf_ll7/issues/872>
Stacked prerequisite: <https://github.com/ll7/robot_sf_ll7/pull/1439>

## Summary

This bundle records the CARLA live replay reruns for the robot spawn-placement blocker found in
Issue #1437. The first projection attempt used CARLA's projected waypoint transform directly and
still failed because the waypoint was on the road surface at `z=0.000`. The final attempt applies
the vehicle spawn height to the projected waypoint and reaches `oracle-replay` on `imech156-u`.

Final live result:

```text
status: oracle-replay
mode: oracle-replay-adapted
projection: x=2.449, y=13.191, z=0.600, yaw=180.159
distance from requested spawn: 18.191 m
```

This is live CARLA execution progress, not native Robot-SF/CARLA metric parity. The robot spawn was
projected onto CARLA's map surface, and the parity adapter correctly reports `unavailable` for
`oracle-replay-adapted`.

## Commands

Initial projection attempt:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/manifest.json \
  --max-steps 10 \
  --pull \
  --json
```

Final replay after adding vehicle spawn height to the projected waypoint:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/manifest.json \
  --max-steps 10 \
  --pull \
  --json
```

Parity gate:

```bash
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf docs/context/evidence/issue_1440_carla_spawn_projection_2026-05-21/robot_sf_metrics_reference.json \
  --carla docs/context/evidence/issue_1440_carla_spawn_projection_2026-05-21/live_replay_imech156_spawn_projection_z06.json \
  --output docs/context/evidence/issue_1440_carla_spawn_projection_2026-05-21/parity_report.json
```

## Files

- `live_replay_imech156_spawn_projection.json`: first projected-spawn attempt; failed at projected
  `z=0.000`.
- `live_replay_imech156_spawn_projection_z06.json`: final live replay output; reached
  `oracle-replay` in `oracle-replay-adapted` mode.
- `robot_sf_metrics_reference.json`: reference placeholder for the parity gate.
- `parity_report.json`: conservative #1110 parity report; all rows are `unavailable` because the
  CARLA replay mode is adapted rather than native/comparable.
- `SHA256SUMS`: checksums for the JSON evidence files.

## Interpretation

Issue #1440 clears the actor-spawn placement blocker for the certified payload by proving that the
robot can be spawned through an explicit CARLA map projection. The projection is large enough
(`18.191 m`) that parent Issue #872 still cannot claim native metric parity. The next parent-level
question is whether a benchmark-facing CARLA replay should use a CARLA-native scenario placement or
an explicit map-alignment contract before comparable trajectory metrics are meaningful.

## Artifact Decision

The JSON files in this directory are small, reviewable evidence and should be tracked. The
`carlasim/carla:0.9.16` Docker image and generated coverage output are local machine state and must
not be committed.
