# Issue #1440 CARLA Robot Spawn Projection

Issue: [#1440](https://github.com/ll7/robot_sf_ll7/issues/1440)
Parent epic: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)
Evidence bundle:
[`docs/context/evidence/issue_1440_carla_spawn_projection_2026-05-21/`](evidence/issue_1440_carla_spawn_projection_2026-05-21/)

## Outcome

The CARLA live replay can now get past the robot actor spawn blocker from Issue #1437 when the exact
Robot-SF start pose is rejected. The branch keeps exact placement first, then uses CARLA
`world.get_map().get_waypoint(project_to_road=True)` as an explicit fallback for the robot vehicle
spawn. The projected waypoint is lifted by the vehicle spawn height before spawning.

The final live run on `imech156-u` reached:

```text
status: oracle-replay
mode: oracle-replay-adapted
```

The adaptation is recorded in both `adaptations` and `replay_metadata.robot_spawn`, including the
requested transform, projected spawn transform, projection source, and parity caveat.

## Interpretation

This is live replay evidence, but not metric-parity evidence. The projected robot spawn moved about
`18.191 m` from the requested Robot-SF start, so the parity adapter treats `oracle-replay-adapted`
as unavailable instead of comparable. That preserves the parent epic's claim boundary: the bridge
can now execute an adapted CARLA replay of the certified payload, but native map-aligned
Robot-SF/CARLA metric parity remains open.

## Validation

CARLA-free validation:

```bash
uv run ruff check robot_sf_carla_bridge/live_replay.py robot_sf_carla_bridge/parity.py tests/carla_bridge/test_t1_live_replay.py tests/carla_bridge/test_parity.py
uv run pytest tests/carla_bridge/test_t1_live_replay.py tests/carla_bridge/test_parity.py tests/carla_bridge/test_parity_cli.py -q --no-cov
```

Live replay:

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
