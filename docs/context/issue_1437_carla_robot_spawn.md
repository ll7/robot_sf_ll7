# Issue #1437 CARLA Robot Actor Spawn Failure

Issue: [#1437](https://github.com/ll7/robot_sf_ll7/issues/1437)
Parent epic: [#872](https://github.com/ll7/robot_sf_ll7/issues/872)
Evidence bundle:
[`docs/context/evidence/issue_1437_carla_robot_spawn_2026-05-21/`](evidence/issue_1437_carla_robot_spawn_2026-05-21/)

## Outcome

The CARLA live replay robot-spawn failure is now narrower than the #1430 result. The branch keeps
the replay fail-closed, but improves the diagnostics and one plausible robustness gap:

- `_blueprint(...)` now falls back to the filtered blueprint list when `library.find(...)` returns
  `None`, not only when it raises.
- Actor spawn failures now report the blueprint id, spawn API, CARLA transform location, and yaw.
- Vehicle spawn and replay transforms now use `z=0.6` instead of the generic `z=0.1` actor height.

The live rerun on `imech156-u` still failed before `oracle-replay`:

```text
CARLA failed to spawn robot with blueprint vehicle.tesla.model3 via try_spawn_actor at x=2.500, y=-5.000, z=0.600, yaw=-0.000
```

## Interpretation

This does not satisfy the parent epic's positive metric-parity requirement. It does satisfy the
Issue #1437 diagnostic slice by replacing the generic `CARLA failed to spawn robot` condition with a
specific blueprint, spawn API, and transform. The #1110 parity adapter continues to report `status:
unavailable` because CARLA replay status/mode is `failed`.

The next likely implementation question is whether Robot-SF T0 coordinates need projection onto a
valid CARLA map spawn point or drivable waypoint before spawning a vehicle actor.

## Validation

CARLA-free validation:

```bash
uv run ruff check robot_sf_carla_bridge/live_replay.py tests/carla_bridge/test_t1_live_replay.py
uv run pytest tests/carla_bridge/test_t1_live_replay.py tests/carla_bridge/test_docker_runtime.py -q
```

Live rerun:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/manifest.json \
  --max-steps 10 \
  --pull \
  --json
```
