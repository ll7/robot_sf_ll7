# Issue #1437 CARLA Robot Actor Spawn Evidence 2026-05-21

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1437>
Parent epic: <https://github.com/ll7/robot_sf_ll7/issues/872>

## Summary

This bundle records the CARLA live replay reruns used to diagnose the robot actor-spawn blocker
observed in #1430. The branch adds two CARLA-free hardening changes:

- robot blueprint lookup now falls back when `BlueprintLibrary.find("vehicle.tesla.model3")`
  returns `None` instead of raising;
- actor spawn failures now include the blueprint id, spawn API, CARLA transform location, and yaw.

The live rerun on fallback host `imech156-u` still failed closed, but with a narrower reason:

```text
CARLA failed to spawn robot with blueprint vehicle.tesla.model3 via try_spawn_actor at x=2.500, y=-5.000, z=0.600, yaw=-0.000
```

The replay reached a live CARLA `0.9.16` server on `Town10HD_Opt`; it did not reach
`oracle-replay`, so the parity report remains `unavailable`.

## Commands

Initial diagnostic rerun after adding detailed spawn diagnostics:

```bash
uv run --with carla==0.9.16 robot-sf-carla-docker-runtime live-replay \
  --manifest docs/context/evidence/issue_1111_carla_setup_smoke_2026-05-18/manifest.json \
  --max-steps 10 \
  --pull \
  --json
```

Final rerun after separating vehicle spawn height from generic actor height:

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
  --robot-sf docs/context/evidence/issue_1437_carla_robot_spawn_2026-05-21/robot_sf_metrics_reference.json \
  --carla docs/context/evidence/issue_1437_carla_robot_spawn_2026-05-21/live_replay_imech156_vehicle_z06.json \
  --output docs/context/evidence/issue_1437_carla_robot_spawn_2026-05-21/parity_report.json
```

## Files

- `live_replay_imech156_after_patch.json`: live replay output after adding detailed spawn
  diagnostics before the spawn API and yaw were added to the message.
- `live_replay_imech156_vehicle_z06.json`: live replay output after using `z=0.6` for vehicle
  spawn and replay transforms.
- `robot_sf_metrics_reference.json`: placeholder Robot-SF record documenting that no metric
  comparison was attempted because CARLA replay failed before metrics existed.
- `parity_report.json`: conservative #1110 parity report; all rows are `unavailable` because
  CARLA status/mode is `failed`.
- `SHA256SUMS`: checksums for the JSON evidence files.

## Interpretation

The robot actor-spawn failure is now narrowed to the specific CARLA blueprint, spawn API, and
transform.
Changing vehicle Z from `0.1` to `0.6` did not clear the live failure on `imech156-u`, so the next
CARLA slice should inspect whether the certified Robot-SF start pose maps to a valid drivable CARLA
spawn area, whether a CARLA map spawn-point projection is needed, or whether a different vehicle
blueprint/spawn strategy is required.

## Artifact Decision

The JSON files in this directory are small, reviewable evidence and should be tracked. The pulled
`carlasim/carla:0.9.16` Docker image is local machine state and must not be committed. No raw CARLA
videos, logs, or bulky replay outputs were added.
