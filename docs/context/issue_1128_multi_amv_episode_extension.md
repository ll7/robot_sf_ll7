# Issue 1128 multi-AMV episode extension

Date: 2026-05-12

## Scope implemented so far

This branch adds a namespaced, additive multi-AMV episode extension block and updates the existing multi-AMV smoke runner to emit it.

Implemented files:

- `robot_sf/benchmark/multi_amv.py`
- `scripts/validation/run_multi_amv_smoke.py`
- `tests/benchmark/test_multi_amv.py`

## Record shape

`multi_amv_episode_extension(...)` returns:

```json
{
  "multi_amv": {
    "enabled": true,
    "num_robots": 2,
    "near_miss_distance_m": 1.0,
    "collision_distance_m": 0.4,
    "deadlock_speed_mps": 0.05,
    "deadlock_window_steps": 10,
    "planner_status": "goal_controller_smoke",
    "planner_note": "...",
    "metrics": {
      "inter_robot": {}
    }
  }
}
```

The block is intentionally namespaced so existing single-robot episode consumers can ignore it without schema migration.

## Fail-closed behavior

The helper raises `ValueError` when called with fewer than two robots or empty inter-robot metrics.

## Remaining work

- Integrate multi-AMV execution into the primary benchmark command path or document the equivalent primary subcommand.
- Carry the `multi_amv.metrics.inter_robot` block into aggregate reports.
- Document planner support beyond the current goal-controller smoke path, or emit explicit unavailable/fail-closed statuses.
- Add schema/report regression tests covering coexistence with single-robot outputs.

## Validation status

No validation commands have been run for this partial implementation in this pass.
