# Issue #1168 Multi-AMV Planner Support Classification

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1168>

## Decision

The multi-AMV benchmark surface currently supports only `goal_controller_smoke` as a native smoke
execution path. It is not real planner-family support and must not be reported as a coordinated
fleet planner.

All existing single-robot benchmark planner families fail closed or remain research-only until they
define a multi-AMV contract for action shape, robot identity, inter-robot collision responsibility,
and metadata reporting.

## Minimum planner contract

A non-smoke multi-AMV planner must define:

- Action shape for all robots, currently expected as a stable fleet action tensor.
- Robot identity semantics so action rows and metrics map to the same robot across the episode.
- Collision responsibility across robot-robot, robot-pedestrian, and robot-obstacle interactions.
- Metadata reporting that records support mode as `native`, `adapter`, `not_available`, or
  `research_only`.
- Benchmark output metadata that distinguishes smoke execution from planner-family support.

## Inventory

The code inventory lives in `robot_sf.benchmark.multi_amv.multi_amv_planner_support_inventory`.

Current classification:

- `goal_controller_smoke`: `native`, but only for the minimal smoke runner.
- `goal`: `not_available`.
- `social_force`: `research_only`.
- `orca`: `not_available`; this is the first plausible non-trivial candidate, but it needs an
  explicit adapter before benchmark use.
- `ppo`: `not_available`.
- `guarded_ppo`: `not_available`.
- `sacadrl`: `not_available`.
- `teb`: `research_only`.

## Fail-closed behavior

`ensure_multi_amv_planner_supported` raises before benchmark execution when a planner family lacks a
multi-AMV contract. Passing `require_non_smoke=True` also rejects `goal_controller_smoke`, which
prevents smoke output from being promoted as real planner support.

## Validation

Use:

```bash
uv run pytest tests/benchmark/test_multi_amv.py -q
```

Full PR readiness should use the stacked base branch for changed-file coverage while this work
depends on `#1128`.
