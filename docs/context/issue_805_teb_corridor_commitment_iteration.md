# Issue 805 TEB Corridor-Commitment Iteration

Date: 2026-04-13

Related issue:
- `robot_sf_ll7#805` TEB corridor-commitment: strengthen obstacle avoidance to fix collision-on-all-topology regression

Relevant implementation surfaces:
- `robot_sf/planner/teb_commitment.py`
- `configs/algos/teb_commitment_camera_ready.yaml`
- `configs/scenarios/sets/issue_805_teb_topology_slice.yaml`
- `tests/planner/test_teb_commitment.py`

## Goal

Tighten the native TEB-inspired corridor-commitment planner enough to improve the `#805`
three-scenario topology slice without broadening it into a different planner family.

## What changed

- Replaced the single forward occupancy probe with a short multi-step corridor score.
- Added committed-heading escalation so the planner can increase lateral deflection when the first
  sidestep is still blocked.
- Added side-flip fallback when the initially preferred corridor is clearly worse.
- Added flank sampling around each candidate heading so the blocked score reacts to side-wall
  occupancy, not just centerline hits.
- Updated the experimental camera-ready config to use a longer blocked-probe horizon.
- Added a dedicated reproducible scenario matrix for the exact `#805` slice.

## Validation commands

Unit tests:

```bash
uv run pytest tests/planner/test_teb_commitment.py -q
```

Topology slice:

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/sets/issue_805_teb_topology_slice.yaml \
  --algo teb \
  --algo-config configs/algos/teb_commitment_camera_ready.yaml \
  --benchmark-profile experimental \
  --out output/benchmarks/issue_805_teb_slice_teb.jsonl \
  --workers 1 \
  --no-resume

uv run robot_sf_bench run \
  --matrix configs/scenarios/sets/issue_805_teb_topology_slice.yaml \
  --algo orca \
  --benchmark-profile experimental \
  --out output/benchmarks/issue_805_teb_slice_orca.jsonl \
  --workers 1 \
  --no-resume
```

## Observed result

TEB after the iteration:

- `line_wall_detour`: `0/3` success, `0 collision`, `3 max_steps`
- `narrow_passage`: `0/3` success, `0 collision`, `3 max_steps`
- `symmetry_ambiguous_choice`: `0/3` success, `3 collisions`

ORCA on the same slice on 2026-04-13:

- `line_wall_detour`: `0/3` success, `2 collisions`, `1 max_steps`
- `narrow_passage`: `3/3 success`
- `symmetry_ambiguous_choice`: `0/3`, `3 max_steps`

## Conclusion

The iteration improved one failure mode but did **not** satisfy issue `#805`.

- The planner no longer collides on `line_wall_detour`; it now stalls instead.
- `narrow_passage` also moved from collision behavior to stall behavior.
- `symmetry_ambiguous_choice` remains a hard collision case.

This means the branch reduced some direct-wall impacts but still finished with `0/9` successes on
the target slice, so it does not meet the issue DoD and is not PR-ready as an issue-closing fix.

## Second iteration changes (2026-04-14)

Root-cause analysis of the stalling and remaining collision after the first iteration:

1. **Speed budget exhaustion** (`line_wall_detour`, `narrow_passage`): The embedded
   `GridRoutePlannerAdapter` used `waypoint_lookahead_cells=5` (0.5 m at 0.1 m/cell). The
   `_nominal_command` linear speed scales with waypoint distance, capping at `min(0.5, 0.9) = 0.5
   m/s` instead of full speed.  At 0.5 m/s × 320 steps × 0.1 s/step = 16 m traversable, but
   detour paths are ~18 m — the budget runs out.

2. **Corner-clipping** (`symmetry_ambiguous_choice`): Default `obstacle_inflation_cells=1` (0.1 m)
   left the A* path too close to the splitter corners.  The robot's unicycle dynamics caused it to
   clip those corners while tracking the route.

3. **Stuck-wall behaviour**: When every committed heading scored above the occupancy threshold, the
   planner previously continued to drive slowly in the "least blocked" direction, pushing against
   the obstacle.

### Changes applied

- `GridRoutePlannerAdapter` embedded in TEB now uses a custom `GridRoutePlannerConfig`:
  - `waypoint_lookahead_cells=10` → 1.0 m target → linear speed = 0.9 m/s (full budget)
  - `obstacle_inflation_cells=3` → 0.3 m clearance matching the robot radius
  - `stop_distance=0.5` → stops earlier when an obstacle enters the forward corridor
- `plan()` refactored to reduce cyclomatic complexity (three helpers extracted:
  `_try_route_command`, `_commitment_step`, `_rescue_or_stop`).
- `_rescue_or_stop`: when all committed headings are occupied, yields to the route guide command;
  stops the robot (returns `(0, 0)`) rather than driving into the wall when no route escape exists.

### Validation

```bash
uv run pytest tests/planner/test_teb_commitment.py -q  # 15 passed
BASE_REF=origin/main scripts/dev/pr_ready_check.sh      # 607 passed, 1 pre-existing SAC failure
```

The topology-slice benchmark has **not** been re-run yet.  Run the commands in the **Validation
commands** section above and record new results here before claiming DoD is met.

## Follow-up boundary

The stalling root cause (speed budget) has been fixed.  The corner-clipping root cause
(insufficient inflation) has been addressed.  Whether these changes deliver the required ≥ 1/3
success on at least one scenario depends on the next benchmark run.

The pre-existing SAC test failure (`test_step_vector_mode_uses_model_prediction_and_fallback_action`
— floating-point ULP difference) is unrelated to TEB and existed before this branch.

Treat this note as the handoff point for benchmark validation of the second iteration.
