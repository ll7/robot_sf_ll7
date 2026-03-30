# Issue 707 ORCA Atomic Tuning Note

## Goal

Re-run the issue-596 atomic suite with a tuned native ORCA configuration that specifically targets
the documented failure slice in `docs/context/issue_596_orca_failure_analysis.md`.

## Config and Artifacts

- Tuned algo config:
  `configs/algos/issue707_orca_tuned.yaml`
- Successful rerun artifact:
  `output/benchmarks/camera_ready/issue707_orca_atomic_issue707_orca_atomic_rerun_20260327_222351`
- Initial partial-failure run that exposed the route-override path bug:
  `output/benchmarks/camera_ready/issue707_orca_atomic_issue707_orca_atomic_20260327_221733`
- Route-override fix proof:
  `tests/test_scenario_loader_route_overrides.py`

## What Changed

The ORCA adapter now adds explicit preferred-velocity shaping for:

- head-on lateral bias,
- deterministic symmetry breaking,
- repeated low-progress stall commitment,
- forward-corridor blocking checks from the occupancy grid,
- obstacle-radius inflation for immediate-start and inside-corner clearance.

This remains the native `algo=orca` path using Python-RVO2. No fallback execution is treated as
success evidence.

## Full-Suite Result

Rerun summary from
`output/benchmarks/camera_ready/issue707_orca_atomic_issue707_orca_atomic_rerun_20260327_222351/reports/campaign_summary.json`:

- episodes: `69`
- success_mean: `0.8116`
- collisions_mean: `0.0870`
- near_misses_mean: `0.9130`
- time_to_goal_norm_mean: `0.5064`
- path_efficiency_mean: `0.9640`
- benchmark_success: `true`

## Failure-Slice Comparison Against The Prior Note

Compared with the March 27 baseline recorded in `docs/context/issue_596_orca_failure_analysis.md`:

- `head_on_interaction`
  - before: `3/3` pedestrian collisions
  - after: `3/3` success
- `corner_90_turn`
  - before: `2/3` obstacle collisions
  - after: `3/3` success
- `start_near_obstacle`
  - before: `3/3` obstacle collisions
  - after: `2/3` success, `1/3` collision
- `narrow_passage`
  - before: `3/3` terminated without collision
  - after: unchanged (`3/3` max-steps / no success)
- `symmetry_ambiguous_choice`
  - before: `3/3` terminated without collision
  - after: unchanged (`3/3` max-steps / no success)
- `u_trap_local_minimum`
  - before: `1` collision + `2` terminated
  - after: worse (`3/3` collisions)

## Additional Observation

The rerun also exposed a non-issue-707 benchmark caveat:

- `line_wall_detour` finished as `2/3` collisions and `1/3` max-steps.

That means the added commitment bias is not a free win. It improves the documented ORCA failure
slice substantially, but it also makes at least one wall-detour topology more brittle and worsens
the expected local-minimum behavior in `u_trap_local_minimum`.

## Interpretation

This is a real benchmark-strengthening improvement for ORCA, not just a smoke-run pass:

- the main interpretable failures from issue 707 improved materially,
- the benchmark now executes the full atomic suite cleanly after fixing the route-override path
  resolution bug,
- the remaining caveats are explicit and should stay part of the issue-596 diagnostic story.

The tuned ORCA profile is therefore a credible follow-up artifact for issue 707, but it should be
documented as a tradeoff rather than a universal ORCA upgrade.
