# Issue #1028 Corridor-Subgoal Recovery

Date: 2026-05-06

Related issues:

- #884: <https://github.com/ll7/robot_sf_ll7/issues/884>
- #1028: <https://github.com/ll7/robot_sf_ll7/issues/1028>
- #1029: <https://github.com/ll7/robot_sf_ll7/issues/1029>
- #1027: <https://github.com/ll7/robot_sf_ll7/issues/1027>

## Goal

Implement the narrow `corridor_subgoal` primitive after #1027 route-corridor diagnostics, then
validate whether it is safe enough to enable for
`hybrid_rule_v3_fast_progress_static_escape`.

## Implementation

The implementation adds a disabled-by-default `corridor_subgoal` source to
`HybridRuleLocalPlannerAdapter`.

The primitive:

- requires `route_guide_enabled` and live route-corridor geometry,
- fails closed when route geometry is missing or a pedestrian is inside the configured activation
  distance,
- activates only when goal progress is nonnegative and near-zero while route-arc progress is stalled
  or regressing,
- generates route-tangent turn/slow subgoal candidates,
- keeps route-tangent recovery turn-only until the robot is aligned with the corridor tangent,
- preserves the `corridor_subgoal` source during candidate de-duplication,
- scores route-tangent progress, corridor centering, tangent alignment, static-clearance margin, and
  command continuity as separate terms,
- applies a strict static-clearance lock while active so the older static-clearance escape/transit
  exceptions do not become the recovery mechanism,
- adds a configurable static-clearance buffer for `corridor_subgoal` and strict active-mode
  candidate evaluation to cover the observed gap between occupancy-grid clearance and the
  environment's continuous obstacle collision check.

`GridRoutePlannerAdapter.route_geometry()` now also exposes `route_next_world` so the local planner
can score immediate corridor centering against the next route segment rather than only the farther
lookahead waypoint.

## Enablement Decision

Do not enable `corridor_subgoal` for `hybrid_rule_v3_fast_progress_static_escape` from this branch.

Enabled probes were rejected:

- `output/ai/autoresearch/issue_1028_corridor_subgoal_recovery/`
  - permissive activation introduced obstacle collisions on low `111`, low `113`, and medium `111`;
  - low `113` regressed from route-complete success to obstacle collision.
- `output/ai/autoresearch/issue_1028_corridor_subgoal_recovery_tight/` and
  `output/ai/autoresearch/issue_1028_corridor_subgoal_recovery_strict/`
  - tighter activation preserved low `113`, but medium `113` still regressed to obstacle collision;
  - the strict-lock probe showed that a linear `corridor_subgoal` command itself can be unsafe in
    the medium `113` corner.
- `output/ai/autoresearch/issue_1028_corridor_subgoal_turn_buffer_probe/`
  - turn-only alignment plus the static-clearance buffer removed the medium `113` obstacle
    collision regression and preserved low `113` success;
  - it did not recover any target timeout: low `111`, medium `111`, medium `112`, and medium `113`
    still timed out at h500;
  - medium `111` selected `corridor_subgoal` only four times before timing out, while medium `113`
    activated the primitive for `139` steps but rejected it for `static_clearance` `138` times.

The final branch therefore leaves the config default disabled. This satisfies the implementation
slice but does not claim a #884 behavior improvement.

## Final Disabled Validation

Final proof artifacts:

- five target traces:
  `output/ai/autoresearch/issue_1028_corridor_subgoal_disabled_final/*_h500/trace.json`
- nominal gate summary:
  `output/ai/autoresearch/issue_1028_corridor_subgoal_disabled_final/nominal_sanity_h500/summary.json`
- stress gate summary:
  `output/ai/autoresearch/issue_1028_corridor_subgoal_disabled_final/stress_slice_h500/summary.json`

Five target h500 outcomes with the primitive disabled:

| Scenario | Seed | Outcome | Obstacle collisions | `corridor_subgoal` active/selected |
|---|---:|---|---:|---:|
| `classic_merging_low` | 111 | timeout | 0 | 0 / 0 |
| `classic_merging_low` | 113 | route-complete success at step `330` | 0 | 0 / 0 |
| `classic_merging_medium` | 111 | timeout | 0 | 0 / 0 |
| `classic_merging_medium` | 112 | timeout | 0 | 0 / 0 |
| `classic_merging_medium` | 113 | timeout | 0 | 0 / 0 |

Gate results:

| Stage | Episodes | Decision | Success rate | Collision rate | Near-miss rate | Execution mode |
|---|---:|---|---:|---:|---:|---|
| `nominal_sanity` h500 | 18 | pass | 1.0 | 0.0 | 0.2778 | adapter |
| `stress_slice` h500 | 24 | tracked | 1.0 | 0.0 | 0.5000 | adapter |

The planner ran through `HybridRuleLocalPlannerAdapter` in adapter mode, with benchmark
availability reported as available in both gate summaries. No fallback or degraded execution was
counted as success.

## Validation Commands

Focused checks:

```bash
rtk uv run ruff check robot_sf/planner/hybrid_rule_local_planner.py \
  robot_sf/planner/grid_route.py \
  tests/planner/test_hybrid_rule_local_planner.py \
  tests/planner/test_grid_route.py

rtk uv run pytest tests/planner/test_grid_route.py \
  tests/planner/test_hybrid_rule_local_planner.py -q
```

Five target seed command shape:

```bash
rtk env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name <scenario> \
  --seed <seed> \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_1028_corridor_subgoal_disabled_final/<scenario>_<seed>_h500
```

Gate commands:

```bash
rtk env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage nominal_sanity \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_1028_corridor_subgoal_disabled_final/nominal_sanity_h500 \
  --docs-root output/ai/autoresearch/issue_1028_corridor_subgoal_disabled_final/docs_h500 \
  --workers 2

rtk env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage stress_slice \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_1028_corridor_subgoal_disabled_final/stress_slice_h500 \
  --docs-root output/ai/autoresearch/issue_1028_corridor_subgoal_disabled_final/docs_h500 \
  --workers 2
```

## Follow-Up Boundary

This PR should not close #884. The next behavior attempt should either:

- verify the exact executed action against the same continuous static-collision surface used by the
  environment, or
- replace the constant-command primitive with a short-horizon corridor optimizer that verifies the
  route-corridor maneuver as a sequence rather than a single constant command.

Update 2026-05-06: #1034 implemented that follow-up boundary by binding environment obstacle-line
geometry, checking short rollout sequences, and adding the tracked
`hybrid_rule_v3_fast_progress_static_escape_continuous` candidate. See
`docs/context/issue_1034_continuous_corridor_maneuver.md` for the target and gate evidence.
