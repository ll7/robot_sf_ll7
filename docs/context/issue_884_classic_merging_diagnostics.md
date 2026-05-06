# Issue #884 Classic Merging Diagnostics

Date: 2026-05-05

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/884>

## Goal

Resolve the remaining `classic_merging_low` and `classic_merging_medium` failures for the
`hybrid_rule_v3_fast_progress_static_escape` policy-search candidate without weakening the hard
static-collision gate or counting fallback/degraded execution as success.

The first pass did not find a PR-safe merge-policy fix. It added reusable decision diagnostics and
recorded two rejected mechanisms so the next attempt could start from concrete trace evidence
rather than repeat the same tuning loop.

A follow-up guarded static-corridor transit mechanism recovered one of the five named seeds and
removed the low-density obstacle collision, while preserving the retained horizon-500 nominal and
stress gates. It is still a partial improvement: four named classic-merging seeds remain timeouts,
so #884 is not closed by this note.

## Diagnostic Change

`HybridRuleLocalPlannerAdapter.last_decision()` now reports:

- `moving_rejection_counts`: rejection reasons restricted to candidates with non-zero linear
  commands,
- `rejection_counts_by_source`: rejection reasons grouped by candidate source such as
  `dynamic_window`, `creep`, `path_follow_0.5m`, or `route_guide`.

The added source-level attribution explains whether a static-corridor stall is caused by every
moving source entering the hard static-clearance band, rather than just reporting aggregate
`static_clearance` counts.

Fresh diagnostics with this change are under
`output/ai/autoresearch/issue_884_diagnostics_after/`. The final step for the four timeout seeds
reports `moving_rejection_counts={"static_clearance": 57}` with source attribution:
`dynamic_window=54`, `creep=1`, `path_follow_0.5m=1`, and `route_guide=1`. The obstacle-collision
seed `classic_merging_low` seed `111` reports `moving_rejection_counts={"static_clearance": 10}`
from `dynamic_window`.

## Baseline Evidence

Existing baseline traces under `output/ai/autoresearch/issue_884_baseline/` reproduce the five
classic-merging failures:

| Scenario | Seed | Baseline outcome | Last dominant rejection |
|---|---:|---|---|
| `classic_merging_low` | 111 | obstacle collision at step `242` | `static_clearance` |
| `classic_merging_low` | 113 | timeout at horizon `500` | `static_clearance` |
| `classic_merging_medium` | 111 | timeout at horizon `500` | `static_clearance` |
| `classic_merging_medium` | 112 | timeout at horizon `500` | `static_clearance` |
| `classic_merging_medium` | 113 | timeout at horizon `500` | `static_clearance` |

The timeout traces end with full-stop commands while moving candidates are rejected by hard static
clearance. The collision trace remains the unsafe side of the same corridor family: it continues
making progress until an obstacle collision.

## Rejected Mechanisms

### Unguarded Static-Corridor Transit

Probe output:
`output/ai/autoresearch/issue_884_corridor_transit_probe/`

Hypothesis: allow a slow command to pass through the conservative static-clearance band when the
robot starts just outside the band, stays above an explicit minimum clearance, and makes local
goal progress.

Result:

- `classic_merging_low` seed `113` recovered route completion.
- `classic_merging_low` seed `111` converted from obstacle collision to timeout.
- `classic_merging_medium` seeds `111` and `112` still timed out.
- `classic_merging_medium` seed `113` regressed from timeout to obstacle collision after repeated
  low-progress creep.

Conclusion: reject the unguarded form. The mechanism needs a recent-progress guard so it can help
through an active corridor passage without indefinitely creeping along the wall after progress has
collapsed.

### Guarded Static-Corridor Transit

Probe output:
`output/ai/autoresearch/issue_884_corridor_transit_guard_probe/`

Hypothesis: keep the bounded static-corridor transit gate, but require recent 3 s progress before
using it. This should preserve the `classic_merging_low` seed `113` corridor recovery and prevent
the late `classic_merging_medium` seed `113` wall-creep collision.

Result on the five named horizon-500 probes:

| Scenario | Seed | Guarded outcome | Change from diagnostic baseline |
|---|---:|---|---|
| `classic_merging_low` | 111 | timeout | obstacle collision converted to timeout |
| `classic_merging_low` | 113 | route-complete success at step `330` | timeout recovered |
| `classic_merging_medium` | 111 | timeout | unchanged |
| `classic_merging_medium` | 112 | timeout | unchanged |
| `classic_merging_medium` | 113 | timeout | unchanged; unguarded collision avoided |

Horizon-500 gate checks:

- `nominal_sanity`: pass, `18/18` successes, `0` collisions, near-miss rate `0.2778`.
- `stress_slice`: tracked, `24/24` successes, `0` collisions, near-miss rate `0.5000`.

Conclusion: keep only the guarded form as a partial safety/progress improvement. It does not resolve
#884 because three medium-density seeds and low seed `111` still fail to complete the route at
horizon `500`.

### Static-Corridor Reorient Promotion

Probe output:
`output/ai/autoresearch/issue_884_static_corridor_reorient/`

Hypothesis: when stalled, no pedestrian is near, the best accepted command is a full stop, and
moving candidates are rejected by static clearance, promote an already accepted rotate-in-place
candidate without relaxing hard safety filters.

Result:

- `classic_merging_low` seed `111` still hit an obstacle collision.
- `classic_merging_low` seed `113` still timed out.
- `classic_merging_medium` seeds `111` and `112` still timed out.
- `classic_merging_medium` seed `113` regressed from timeout to obstacle collision.

Conclusion: reject. Rotation promotion changes the terminal behavior but does not create a safe
route-completing corridor policy.

### Classic-Merging Sampling Overrides

Probe outputs:

- `output/ai/autoresearch/issue_884_sampling_probe/`
- `output/ai/autoresearch/issue_884_angular_sampling_probe/`

Hypothesis: the candidate lattice is too coarse to find a valid command through the narrow static
corridor, so increasing samples might recover a safe command while preserving the hard static gate.

Results:

- `linear_samples: 11` plus `angular_samples: 17` avoided the low-seed obstacle collision but
  converted the five probes into earlier or unchanged static deadlocks with zero successes.
- `angular_samples: 17` alone still produced obstacle collisions on `classic_merging_low` seed
  `111` and `classic_merging_medium` seed `112`, with the other three probes timing out.

Conclusion: reject. Higher-resolution local sampling is not enough; it either stalls in the same
static band or still finds unsafe progress commands.

## Current Conclusion

Issue #884 remains unresolved. The guarded corridor-transit slice is safe enough to preserve because
it removes one obstacle collision and recovers one route-complete seed without regressing the
retained horizon-500 nominal/stress gates, but it is not a closing mechanism. The next credible
implementation needs a route- or corridor-aware policy mechanism that proves actual route
completion on the remaining four classic-merging seeds. The new diagnostics make that next
mechanism easier to evaluate by showing which candidate sources are blocked by static clearance at
each stalled step.

Do not promote the rejected reorientation or sampling overrides as benchmark improvements. They are
worktree-local failed experiments, not durable candidate configs.

Update 2026-05-06: #1028 implemented a guarded `corridor_subgoal` primitive behind disabled
configuration flags. Initial enablement probes regressed obstacle collisions; a later turn-only
plus static-clearance-buffer probe removed that collision regression on the targeted slice but
recovered none of the h500 timeouts. The candidate remains disabled and #884 remains unresolved. See
`docs/context/issue_1028_corridor_subgoal_recovery.md` for the #1029 validation matrix and the
follow-up boundary.
## Consolidated Issue Comment Contract

Date: 2026-05-06

Issue #884 now has enough trace history that another broad constant sweep is not acceptable. The
issue comments and follow-up research comment establish this contract for the next implementation
attempt:

- Keep hard static-collision safety strict. Partial recovery that introduces obstacle-collision
  regressions is not benchmark-strengthening evidence.
- Treat the remaining failures as a route-corner/static-corridor policy problem, not as an ORCA
  replacement task, a global hard-margin relaxation, or a higher-sampling DWA sweep.
- Do the broader research/design pass first, then implement a narrow route-corner or corridor
  mechanism with explicit diagnostics and acceptance criteria.
- Do not hide #884 failures behind `policy_stack_v1` fallback, degraded, or proposal-rejection
  success. Any portfolio use must preserve native proposal-status diagnostics.

Current scope update: as of 2026-05-06, implementation is intentionally deferred out of the active
work goal. The next #884 pass should start with research issue
[#1022](https://github.com/ll7/robot_sf_ll7/issues/1022), then split into deeper
design/implementation work with explicit proof requirements.

The issue body still defines the closing acceptance criteria:

- document the targeted hypothesis before implementation,
- evaluate at least `classic_merging_low` seeds `111` and `113`, plus
  `classic_merging_medium` seeds `111`, `112`, and `113`,
- introduce no new obstacle-collision regressions on nominal/stress gates,
- show that the mechanism improves over the current `130/141` baseline with a full matrix or a
  justified targeted matrix.

### Consolidated Failed Attempts

The issue comments collectively reject these paths:

| Mechanism | Outcome | Decision |
|---|---|---|
| Global or scenario-scoped `static_hard_safety_margin: 0.0` | Unsafe or still blocked by static clearance | Do not retry as a closing path |
| Narrow static-recenter promotion | Converted multiple timeout seeds to obstacle collisions | Rejected |
| `scenario_adaptive_orca_v1` | Avoided some collisions but still timed out target seeds | Not a closing mechanism |
| Static-corridor rotate promotion | Failed target seeds and regressed medium seed `113` to collision | Rejected |
| `linear_samples: 11`, `angular_samples: 17`, or angular-only sampling increase | Still timed out or collided | Rejected |
| Grid-route edge projection | Preserved low seed `113` success but did not improve the five-seed metric | Rejected |
| Route-specific slow static-corridor stall recovery | Timed out all five probes and regressed low seed `113` | Rejected |
| Very-slow unguarded corridor transit | Preserved low seed `113` but collided on medium seeds `112` and `113` | Rejected |
| Guard-band widening to `static_corridor_transit_initial_band: 0.15` | Preserved low seed `113` but collided on medium seeds `111/112/113` | Rejected |
| Existing `hybrid_orca_sampler_v1` and `planner_selector_v1` probes | Collided in representative classic-merging probes | Rejected |

### Research Hypothesis To Implement

The remaining failures appear to come from a local-horizon/reference mismatch around the
classic-merging route corner or corridor transition. The current hybrid-rule planner samples
constant-velocity dynamic-window arcs, direct path-following commands, optional route-guide
commands, stop, creep, and rotate candidates. In the failing corridor geometry, moving commands
either enter the hard static-clearance band or do not make enough route-arc progress, so the score
often favors stopping. The low seed `111` collision is the unsafe mirror of the same family: the
planner can keep making progress without anticipating the corridor corner before static clearance
collapses.

The implementation should therefore reuse the existing occupancy-grid route machinery to build a
small route-corridor subgoal/recovery primitive. It should not introduce a new global planner. The
primitive should be guarded by trace-backed signals:

- stalled or near-stalled `progress_windows["3s"]`,
- moving `static_clearance` rejections dominating the decision,
- source-level rejection spread across `dynamic_window`, `path_follow`, `route_guide`, or `creep`,
- no nearby pedestrian inside the existing slow/stop distance,
- route waypoint or route-corridor geometry available,
- current static clearance still above occupied-cell collision.

For the collision case, the mechanism may also need an anticipatory route-corner trigger so the
planner slows or chooses a corridor-following primitive before the hard static band collapses.

### Diagnostic Fields Requested Before Or With Policy Change

The next trace increment should expose route-corridor geometry instead of only candidate rejection
counts. Useful fields for `last_decision()` and step diagnostics are:

- `route_waypoint_world`,
- `route_corner_distance`,
- `route_tangent_heading`,
- `route_heading_error`,
- `corridor_width_estimate`,
- `corridor_center_clearance`,
- `robot_lateral_offset_to_corridor`,
- `route_arc_progress_1s`,
- `route_arc_progress_3s`,
- `best_corridor_primitive`,
- `corridor_primitive_count`,
- `corridor_primitive_rejection_counts`,
- `corridor_primitive_min_static_clearance`,
- `corridor_primitive_min_dynamic_clearance`,
- `corridor_trigger_reason`.

The first implementation may keep the diagnostic names narrower if the underlying route-corridor
primitive is smaller, but it should report enough geometry and rejection state to prove whether the
route-corridor hypothesis is actually active on the five target seeds.

### Implementation Boundary For The Next Pass

The preferred mechanism is a `corridor_subgoal` candidate source in
`HybridRuleLocalPlannerAdapter`, generated from the existing `GridRoutePlannerAdapter` route
waypoint/path machinery. It should:

- choose a center-biased local subgoal along the routed corridor rather than the raw SVG waypoint,
- generate a small number of slow, bounded corridor-following commands or primitives,
- fail closed if any rollout pose enters an occupied cell or violates the hard static/dynamic
  safety checks,
- score route-arc progress and corridor alignment separately from Euclidean goal progress,
- expose source-level rejection diagnostics for accepted and rejected corridor candidates.

This scope is intentionally smaller than a full trajectory optimizer, state lattice planner, or
control-barrier-corridor implementation. Those remain research references and possible follow-up
directions if the narrow route-corridor primitive cannot satisfy the five-seed proof.

## Validation Commands

Focused code validation:

```bash
rtk uv run ruff check robot_sf/planner/hybrid_rule_local_planner.py \
  tests/planner/test_hybrid_rule_local_planner.py
rtk uv run pytest tests/planner/test_hybrid_rule_local_planner.py -q
```

Rejected reorient probe shape:

```bash
LOGURU_LEVEL=WARNING rtk uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name classic_merging_low \
  --seed 111 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_884_static_corridor_reorient/classic_merging_low_111_h500
```

The same command shape was run for:

- `classic_merging_low` seeds `111`, `113`,
- `classic_merging_medium` seeds `111`, `112`, `113`.

Sampling probes used the same five scenario/seed pairs with output roots
`output/ai/autoresearch/issue_884_sampling_probe/` and
`output/ai/autoresearch/issue_884_angular_sampling_probe/`.

Guarded corridor-transit targeted probes used:

```bash
rtk env LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage full_matrix \
  --scenario-name classic_merging_low \
  --seed 113 \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_884_corridor_transit_guard_probe/classic_merging_low_113_h500
```

The same command shape was run for the five named scenario/seed pairs. Horizon-500 gate checks:

```bash
rtk env LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage nominal_sanity \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_884_corridor_transit_guard_regression/nominal_sanity_h500 \
  --docs-root output/ai/autoresearch/issue_884_corridor_transit_guard_regression/docs_h500 \
  --workers 2

rtk env LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape \
  --stage stress_slice \
  --horizon 500 \
  --output-dir output/ai/autoresearch/issue_884_corridor_transit_guard_regression/stress_slice_h500 \
  --docs-root output/ai/autoresearch/issue_884_corridor_transit_guard_regression/docs_h500 \
  --workers 2
```
