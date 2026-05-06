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
