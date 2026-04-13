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

## Follow-up boundary

The remaining gap looks structural rather than scalar:

- simple probe/gain tuning helped `line_wall_detour` but not `symmetry_ambiguous_choice`,
- route-awareness through the existing `goal.next` signal was not enough to unlock successes,
- removing the direct route-command shortcut from the route-guidance path still left the slice at
  `0/9` success, so that detour was discarded,
- the next credible step is a stronger topology-aware guide for blocked states rather than another
  small gain tweak.

Treat this note as the handoff point for any continuation on `#805`.
