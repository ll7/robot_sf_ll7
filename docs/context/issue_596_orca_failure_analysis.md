# Issue 596 ORCA Failure Analysis

This note summarizes targeted investigation of the remaining ORCA failures on the
issue-596 atomic navigation suite after installing the benchmark-ready ORCA dependency
via `uv sync --extra orca`.

## Scope

Evaluated policy:
- `socnav_orca`

Scenario set:
- `configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml`

Full-suite artifact root:
- `output/benchmarks/issue596_orca_atomic_full_20260327_rerun`

Targeted visual probes:
- `output/benchmarks/issue596_orca_probe_head_on`
- `output/benchmarks/issue596_orca_probe_narrow`
- `output/benchmarks/issue596_orca_probe_symmetry`
- `output/benchmarks/issue596_orca_probe_start_near`
- `output/benchmarks/issue596_orca_probe_corner`
- `output/benchmarks/issue596_orca_probe_utrap`

Video/frame outputs:
- `output/recordings/issue596_orca_probe_head_on`
- `output/recordings/issue596_orca_probe_narrow`
- `output/recordings/issue596_orca_probe_symmetry`
- `output/recordings/issue596_orca_probe_start_near`
- `output/recordings/issue596_orca_probe_corner`
- `output/recordings/issue596_orca_probe_utrap`

## Summary

ORCA performs strongly on the simple frame-consistency and basic static-obstacle cases,
but the remaining failures are not random. They cluster into two modes:

1. Immediate collision failures
   - `head_on_interaction`
   - `start_near_obstacle`
   - `corner_90_turn`
   - one seed of `u_trap_local_minimum`

2. Low-speed deadlock / indecision failures
   - `narrow_passage`
   - `symmetry_ambiguous_choice`
   - two seeds of `u_trap_local_minimum`

This supports keeping these scenarios in the suite. They probe real local-planner failure
modes rather than arbitrary benchmark harshness.

## Scenario Findings

### `head_on_interaction`

Observed behavior:
- all three seeds end in pedestrian collision
- no obstacle collisions
- multiple near-misses precede contact
- average speed stays high instead of decaying into a cautious resolution

Likely failure mechanism:
- reciprocal head-on avoidance does not produce a stable lateral bypass in this corridor setup
- ORCA keeps a forward-committed line and enters contact rather than yielding enough sideways

Visual evidence:
- contact sheets under
  `output/recordings/issue596_orca_probe_head_on/frames/*/contact_sheet.png`
- the robot and pedestrian remain near the centerline until impact

### `narrow_passage`

Observed behavior:
- all three seeds terminate without collision
- very low average speed
- `low_speed_frac` is roughly `0.75`

Likely failure mechanism:
- ORCA treats the passage as feasible but not comfortably traversable from its local constraints
- instead of committing through the bottleneck, it stalls near the entrance and never recovers

Visual evidence:
- contact sheets under
  `output/recordings/issue596_orca_probe_narrow/frames/*/contact_sheet.png`
- the robot remains almost fixed at the mouth of the passage for most of the episode

### `symmetry_ambiguous_choice`

Observed behavior:
- all three seeds terminate without collision
- low average speed
- high `low_speed_frac`

Likely failure mechanism:
- symmetric left/right alternatives create a tie with no explicit symmetry-breaking rule
- ORCA does not meaningfully commit to either branch and remains near the initial side of the
  central divider

Visual evidence:
- contact sheets under
  `output/recordings/issue596_orca_probe_symmetry/frames/*/contact_sheet.png`
- the robot makes only small local motion and never selects a side around the blocker

### `start_near_obstacle`

Observed behavior:
- all three seeds collide with the obstacle almost immediately
- episodes are extremely short

Likely failure mechanism:
- initial clearance is too small for ORCA's local corrective step to recover safely
- the planner reacts after the geometry is already effectively unrecoverable

Visual evidence:
- contact sheets under
  `output/recordings/issue596_orca_probe_start_near/frames/*/contact_sheet.png`
- the robot clips the adjacent obstacle within the first few frames

### `corner_90_turn`

Observed behavior:
- two seeds collide with the corner obstacle
- one seed succeeds

Likely failure mechanism:
- ORCA can follow the corridor but does not robustly maintain turning clearance at the inside
  corner
- success is seed-sensitive because the exact start pose changes how aggressively it cuts inward

Visual evidence:
- contact sheets under
  `output/recordings/issue596_orca_probe_corner/frames/*/contact_sheet.png`
- failed seeds cut too close to the inside obstacle before the turn is completed

### `u_trap_local_minimum`

Observed behavior:
- one seed collides
- two seeds terminate without collision
- failed non-collision runs are slow and indecisive

Likely failure mechanism:
- ORCA lacks a mechanism for escaping the U-shaped local minimum once it enters the trap mouth
- one run commits into the wall, while the others settle into a low-motion stall inside or near
  the trap

Visual evidence:
- contact sheets under
  `output/recordings/issue596_orca_probe_utrap/frames/*/contact_sheet.png`
- terminated runs remain inside the U geometry instead of reversing or re-planning out

## Investigation Methods That Worked

These repo-native methods were enough to diagnose the failures:

1. Targeted policy analysis with videos and extracted frames

```bash
SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python scripts/tools/policy_analysis_run.py \
  --training-config configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml \
  --scenario configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml \
  --policy socnav_orca \
  --scenario-id narrow_passage \
  --max-seeds 3 \
  --videos \
  --extract-frames \
  --output output/benchmarks/issue596_orca_probe_narrow \
  --video-output output/recordings/issue596_orca_probe_narrow \
  --extract-frames-output output/recordings/issue596_orca_probe_narrow/frames
```

2. Contact-sheet review from `extract_failure_frames.py`
   - the automatically generated `contact_sheet.png` files were sufficient for fast diagnosis

3. Episode-level metrics from `episodes.jsonl`
   - collision vs terminated split
   - `low_speed_frac`
   - `avg_speed`
   - collision subtype counts

## Follow-up Hypotheses

If ORCA should do better on some of these cases, the most plausible tuning or implementation
questions are:

- head-on: lateral bias / time-horizon / neighbor-distance settings may be too weak for decisive
  corridor bypass
- narrow passage: local feasibility check may be overly conservative, causing stall instead of
  commitment
- symmetry: no tie-break mechanism is visible at the behavior level
- start-near-obstacle: initialization may place the robot inside a regime where purely reactive
  recovery is unrealistic
- cornering: local constraint update may under-account for required turning clearance around inner
  corners
- U-trap: expected local-minimum weakness; likely not fixable without stronger global escape logic

## Recommendation

Treat these failing scenarios as valuable issue-596 failure probes, not as evidence that the whole
suite is too difficult.

If a separate `sanity-simple` promotion gate is introduced, it should likely exclude:
- `head_on_interaction`
- `narrow_passage`
- `symmetry_ambiguous_choice`
- `u_trap_local_minimum`
- possibly `corner_90_turn` and `start_near_obstacle`

Those scenarios are still useful in the broader atomic suite because they surface planner-specific
failure structure clearly.

## Issue 707 Follow-Up (2026-03-27)

A later tuned ORCA rerun is recorded in:

- `docs/context/issue_707_orca_atomic_tuning.md`
- `output/benchmarks/camera_ready/issue707_orca_atomic_issue707_orca_atomic_rerun_20260327_222351`

That rerun materially improved several of the failures documented above:

- `head_on_interaction`: improved from `3/3` collisions to `3/3` success
- `corner_90_turn`: improved from `2/3` collisions to `3/3` success
- `start_near_obstacle`: improved from `3/3` collisions to `2/3` success, but still kept a
  `1/3` obstacle collision

But it did **not** solve every issue:

- `narrow_passage`: unchanged stall / no-success behavior
- `symmetry_ambiguous_choice`: unchanged stall / no-success behavior
- `u_trap_local_minimum`: worsened from mixed stall/collision to `3/3` collisions
- `line_wall_detour`: newly regressed to `2/3` obstacle collisions and `1/3` max-steps

The issue-707 changes that drove those tradeoffs were:

- stronger head-on lateral bias and symmetry breaking, which matter because they turn the prior
  centerline collision cases into decisive bypass behavior,
- stall-commit persistence and forward-corridor probing, which matter because they help ORCA keep
  moving through some previously indecisive local interactions,
- obstacle margin and corner-clearance inflation, which matter because they reduce immediate
  clipping at constrained starts and inside corners.

The remaining limitation is that the same local-commit machinery still trades one failure mode for
another on topology-heavy scenes: some cases continue to stall (`narrow_passage`,
`symmetry_ambiguous_choice`), while others become more collision-prone (`u_trap_local_minimum`,
`line_wall_detour`) when the commit bias picks a bad side or stays too aggressive near walls.

Interpret the issue-707 tuning as a real ORCA improvement with explicit tradeoffs, not a blanket
resolution of all issue-596 failure structure. See
`docs/context/issue_707_orca_atomic_tuning.md` for the full regression and benchmark tradeoff
summary.
