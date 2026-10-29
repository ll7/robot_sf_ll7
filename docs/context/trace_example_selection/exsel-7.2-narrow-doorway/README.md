# EXSEL-7.2-NARROW-DOORWAY: screening result

**Result: ACCEPTED seed 1179 at screening index 4 of 40.** Both arms' trace reruns match
their screening rows on every match field; trace accepted.

- **Question.** Which development seed is the protocol's narrow-doorway example: the
  first seed in the committed order on which, in `francis2023_narrow_doorway`, orca times
  out and goal ends in an obstacle collision, with both arms inside all their recorded
  frozen bands?
- **Protocol.** ll7/diss `docs/context/trace_example_selection_protocol.yaml`, claim
  EXSEL-7.2-NARROW-DOORWAY (decision EXSEL-TRACE-SELECTION-PROTOCOL-2026-10-28). The
  claim block, rules and acquisition spec are copied verbatim into `contract.yaml.txt`.
  The original `contract.yaml` was pushed in commit
  3e60a0924303eaa639b1871fb9bd71743306a80e before the first screening episode.
- **Source commit.** Unmodified 0.0.8 release code at
  373dbfde4f39667cf9e8732dabe7df5118bdeab1 (branch `exscr1-exec-373dbfde`), release
  campaign config, release scenario matrix, release orca and goal planner entries,
  authored horizon 400 steps, dt 0.1 s. Tier: development-seed diagnostic, new
  current-version execution; not release evidence.
- **Base rate the caption must quote.** Frozen pattern **30/30** (release episodes of
  `francis2023_narrow_doorway` on which orca times out and goal has an obstacle
  collision on the same seed; unconditional orca timeouts 30/30 and goal obstacle
  collisions 30/30 per the claim-keyed report). Never a rate read off this trace.

## Screening (all screened seeds and arms; `screening_ledger.csv`)

| index | seed | orca (class; clearance m) | goal (class; clearance m) | verdict |
|---|---|---|---|---|
| 1 | 1178 | T, 400 steps; 2.9721 | C obstacle, 121 steps; 3.7277 | reject: orca and goal clearance below band |
| 2 | 1170 | T, 400 steps; 2.9762 | C obstacle, 120 steps; 3.7062 | reject: goal clearance below band |
| 3 | 1191 | T, 400 steps; 3.0045 | C obstacle, 115 steps; 4.1344 | reject: orca clearance above band |
| 4 | 1179 | T, 400 steps; 2.9832 | C obstacle, 110 steps; 4.2937 | **accepted** |

Every screened row matched its arm's class; rejections are band-only. Exposure steps
were 0 in all eight rows (band [0, 0]). Frozen clearance bands: orca [2.976, 3.002] m,
goal [3.931, 4.443] m, inclusive; time to goal has no band for either arm. Goal's
collisions are recorded as obstacle_collision_count 1 and wall_collisions 1.0, with no
pedestrian collision. Rows are `mode = native`; see `../provenance/fallback_audit.md`.

## Trace (`trace/`)

- `orca_seed1179_step_trace.json.gz` (400 steps) and `goal_seed1179_step_trace.json.gz`
  (110 steps): the rerun rows' `algorithm_metadata.simulation_step_trace` (schema
  simulation-step-trace.v2, dt 0.1) with row identity.
- `trace_check.json`: both reruns match their screening rows on outcome, collision kind,
  termination reason, steps, the three collision counts, near_misses and minimum
  pedestrian separation. Trace versus row: step counts equal; minimum pedestrian
  surface clearance equal (difference below 1e-9 m); 0 trace steps below the
  near-miss distance against row near_misses 0. The goal trace's last step records
  an obstacle collision; the orca trace's last step records none.

## Comparison contract (reconstruction)

Same contract as the walkthrough packet: exact equality on every serialized field except
`timestamps.start`, `timestamps.end`, `timing.steps_per_second` and `wall_time_sec`.
Checked with a 1-seed rerun (seed 1178): orca 725 and goal 720 flattened fields, only
those four differ (`../provenance/reconstruction.json`).

In this claim directory, `SHA256SUMS` covers every tracked packet file except
`SHA256SUMS` itself.

## Rerun

As in `../exsel-6.4-measure-walkthrough/README.md`, with this claim's `contract.yaml.txt`.
