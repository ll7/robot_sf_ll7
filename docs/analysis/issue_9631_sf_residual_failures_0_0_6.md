# Social-force residual failures in the frozen 0.0.6 campaign

Claim boundary: descriptive analysis of the frozen release bundle only. No episode was stepped or rerun, and no runtime or frozen artifact was changed. Evidence tier is analysis-only; no causal, planner-general, paper, or benchmark-success claim follows.

Goal-adjacent exclusion status: not separable. Issue #9429 proved its trace-dependent `goal_adjacent_timeout.v1` predicate unavailable for all 2,108 classified/timeout rows because the bundle sets `record_simulation_step_trace=false`. `NA` is not zero, so no episode below can be assigned to, or excluded as, a goal-adjacent timeout. The residual subset is therefore the full non-success set described by retained termination signals, not a positively identified post-exclusion remainder.

## Input identity

- Bundle: `benchmark_0_0_6_s30_h600_20260911_publication_bundle`
- Campaign: `benchmark_0_0_6_s30_h600_20260911`
- Source commit: `31cdfe0361abe2c520117a17f99c1b7a0aba4359`
- Checksum verification: 111 files, 740933980 bytes
- Social-force rows read: 1440
- Row source-commit agreement: 1440/1440

## Termination summary (retained signals)

- Successes (`route_complete`): 51/1440 (0.0354)
- Collisions (`collision_event`): 785/1440 (0.5451)
- Timeouts without collision: 604/1440 (0.4194), of which `terminated`: 554, `max_steps`: 50

Collision subtypes (retained count fields; disjoint in this arm, so the counts below sum to the collision total):

| collision subtype | episodes |
| --- | --- |
| obstacle+wall | 216 |
| pedestrian | 569 |

## Outcomes by scenario family

| scenario family | episodes | success | collision | timeout |
| --- | --- | --- | --- | --- |
| accompanying_peer | 30 | 0 | 30 | 0 |
| blind_corner | 30 | 0 | 30 | 0 |
| bottleneck | 120 | 0 | 60 | 60 |
| circular_crossing | 30 | 0 | 7 | 23 |
| cross_trap | 90 | 0 | 13 | 77 |
| crossing | 30 | 0 | 30 | 0 |
| crowd_navigation | 30 | 0 | 28 | 2 |
| doorway | 90 | 0 | 0 | 90 |
| down_path | 30 | 0 | 30 | 0 |
| entering_elevator | 30 | 0 | 0 | 30 |
| entering_room | 30 | 0 | 0 | 30 |
| exiting_elevator | 30 | 0 | 30 | 0 |
| exiting_room | 30 | 0 | 0 | 30 |
| following_human | 30 | 0 | 30 | 0 |
| frontal_approach | 30 | 3 | 26 | 1 |
| group_crossing | 90 | 0 | 0 | 90 |
| head_on_corridor | 60 | 17 | 39 | 4 |
| intersection_no_gesture | 30 | 0 | 30 | 0 |
| intersection_proceed | 30 | 0 | 30 | 0 |
| intersection_wait | 30 | 0 | 30 | 0 |
| join_group | 30 | 0 | 30 | 0 |
| leading_human | 30 | 1 | 28 | 1 |
| leave_group | 30 | 0 | 30 | 0 |
| merging | 60 | 0 | 0 | 60 |
| narrow_doorway | 30 | 0 | 0 | 30 |
| narrow_hallway | 30 | 0 | 30 | 0 |
| overtaking | 60 | 0 | 5 | 55 |
| parallel_traffic | 30 | 1 | 29 | 0 |
| pedestrian_obstruction | 30 | 0 | 30 | 0 |
| pedestrian_overtaking | 30 | 29 | 1 | 0 |
| perpendicular_traffic | 30 | 0 | 30 | 0 |
| robot_crowding | 30 | 0 | 29 | 1 |
| robot_overtaking | 30 | 0 | 30 | 0 |
| station_platform | 30 | 0 | 10 | 20 |
| t_intersection | 60 | 0 | 60 | 0 |

## Retained progress signals over non-collision timeouts

- Deadlock detector true: 604/604 timeouts with a known deadlock flag (206 true outside timeouts, 0 true on successes)
- `failure_to_progress` steps: count=604, mean=233.9073, median=237.0000, min=15.0000, max=342.0000
- `stalled_time` seconds: count=604, mean=4.6295, median=2.9000, min=0.0000, max=14.4000
- `avg_speed` m/s: count=604, mean=0.6392, median=0.3741, min=0.1337, max=1.5417

## Retained-trace case audit (group-crossing seed 132)

The dissertation records a group-crossing seed-22 stall with about 2.9 m net displacement while the nearest pedestrian remains more than about 6.5 m away, and states the episode 'does not identify what holds the robot back'. The retained rows below are the same seed index under `paper_eval_s30` seeds 111-140 (seed 132); they are timeout cases with the recorded signals shown, not a mechanism identification:

| scenario | status | steps | avg_speed m/s | stalled_time s | failure_to_progress steps | deadlock | ped collisions | obstacle collisions | min clearance m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classic_group_crossing_low | timeout | 500 | 0.3081 | 1.3000 | 308.0000 | True | 0 | 0 | 2.1098 |
| classic_group_crossing_medium | timeout | 500 | 0.3050 | 1.9000 | 307.0000 | True | 0 | 0 | 2.1098 |
| classic_group_crossing_high | timeout | 500 | 0.3050 | 1.9000 | 308.0000 | True | 0 | 0 | 2.1098 |

## Table 7.1 cross-check

The dissertation table behind issue #9631 reports 30 social-force successes out of 1,440 (2.08%). The retained bundle rows record 51 `route_complete` episodes out of 1440 (3.54%). The counting definitions differ (the dissertation cell is a typed aggregate; the bundle field is the runtime completion flag), so neither number is corrected here. The dissertation owner owns reconciling the two definitions; this report uses the retained row fields throughout.

## Competing-explanation verdicts

1. Residual failures are predominantly collision terminations in specific interaction-heavy families: supported as a descriptive concentration. Collisions are 785/1389 of the non-success episodes, and the family table above shows where they concentrate. Whether any collision is itself a goal-adjacent limit-cycle variant is unidentifiable without step traces.
2. Some failures are non-goal-adjacent low-progress timeouts: observed as timeout cases with retained stall signatures (deadlock-true timeouts and the seed-case audit), but 'non-goal-adjacent' cannot be asserted while the goal-adjacent predicate is unavailable.
3. Failures are scenario-specific with no planner-wide mechanism: not excluded. Both collisions and timeouts concentrate in a subset of families, which is compatible with scenario-specific causes and with a planner-wide weakness that only some families trigger.
4. Available release outputs do not preserve enough mechanism signals: confirmed for the goal-adjacent question (no step traces, no force time series) and for any claim requiring per-step command/yaw or force-component series.

Overall: at least one recurring residual signature is connected to recorded signals across many episodes — collision termination concentrated by family, and stall-flagged timeouts — so the result is a descriptive classification, not an unresolved verdict. The causal attribution of either signature remains unestablished, and the goal-adjacent share remains unmeasured.

## Stop-rule compliance

Only the exact pinned bundle, the #9429 report, source/configuration identity, and retained traces were audited. No new trace was captured, no simulator was run, no parameter was swept, and no planner or campaign comparison was performed.

## Reproduction

Run the committed analyzer against the checksummed release bundle:

```bash
uv run python scripts/analysis/analyze_sf_residual_failures_issue_9631.py \
  --bundle-root /path/to/benchmark_0_0_6_s30_h600_20260911_publication_bundle \
  --case-seed 132 \
  --output docs/analysis/issue_9631_sf_residual_failures_0_0_6.md
```

Report schema: `issue_9631_sf_residual_report.v1`.
