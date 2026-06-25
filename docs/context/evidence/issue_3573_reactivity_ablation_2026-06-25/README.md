# Issue #3573 — reactive-vs-replay pedestrian-reactivity ablation (real runner)

**Status:** the deferred runs + open-loop replay pedestrian mode are built and run through the real
benchmark runner. Diagnostic-tier result: **disabling pedestrian reactivity to the robot
substantially raises collisions**, i.e. reactive (yielding) pedestrians flatter planners — the
"over-yield" concern #3573 set out to quantify.

## What landed

1. **Open-loop replay (non-reactive) pedestrian mode** in the real runner. `build_env_config`
   (`robot_sf/benchmark/map_runner_env.py`) reads an optional scenario key
   `peds_have_robot_repulsion` and toggles `sim_config.prf_config.is_active`. `True` (default) =
   reactive social-force pedestrians; `False` = robot-response term disabled (pedestrians follow
   social-force dynamics among themselves but do not yield to the robot). Backward-compatible.
2. **Paired ablation campaign** `scripts/benchmark/run_reactivity_ablation_campaign_issue_3573.py`:
   per planner, runs the reactive and replay conditions over identical scenarios + seeds (common
   random numbers) through `map_runner.run_map_batch`, aggregates collision-rate / near-miss-rate /
   mean min-clearance, and feeds the per-planner `ReactivityContrast` into the merged
   `assess_reactivity_ablation` quantifier (#3594).

## Result (`report.json`: `classic_crossing_subset`, goal + orca, 4 seeds, horizon 150, 8 episodes/condition)

| planner | condition | collision-rate | near-miss-rate | mean clearance (m) |
| --- | --- | --- | --- | --- |
| goal | reactive | 0.500 | 0.625 | 0.567 |
| goal | replay | 0.625 | 0.625 | 0.487 |
| orca | reactive | 0.125 | 0.250 | 0.781 |
| orca | replay | 0.500 | 0.625 | 0.520 |

- Disabling reactivity (replay) **raised collisions** (orca 0.125 → 0.50, goal 0.50 → 0.625) and
  **reduced separation** for both planners. With common random numbers the contrast is attributable
  to pedestrian reactivity.
- `mean_replay_collision_inflation = −0.25` (reactive − replay): negative because the **reactive**
  condition is the one that flatters here (yielding pedestrians avoid the robot's intrusions). Under
  the quantifier's grounded-replay sign convention `planners_flattered_by_replay` is therefore empty;
  the *magnitude* of the deltas is the reactivity-sensitivity signal.
- `ranking_is_reactivity_sensitive = False`: orca stays safer than goal in both conditions at this
  small matrix (the rank does not flip, though the magnitudes shift substantially).

## Sign convention & operationalization (read carefully)

"Replay/non-reactive" here = robot→pedestrian force disabled in a **live** social-force sim, **not**
pre-recorded trajectory playback (SocNavBench-style grounded replay, for which the merged quantifier's
"replay flatters" naming was written). With live non-reactive pedestrians, removing reactivity reveals
intrusion hazard rather than hiding it, so flattering accrues to the reactive (over-yielding)
condition. Read the quantifier's `replay_flatters` / `*_inflation` fields with this in mind.

## Caveats

- **Horizon-sensitive.** The robot must reach pedestrian proximity for the effect to register; at
  short horizons (< ~150 steps on this family) the robot has not reached the crossing and the
  contrast is near-null. Use a horizon long enough for the episode to complete (campaign default 300).
- **Small N / diagnostic-tier.** 2 scenarios × 4 seeds. Not paper-grade; a firmer claim needs the
  predeclared matrix at seed-sufficiency budget with claim-card review, and ideally ≥3 planners for a
  rank-reversal test.

## Reproduce

```bash
uv run python scripts/benchmark/run_reactivity_ablation_campaign_issue_3573.py \
  --planners goal orca --seeds 101 102 103 104 --horizon 150 \
  --report-json output/issue_3573_reactivity_campaign/report.json
```

Tests: `tests/benchmark/test_map_runner_env_reactivity_issue_3573.py` (env toggle) and
`tests/benchmark/test_run_reactivity_ablation_campaign_issue_3573.py` (aggregation + end-to-end smoke).
