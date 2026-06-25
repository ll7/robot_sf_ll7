# Issue #3556 — ScenarioBelief drop-vs-retain through the REAL benchmark runner

**Status:** campaign harness built + a real `stream_gap` bug fixed; the safety contrast is
**`inconclusive_oracle_unsafe`** on the crossing family because the oracle baseline is not near-safe.

## Claim boundary (read first)

- **What landed:** (1) a campaign harness that runs the `oracle` / `uncertain_retained` /
  `uncertain_dropped` contrast through the **real** `robot_sf.benchmark.map_runner.run_map_batch`, and
  (2) a **bug fix that made `stream_gap` actually function in the benchmark runner** (see below).
- **What is NOT yet established:** a clean real-runner safety result. On the `classic_crossing_subset`
  family the oracle baseline collides every episode (`collision_rate 1.0`, `success 0.0`), so there is
  no near-safe headroom to attribute a dropping effect — exactly the #3471 caveat, now in the real env.
- **Uncertainty source:** a configurable field-of-view rule, not a calibrated perception model.

## The bug this uncovered and fixed

While promoting the contrast to the real runner, `stream_gap` was found to be **blind in the
benchmark**: `_extract_state` read the *nested* SOCNAV observation (`obs["robot"]`,
`obs["pedestrians"]`), but `map_runner` feeds a *flat* observation (`robot_position`,
`pedestrians_positions`, `goal_current`, ...). So in every benchmark episode the planner extracted
`robot_pos=[0,0]`, `goal=[0,0]`, `n_peds=0` and drove blind. The belief gate could never act because
no pedestrians were ever seen.

Fix (this PR):
- `robot_sf/planner/stream_gap.py::_extract_state` now accepts **both** the nested SOCNAV observation
  and the flat benchmark observation. This makes `stream_gap` see the robot, goal, and pedestrians in
  `map_runner` for the first time.
- `robot_sf/benchmark/scenario_belief_policy_hook.py` reads the flat observation and writes the
  uncertainty sidecar to `pedestrians_uncertainty` (where the fixed `_extract_state` reads it), so the
  belief gate operates in the real runner.

After the fix, the planner engages pedestrians (per-episode near-miss count fell from ~30 to ~0.4 as
it stopped wandering) and the modes **differentiate**: `uncertain_dropped` shows a different
worst-case clearance than `oracle`/`uncertain_retained`, confirming the gate now drops out-of-FOV
agents in the real runner.

## Result (12 seeds x 2 crossing scenarios, FOV 120°)

| Mode | collision rate | near misses | worst min clearance | success |
| --- | --- | --- | --- | --- |
| `oracle` | 1.0 | 3 | -0.3487 | 0.0 |
| `uncertain_retained` | 1.0 | 3 | -0.3487 | 0.0 |
| `uncertain_dropped` | 1.0 | 3 | -0.3137 | 0.0 |

**Decision: `inconclusive_oracle_unsafe`** — the gate now acts (dropped differs on clearance),
but the oracle baseline collides every episode (`oracle_near_safe: false`), so this contrast cannot
attribute collision or near-miss effects to dropping uncertain agents.

## Reproduce

```bash
uv run --with torch python scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py \
  --out-dir output/issue_3556_belief_mode_campaign \
  --report-json docs/context/evidence/issue_3556_belief_mode_real_runner_2026-06-24/report.json
```

## Next step (the remaining #3556 gap)

Find / author a **near-safe crossing family** (easier geometry, occlusion-bearing so out-of-FOV
agents are safety-relevant) where the oracle baseline clears most episodes. Only then can the
real-runner drop-vs-retain contrast be cleanly classified `revise` vs `retention_dominates`. The
harness and the now-functional `stream_gap` benchmark path make that a config + seed-budget run.
