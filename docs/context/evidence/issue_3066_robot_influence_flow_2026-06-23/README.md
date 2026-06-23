# Issue #3066 — Robot Influence on Pedestrian Flow (v0 bounded slice)

**Status:** `diagnostic` · **claim_boundary:** `diagnostic_only` ·
**evidence_tier:** `smoke` · **paper_grade:** `false`

## What this is

A LOCAL, bounded, same-seed campaign that asks whether the robot's *policy*
measurably perturbs nearby pedestrian motion, beyond ordinary seed variance. It
reuses the repository benchmark runner (`robot_sf.benchmark.runner.run_batch`);
it does not reimplement the simulator.

## Reproduce

```bash
uv run python scripts/benchmark/run_robot_influence_flow_slice_issue_3066.py
```

- **git HEAD:** `cbec866be3deb99858bce200f2824000a3c28e6b`
- **Policies (robot):** `social_force` (baseline-safe) vs `orca` — two distinct
  rule-based baselines, CPU-only, no checkpoints / no GPU.
- **Scenarios (corridor/crossing subset of issue_3059 suite):**
  `classic_head_on_corridor_low`, `classic_crossing_low`.
- **Seeds (same across both policies):** `111, 112, 113`.
- **Horizon:** 240 steps. **Total episodes:** 12 (2 policies × 2 scenarios × 3 seeds).

## Result

- **12/12 rows usable, 0 degraded / fail-closed.**
- **Robot-influence signal:** near-vs-far pedestrian reduction
  (`ped_impact_accel_delta_mean`, `ped_impact_turn_rate_delta_mean`) — the
  difference in pedestrian acceleration / turn-rate when near the robot vs far.
- **Flow deltas (orca − social_force):** 3 of 4 powered deltas exceed pooled seed
  variance, suggesting the robot policy does change near-field pedestrian motion.
- **Behavior model:** classic Social-Force pedestrians (fast-pysf backend). No
  control-trajectory cohort comparison was available, so cohort-level
  distributional flow metrics report `unavailable` (honest denominator gap).

### Caveats (why this stays diagnostic-only, not benchmark/real-world)

- Tiny n (3 seeds/cell) and very low pedestrian density → near-field samples are
  sparse; one `social_force` corridor cell shows an outlier accel mean driven by
  a small near-sample denominator. The driver reports this transparently with its
  seed variance rather than discarding it.
- Robot influence is interpreted **separately** from nav performance: in these
  cells `orca` reaches the goal (success=1.0) while `social_force` times out
  (success=0.0). The flow deltas are therefore partly confounded with whether the
  robot is still moving through the crowd — this is a v0 diagnostic, not an
  isolated causal estimate of policy-on-flow.
- No real-world pedestrian-flow, crowd-comfort, fairness, or sim-to-real claim.

## Files

- `report.json` — full per-row metrics, aggregates, flow deltas, classification.
- Durable raw episodes + markdown live under (git-ignored)
  `output/issue_3066_robot_influence/`.
