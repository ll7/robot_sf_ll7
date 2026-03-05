# 040 Failure Taxonomy

## Buckets
1. Deadlock / freeze
2. Oscillation / indecision
3. Late avoidance collision
4. Route-tracking loss
5. Over-conservative timeout

## Process
- For each failed run, collect first-failure frame and short clip.
- Assign one primary cause and one secondary cause.
- Link back to candidate config in experiment registry.

## Observed taxonomy (2026-03-05 full + iter1)
- Source artifacts:
  - `output/tmp/planner_portfolio/campaign_full_v1/failure_taxonomy.json`
  - `output/tmp/planner_portfolio/campaign_iter1_v2/failure_taxonomy.json`

### Dominant failure mode
1. Over-conservative timeout (`termination_reason=max_steps`) is the primary global bottleneck across all families.
2. Collision is the secondary mode; when it happens, it is almost entirely pedestrian collision in these runs.
3. Obstacle-collision counts are zero in both campaigns, indicating crowd negotiation and decision latency are the core problems.

### Full-v1 summary
- `prediction_planner_anchor`:
  - hard: `6 collision`, `1 max_steps`
  - global: `23 collision`, `38 max_steps`, `5 success`
- `mppi_social_baseline`:
  - hard: `1 collision`, `6 max_steps`
  - global: `29 collision`, `36 max_steps`, `1 success`
- `risk_dwa` variants and `hybrid_portfolio_default`:
  - hard: `1 collision`, `6 max_steps`
  - global: zero success with mix of collision and timeout.

### Iter1-v2 shifts
- `prediction_fast_commit` improved termination mix:
  - hard: success increased to `1/7` (`0.143`)
  - global: success increased to `6/66` (`0.091`)
- `prediction_safe_progress`:
  - hard success also `1/7`, but global stayed at anchor-level (`5/66`).
- `mppi_goal_drive` and `hybrid_predictive_bias` remained timeout/collision dominated.

### Iter2-v3 shifts (collision-aware tuning)
- `prediction_balanced_guard` delivered the best tradeoff:
  - global success: `7/66` (`0.106`)
  - global total collisions: `5` (all pedestrian), down from iter1 fast (`12`)
  - termination mix: `27 collision`, `32 max_steps`, `7 success`
- `prediction_safe_progress_anchor`:
  - global success: `6/66` (`0.091`)
  - global total collisions: `6`
- `prediction_fast_commit_anchor` regressed to global success `5/66` with collisions unchanged (`12`).

Conclusion: moderate collision/TTC hardening with preserved progress pressure is currently the strongest direction.

### Scenario clusters with persistent failures
- High-frequency zero-success clusters (both campaigns):
  - `classic_bottleneck_low|medium|high`
  - `classic_realworld_double_bottleneck_high`
  - `classic_crossing_low|medium` (and often `high`)
- Interpretation: bottleneck + crossing density remains the key unsolved regime for BR-07.
