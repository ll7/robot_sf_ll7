# 050 Planner Family Results

## Families
- prediction_planner
- risk_dwa
- mppi_social
- hybrid_portfolio
- predictive_mppi

## Ranking policy
1. hard-suite success
2. global success
3. hard-suite min distance
4. global min distance

## Status
- 2026-03-05 smoke campaign (`francis2023_robot_overtaking`, 2 candidates):
  - `mppi_social_smoke`: hard/global success `0.0/0.0`, hard/global min distance `1.099/1.703` (rank 1).
  - `risk_dwa_smoke`: hard/global success `0.0/0.0`, hard/global min distance `0.705/1.309` (rank 2).
- Interpretation: initial non-learning planners are wired and measurable; they currently improve clearance more than success on this smoke scenario and need full-matrix sweeps.

- 2026-03-05 full campaign v1 (`configs/benchmarks/portfolio_sweep_grid_v1.yaml`, 6 candidates):
  - rank 1 `prediction_planner_anchor`: hard/global success `0.000/0.076`
  - rank 2 `mppi_social_baseline`: hard/global success `0.000/0.015`
  - ranks 3-6 (`risk_dwa`, `mppi_social_dense_safe`, `hybrid_portfolio_default`): hard/global success `0.000/0.000`
- Interpretation: predictive family clearly dominates success; non-predictive families currently act as safer/clearer but less goal-completing baselines.

- 2026-03-05 iter1 v2 (`configs/benchmarks/portfolio_sweep_grid_v2_iter1.yaml`, 5 candidates):
  - rank 1 `prediction_fast_commit`: hard/global success `0.143/0.091`, hard/global min distance `1.369/1.326`
  - rank 2 `prediction_safe_progress`: hard/global success `0.143/0.076`
  - rank 3 `prediction_anchor_v1`: hard/global success `0.000/0.076`
  - rank 4 `mppi_goal_drive`: hard/global success `0.000/0.015`
  - rank 5 `hybrid_predictive_bias`: hard/global success `0.000/0.000`
- Interpretation: first tuning loop produced a measurable success lift without collapsing clearance, but absolute success is still far below BR-07 target (`>=0.8`), so further predictive-family optimization is required.

- 2026-03-05 iter2 v3 (`configs/benchmarks/portfolio_sweep_grid_v3_iter2.yaml`, 5 candidates):
  - rank 1 `prediction_balanced_guard`: hard/global success `0.143/0.106`, hard/global min distance `1.370/1.377`
  - rank 2 `prediction_safe_progress_anchor`: hard/global success `0.143/0.091`
  - rank 3 `prediction_fast_commit_anchor`: hard/global success `0.143/0.076`
  - rank 4 `prediction_ttc_shield`: hard/global success `0.000/0.091`
  - rank 5 `hybrid_predictive_shield`: hard/global success `0.000/0.015`
- Interpretation: iter2 improved both global success and collision profile over iter1 champion. Predictive variants remain dominant; hybrid still underperforms.

- 2026-03-05 iter5 v5 focused hybrid sweep (`output/tmp/portfolio_sweep_grid_v5_hybrid_focus.yaml`, 5 candidates):
  - rank 1 `prediction_balanced_guard_v5`: hard/global success `0.143/0.091`
  - rank 2 `prediction_safe_progress_v5`: hard/global success `0.143/0.091`
  - rank 3 `hybrid_progressive_v5`: hard/global success `0.000/0.045`
  - rank 4 `hybrid_prediction_orca_guard_v5`: hard/global success `0.000/0.030`
  - rank 5 `hybrid_prediction_dominant_v5`: hard/global success `0.000/0.015`
- Interpretation: hybrid tuning improved hybrid global success vs prior (`0.015 -> 0.045`) but still does not challenge predictive variants. Iter2 champion (`prediction_balanced_guard` global `0.106`) remains unbeaten.

- 2026-03-05 horizon sensitivity (`prediction_planner_camera_ready`, classic_interactions matrix):
  - horizon `100`: success `0.000` (collision `18`, max_steps `48`)
  - horizon `120`: success `0.076` (collision `22`, max_steps `39`)
  - horizon `140`: success `0.091` (collision `24`, max_steps `36`)
  - horizon `160`: success `0.182` (collision `27`, max_steps `27`)
- Interpretation: timeout budget is a major limiter. Increasing horizon substantially reduces max-step failures and doubles success from `0.091` to `0.182` (120 -> 160), with a collision tradeoff that must be managed.

- 2026-03-05 predictive MPPI hard-suite probes (`output/tmp/planner_portfolio/direct_predictive_mppi_hard_safety.jsonl` and `output/tmp/planner_portfolio/tune_predictive_mppi_hard_20260305/*.jsonl`):
  - `predictive_mppi current_safe`: hard success `0.000`, termination `6 max_steps / 1 collision`, mean min distance `1.528`
  - `predictive_mppi relaxed_progress`: hard success `0.000`, termination `6 max_steps / 1 collision`, mean min distance `1.523`
- Interpretation: sequence optimization over the learned predictor can suppress some immediate collisions, but it currently pays for that with severe timeout inflation and high runtime. It is not competitive with the predictive anchor and should remain experimental only.
