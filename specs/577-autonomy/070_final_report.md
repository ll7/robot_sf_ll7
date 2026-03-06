# 070 Final Report (Working Draft)

## Objective
Deliver best-performing planner under v2 semantics with full reproducibility evidence.

## Current state
- Implemented:
  - New planners: `risk_dwa`, `mppi_social`, `hybrid_portfolio`.
  - Benchmark routing + readiness/metadata wiring.
  - New campaign tool: `scripts/validation/run_planner_portfolio_campaign.py`.
  - Measurement hardening: collision-safe checkpoint tags in predictive campaign and dataset degeneracy diagnostics in predictive training.
- Validated:
  - Unit tests for new planners and campaign helpers.
  - Smoke campaign with portfolio script completed and produced summary/report artifacts.
  - Full campaign v1 completed:
    - summary: `output/tmp/planner_portfolio/campaign_full_v1/campaign_summary.json`
    - report: `output/tmp/planner_portfolio/campaign_full_v1/campaign_report.md`
  - Full iter1 campaign completed:
    - summary: `output/tmp/planner_portfolio/campaign_iter1_v2/campaign_summary.json`
    - report: `output/tmp/planner_portfolio/campaign_iter1_v2/campaign_report.md`
  - Failure taxonomy extracted for both runs:
    - `output/tmp/planner_portfolio/campaign_full_v1/failure_taxonomy.json`
    - `output/tmp/planner_portfolio/campaign_iter1_v2/failure_taxonomy.json`
  - Guarded PPO compare completed:
    - `output/tmp/planner_portfolio/guarded_ppo_compare/hard_summary.json`
    - `output/tmp/planner_portfolio/guarded_ppo_compare/global_summary.json`

## Latest measured champion
- Candidate: `guarded_ppo_v3` (from `configs/algos/guarded_ppo_camera_ready.yaml`)
- Performance:
  - hard success: `0.143` (same as plain PPO)
  - global success: `0.242` (vs plain PPO `0.227`)
  - global mean min distance: `0.851` (vs plain PPO `0.788`)
  - pedestrian collisions: `1` (vs plain PPO `7`)
- Decision: keep as current branch champion for next tuning loop.

## New evidence from latest loop (iter5 + sensitivity)
- Iter5 focused hybrid sweep (`output/tmp/planner_portfolio/campaign_iter5_v5_hybrid_focus`):
  - best: `prediction_balanced_guard_v5` with hard/global success `0.143/0.091`
  - best hybrid: `hybrid_progressive_v5` with hard/global success `0.000/0.045`
- Outcome:
  - No candidate beat iter2 champion global success (`0.106`).
  - Hybrid family improved from `0.015` to `0.045` global success but remains clearly below predictive family.

- Horizon sensitivity for prediction planner (`output/tmp/planner_portfolio/horizon_sweep_prediction_20260305_171017/summary.json`):
  - `h=100`: success `0.000`
  - `h=120`: success `0.076`
  - `h=140`: success `0.091`
  - `h=160`: success `0.182`
- Interpretation:
  - `max_steps` budget is a first-order bottleneck for completion.
  - Raising horizon yields meaningful success gains, but also increases collision exposure. Future work should combine adaptive horizon with stronger collision safeguards.

## Experimental predictive sequence optimizer
- Added `predictive_mppi` as a new learned-prediction sequence optimizer:
  - module: `robot_sf/planner/predictive_mppi.py`
  - config: `configs/algos/predictive_mppi_camera_ready.yaml`
  - benchmark wiring: `robot_sf/benchmark/map_runner.py`
- Result from hard-suite probes:
  - `current_safe`: `0/7` success, `6 max_steps`, `1 collision`
  - `relaxed_progress`: `0/7` success, `6 max_steps`, `1 collision`
- Interpretation:
  - The hard safety gate fixed the worst failure mode from the first draft (immediate crossing collisions), but the planner is still not viable as a champion candidate.
  - It is materially slower than the predictive anchor and converts too many episodes into timeouts.
  - Decision: keep in-tree as an experimental family, but stop spending primary tuning budget on it until the predictor or action horizon changes substantially.

## Guarded PPO result
- `guarded_ppo` keeps PPO as the primary action source and only intervenes when a short-horizon rollout predicts unsafe pedestrian or obstacle clearance.
- Global benchmark delta vs plain PPO v3:
  - success: `15/66 -> 16/66`
  - collision terminations: `27 -> 23`
  - max-steps terminations: `24 -> 27`
  - pedestrian collision count: `7 -> 1`
- Interpretation:
  - This is a credible improvement, not noise. The guard trades some aggressive progress for much better safety and still finishes one additional episode overall.
  - The remaining problem is targeted conservatism: `classic_bottleneck_medium` regresses from `3/3` success to `1/3`.
  - A lighter guard recovered that bottleneck scenario but gave back the collision reduction, so the current default remains the better global tradeoff.

## Risk and limitation snapshot
- Success remains far below BR-07 target (`>=0.8`), so this is still early-stage.
- Dominant failure mode is timeout (`max_steps`), not obstacle collision.
- Persistent hardest regimes are bottleneck and dense crossing scenarios.

## Next concrete steps
1. Tune guarded PPO specifically for bottleneck regressions:
   - keep doorway safety win while reducing `classic_bottleneck_medium` over-blocking.
2. Generate video evidence for plain PPO vs guarded PPO on the changed scenarios:
   - `classic_doorway_low`
   - `classic_doorway_high`
   - `classic_bottleneck_medium`
3. Repeat guarded PPO benchmark with a second seed-repeat pass before any promotion.
4. Only after guarded PPO stabilizes, return to predictor/model-side work for a non-policy champion path.
