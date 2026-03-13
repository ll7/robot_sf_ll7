# Benchmark Planner-Family Coverage Matrix

This document maps the current `robot_sf_ll7` planner/config stack to Alyassi-style planner families
for benchmark-facing use. It is intended to prevent manuscript-side repos from overclaiming support
that is only partial, experimental, or still roadmap-only.

Use this matrix conservatively:

- Treat `implemented and benchmarkable` rows as the only rows safe to cite as currently runnable
  benchmark support.
- Treat `implemented but experimental` rows as available for controlled benchmark experiments, but
  not baseline-ready by default.
- Treat `conceptually adjacent only` rows as partial proxies or preparatory infrastructure, not as
  evidence that the corresponding literature family is fully covered.
- Treat `missing` rows as roadmap items only.

The matrix is benchmark-facing. It reflects current entrypoints, config surfaces, readiness guards,
and kinematics/adapters in this repository, not a generic literature survey.

## Matrix

| Alyassi-style family | Coverage status | Planner / config entrypoint | Planner-facing input / observation contract | Kinematics / adapter status | Benchmark note |
| --- | --- | --- | --- | --- | --- |
| Goal-following heuristic baseline | implemented and benchmarkable | `algo=goal`; no planner-specific config path | Structured state: robot pose/velocity/goal/radius, agent pose/velocity/radius, obstacle segments | Native unicycle command output; differential-drive compatible in the frozen paper profile | Baseline-ready; canonical paper baseline |
| Force-based interaction baseline | implemented and benchmarkable | `algo=social_force`; adapter metadata emitted by benchmark runner | Same structured state fields as `goal`; force-based interaction adapter | Adapter emits unicycle commands; differential-drive compatible in the frozen paper profile | Baseline-ready; canonical paper baseline |
| Reciprocal-avoidance interaction baseline | implemented and benchmarkable | `algo=orca`; adapter metadata emitted by benchmark runner | Same structured state fields as `goal`; reciprocal-avoidance adapter | Adapter emits unicycle commands; differential-drive compatible in the frozen paper profile | Baseline-ready; canonical paper baseline, dependency-sensitive to `rvo2` availability |
| End-to-end learned navigation baseline | implemented and benchmarkable | `algo=ppo`; canonical benchmark configs under `configs/training/ppo/` and benchmark algo config under `configs/baselines/ppo_15m_grid_socnav.yaml` | Dict observation keyed to the trained model input contract, for example `occupancy_grid`, `goal_current`, `robot_position`, and optional predictive foresight fields | Native mixed command path with adapter support; benchmark output contract normalizes to unicycle semantics; differential-drive compatible in the frozen paper profile | Benchmarkable, but still treated as experimental in readiness metadata unless provenance and quality-gate requirements are met |
| Safety-aware learned navigation | implemented but experimental | `algo=guarded_ppo`; canonical benchmark config under `configs/algos/guarded_ppo_camera_ready.yaml` | PPO dict observation contract plus short-horizon safety veto metadata | Mixed native/adapter path; differential-drive compatible in current benchmark stack | Useful challenger family, but not baseline-ready and not yet paper-frozen as a primary family |
| Prediction-aware local planning | implemented but experimental | `algo=prediction_planner`; canonical config `configs/algos/prediction_planner_camera_ready.yaml` | Structured state plus predictive-planning parameters such as horizon, clearance, TTC, rollout lattice, and predictive checkpoint id | Adapter emits unicycle commands; differential-drive compatible in the frozen paper profile | Implemented and benchmark-runnable, but still experimental and checkpoint-dependent |
| Prediction-aware sequence optimization | implemented but experimental | `algo=predictive_mppi`; config `configs/algos/predictive_mppi_camera_ready.yaml` | Structured state plus learned prediction rollout terms and short sequence-optimization controls | Adapter emits unicycle commands; differential-drive compatible in principle | Explicit testing-only planner; blocked unless `allow_testing_algorithms: true` |
| Non-learning dynamic-window / sampling local planning | implemented but experimental | `algo=risk_dwa`, `algo=mppi_social`, `algo=hybrid_portfolio`, `algo=stream_gap`, `algo=gap_prediction`; configs under `configs/algos/` | Structured state plus planner-specific local risk, TTC, lattice, or veto parameters | Adapter emits unicycle commands; differential-drive compatible in principle | Implemented for controlled R&D, but guarded as testing-only because benchmark value is still unstable |
| Human-state-aware / crowd-aware deep RL proxies | implemented but experimental | `algo=sacadrl`, `algo=socnav_sampling`; no paper-facing frozen config | Structured state through legacy adapter/model-specific contracts | Adapter emits unicycle commands; differential-drive compatibility depends on adapter path | Present as adapter/model-sensitive legacy baselines, but not part of the frozen paper-facing matrix |
| SocNavBench adapter family | conceptually adjacent only | `algo=socnav_bench`; dependency-sensitive adapter path | External benchmark-style observation/adapter contract | Adapter emits unicycle commands when dependencies are present | Present as a bridge/proxy, not as fully integrated benchmark-facing family support |
| World-model RL challenger family | conceptually adjacent only | DreamerV3 BR-08 configs under `configs/training/rllib_dreamerv3/` | RLlib observation contract (`drive_state + rays` and benchmark-aligned `socnav_struct + occupancy_grid` challenger profiles) | Uses RLlib env-runner stack rather than the PPO rollout stack; no frozen paper-facing planner row yet | Training and parity support exist, but benchmark evidence is still pending; do not claim benchmarked family support yet |
| Diffusion / transformer / multimodal social trajectory planners | missing | none | none | none | No current in-repo benchmark-facing implementation |
| Full MCTS / long-horizon belief-space predictive planning | missing | none | none | none | Not currently implemented; only partial short-horizon predictive planning exists |

## Notes on interpretation

- `baseline-ready` in the benchmark stack currently means `goal`, `social_force`, and `orca`.
- `ppo` is benchmarkable and paper-facing when provenance and quality-gate requirements are satisfied,
  but it is still tracked as `experimental` in the generic readiness catalog because the benchmark
  stack distinguishes generic availability from paper-grade promotion.
- Testing-only planners (`risk_dwa`, `mppi_social`, `predictive_mppi`, `hybrid_portfolio`,
  `stream_gap`, `gap_prediction`) are intentionally guarded by
  `allow_testing_algorithms: true` to prevent accidental inclusion in broad or paper-facing runs.
- Manuscript-facing repos should cite this matrix alongside concrete benchmark artifacts and should
  not infer that a partially related planner family is fully covered unless it appears above as
  `implemented and benchmarkable`.
