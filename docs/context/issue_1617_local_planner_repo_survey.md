# Issue #1617 Local-Planner Repository Survey

Date: 2026-05-29

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/1617>

Evidence manifest:
[`evidence/issue_1617_local_planner_repo_survey.json`](evidence/issue_1617_local_planner_repo_survey.json)

## Scope

This note surveys recent or still-active local-planner, social-navigation, and dynamic-obstacle
repositories for ideas that can be translated into Robot SF-native diagnostics, configs, or
benchmark slices. It is not an upstream integration plan. Repositories are treated as inspiration
unless a later issue proves licensing, data, dependency, and benchmark-contract fit.

## Selection Rules

Included repositories had public GitHub metadata and enough README, release, or topic context to
identify a transferable idea. I preferred repositories that were updated recently, released recent
planner work, or addressed local planning under dynamic/social constraints.

Excluded repositories when the likely next step would be an upstream code import, a ROS-only
adapter, or a benchmark claim without a Robot SF-native validation path.

## Survey Table

| Repository | Observed state | Transferable idea | Robot SF fit |
| --- | --- | --- | --- |
| [`tud-amr/mpc_planner`](https://github.com/tud-amr/mpc_planner) | Apache-2.0; `main` at `3002da6`; updated 2026-05-26; v1.1.0 SH-MPC release published 2025-01-09. | Topology-aware MPC diagnostics for distinct route hypotheses around dynamic obstacles; chance-constrained handling of non-Gaussian obstacle uncertainty. | High for diagnostics. Use as design inspiration only, not as ROS/C++ code import. |
| [`Zhefan-Xu/Intent-MPC`](https://github.com/Zhefan-Xu/Intent-MPC) | MIT; `main` at `f9d050c`; updated 2026-05-27; README frames an RA-L 2025 intent-prediction-driven MPC system for dynamic environments. | Intent-conditioned risk or prediction interfaces. | Medium. UAV framing is not directly portable, but intent-conditioned local risk is relevant. |
| [`TommasoVandermeer/social-jym`](https://github.com/TommasoVandermeer/social-jym) | Apache-2.0; `main` at `212ea77`; updated 2026-05-19; topics and tree show JAX, Gym-style social navigation, ORCA, DWA, SARL-PPO, and vectorized training. | JAX-vectorized social-navigation training environment and policy comparison patterns. | Medium. Good performance reference, but JAX is not a near-term Robot SF dependency. |
| [`sepsamavi/safe-interactive-crowdnav`](https://github.com/sepsamavi/safe-interactive-crowdnav) | MIT; `master` at `c702fb8`; updated 2026-05-13; README describes SICNav T-RO and SICNav-Diffusion RA-L code. | MPC-style safe crowd navigation, DWA comparison, and CrowdSimPlus social-force pedestrians. | High for planner-interface and scenario ideas; direct import requires dependency and benchmark review. |
| [`HauserDong/HomoMPC`](https://github.com/HauserDong/HomoMPC) | MIT; `master` at `e31d7a7`; updated 2026-04-20; README describes homotopy-aware multi-agent navigation via distributed MPC. | Homotopy-aware local/global planning diagnostics. | Medium. Useful support for topology-diagnostic follow-up; ROS multi-agent stack is not directly portable. |
| [`CMU-TBD/SocNavBench`](https://github.com/CMU-TBD/SocNavBench) | MIT; `master` at `2724ea8`; updated 2026-05-12; README describes a grounded simulation testing framework for social navigation. | Scenario and metric structure using recorded pedestrian datasets. | Medium. Reference only until data provenance and redistribution boundaries are reviewed. |
| [`CognitiveAISystems/Dynamic-Neural-Potential-Field`](https://github.com/CognitiveAISystems/Dynamic-Neural-Potential-Field) | Apache-2.0; `main` at `071f010`; updated 2026-05-21; README frames an ICRA 2026 Dyn-NPField system. | Learned differentiable local potential or risk surface injected into an MPC-style planner. | High for a Robot SF-native interface contract before planner integration. |
| [`Arena-Rosnav/arena-rosnav`](https://github.com/Arena-Rosnav/arena-rosnav) | MIT; `master` at `5de9d38`; updated 2026-05-27; topics indicate ROS navigation benchmarking, DRL, PPO, PyTorch, and simulation. | Benchmark/workflow organization for learned navigation stacks. | Medium. README metadata alone is not enough to claim specific planner behavior. |
| [`nkuhzx/ssDRL`](https://github.com/nkuhzx/ssDRL) | Apache-2.0; `main` at `bfa988b`; updated 2026-04-03; README describes social-stressed DRL for crowd-comfort navigation. | Social-stress reward or metric framing for crowd comfort. | Medium. Concept only until Robot SF comfort metrics are audited. |
| [`FK-75/social-navigation-robot`](https://github.com/FK-75/social-navigation-robot) | GitHub license metadata reports `NOASSERTION`; `main` at `e84c1a5`; updated 2026-04-19; README describes Webots social navigation with conservative, neutral, and open proxemic profiles. | Proxemic-profile scenario slice comparing comfort and efficiency. | Medium. No code or asset import without a license decision. |
| [`SPALaboratory/SiT-Dataset`](https://github.com/SPALaboratory/SiT-Dataset) | Apache-2.0; `main` at `b5fa5ae`; updated 2026-05-28; README describes the NeurIPS 2023 SiT Dataset and benchmark for social navigation robots. | Socially interactive pedestrian trajectories and semantic-map references for future scenario staging. | Medium. Audit download and redistribution terms before deriving fixtures. |
| [`rst-tu-dortmund/mpc_local_planner`](https://github.com/rst-tu-dortmund/mpc_local_planner) | GPL-3.0; `master` at `5b4e465`; updated 2026-05-19; older ROS `base_local_planner` MPC plugin. | Classic MPC local-planner decomposition and receding-horizon examples. | Low. Useful reference only; GPL and older code make it unsuitable as a direct dependency. |

## Sampled Entry Points

The survey spot-checked README metadata plus lightweight repository-tree entry points. Examples:

* `tud-amr/mpc_planner`: `mpc_planner/include/mpc_planner/planner.h`,
  `mpc_planner/src/planner.cpp`, and `mpc_planner_dingo/config/guidance_planner.yaml`.
* `Zhefan-Xu/Intent-MPC`: `autonomous_flight/cfg/mpc_navigation/planner_param.yaml`,
  `autonomous_flight/cfg/mpc_navigation/predictor_param.yaml`, and
  `autonomous_flight/include/autonomous_flight/mpcNavigation.cpp`.
* `TommasoVandermeer/social-jym`: `notebooks/train_socialnav_policy.ipynb`,
  `socialjym/envs/socialnav.py`, `socialjym/envs/lasernav.py`,
  `socialjym/policies/sarl_ppo.py`, and `socialjym/policies/dwa.py`.
* `sepsamavi/safe-interactive-crowdnav`: `sicnav/configs/policy.config`,
  `sicnav/policy/campc.py`, `sicnav/policy/dwa.py`,
  `sicnav/utils/mpc_utils/mpc_env.py`, and `crowd_sim_plus/envs/crowd_sim_plus.py`.
* `HauserDong/HomoMPC`: `src/global_planner/copath/include/Voronoi.hpp`,
  `src/global_planner/copath/scripts/HomoCoPath.py`, and
  `src/global_planner/copath/src/generate_homotopic_path_binding.cpp`.
* `CMU-TBD/SocNavBench`: `docs/usage.md`,
  `agents/humans/recorded_human.py`, `agents/humans/datasets/*`,
  `metrics/cost_functions.py`, and `socnav/socnav_renderer.py`.
* `CognitiveAISystems/Dynamic-Neural-Potential-Field`:
  `NPField/config/generate_MPC_config.py`, `NPField/script_d1/NPField_model.py`,
  `NPField/script_d1/mpc_params.py`, and `NPField/script_d1/train_model.py`.
* `Arena-Rosnav/arena-rosnav`: `arena_bringup/configs/benchmark/config.yaml`,
  `arena_bringup/configs/benchmark/contests/allplanners.yaml`, and
  `arena_bringup/configs/training/reward_functions/default.yaml`.
* `nkuhzx/ssDRL`: `crowd_nav/configs/*.config`, `crowd_nav/policy/ssDRL.py`,
  `crowd_nav/train.py`, and `crowd_sim/envs/crowd_sim.py`.
* `FK-75/social-navigation-robot`: `controllers/social_robot/lidar_module.py`,
  `controllers/social_robot/movement_module.py`,
  `controllers/social_robot/testing/compare_profiles.py`, and
  `worlds/dementia_care_world.wbt`.
* `SPALaboratory/SiT-Dataset`: `detection/BEVDepth/configs/_base_/datasets/sit_3d.py`
  and related BEVDepth model/config directories.
* `rst-tu-dortmund/mpc_local_planner`: `mpc_local_planner/cfg/mpc_controller.cfg`,
  `mpc_local_planner/include/mpc_local_planner/controller.h`, and
  `mpc_local_planner/mpc_local_planner_plugin.xml`.

## Follow-Up Issues Opened

* #1674: Evaluate local-planner topology-hypothesis diagnostics.
  <https://github.com/ll7/robot_sf_ll7/issues/1674>
* #1675: Prototype learned local risk-surface planner interface.
  <https://github.com/ll7/robot_sf_ll7/issues/1675>
* #1676: Add proxemic-profile comfort scenario slice.
  <https://github.com/ll7/robot_sf_ll7/issues/1676>
* #1677: Audit SiT Dataset terms for social-navigation scenario staging.
  <https://github.com/ll7/robot_sf_ll7/issues/1677>

## Recommended Order

1. Start with #1674 because it can use existing Robot SF traces and should clarify whether topology
   ambiguity is a real local-planner failure mode before implementing a planner.
2. Run #1675 after the diagnostic need is clearer, because a learned risk surface should have a
   consumer contract before model or adapter work begins.
3. Use #1676 to sharpen comfort-vs-efficiency reporting once metric coverage is understood.
4. Treat #1677 as a data-governance gate before any SiT-derived scenario work.

## Non-Claims

This survey does not claim any external planner is benchmark-ready in Robot SF. It does not import
code, assets, checkpoints, or datasets. Each follow-up must fail closed if licensing, dependency,
artifact, or benchmark-contract requirements are not met.
