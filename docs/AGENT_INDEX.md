# Agent Index

Quick pointers for agents and automation working in the Robot SF repo. This index favors
entry points, contracts, and where to change behavior.

## Training Entry Points
- `scripts/training/train_ppo.py` - supported config-driven PPO expert training entrypoint.
- `scripts/training_ppo.py` - removed legacy PPO entry point; fails closed with a migration command.
- `scripts/training_a2c.py` - retired legacy A2C entry point; fails closed with config-first migration guidance.
- `scripts/wandb_ppo_training.py` - retired legacy W&B PPO entry point; use `scripts/training/train_ppo.py` with W&B enabled in config.
- `scripts/training/launch_optuna_expert_ppo.py` - supported config-driven Optuna PPO-imitation launcher.
- `scripts/hparam_opt.py` - retired legacy Optuna entry point; fails closed with migration guidance.
- `scripts/training/collect_expert_trajectories.py` - dataset collection for imitation pipelines.
- `scripts/training/pretrain_from_expert.py` - behavioral cloning pretrain from expert data.
- `scripts/training/train_ppo_with_pretrained_policy.py` - PPO fine-tuning from pretrain.
- `scripts/training_ped_ppo.py` - legacy pedestrian PPO entry point.
- `examples/advanced/16_imitation_learning_pipeline.py` - end-to-end pipeline walkthrough.

## Benchmark Entry Points
- `robot_sf/benchmark/cli.py` - benchmark CLI implementation (console entrypoint).
- `scripts/run_social_navigation_benchmark.py` - run standard benchmark flows.
- `scripts/classic_benchmark_full.py` - full classic interactions benchmark run.
- `scripts/benchmark_planner.py` - planner benchmarking.
- `scripts/benchmark02.py` - retired legacy performance benchmark; fails closed with smoke/worker migration guidance.
- `scripts/validation/performance_smoke_test.py` - maintained headless performance smoke.
- `robot_sf/benchmark/full_classic/orchestrator.py` - programmatic full-classic runner.
- `scripts/tools/run_camera_ready_benchmark.py` - config-driven camera-ready campaign runner; canonical path for paper-facing runs.
- `scripts/tools/policy_analysis_run.py` - policy sweep with per-scenario seed-holdout support.

## Benchmark Analysis & Export
- `scripts/tools/analyze_camera_ready_campaign.py` - post-run campaign diagnostics and summary.
- `scripts/tools/paper_results_handoff.py` - export interval-inclusive paper Results rows with bootstrap CIs from a frozen campaign bundle.
- `robot_sf/benchmark/seed_variance.py` - per-scenario/per-planner seed-variability analysis module (used by paper issue #74 pilot).
- `scripts/tools/benchmark_publication_bundle.py` - package a campaign into a DOI-ready publication bundle.

## Observation Modes
- `ObservationMode.DEFAULT_GYM` - stacked `drive_state` + `rays` observation dict.
- `ObservationMode.SOCNAV_STRUCT` - structured SocNav observations (flattened for SB3).
- `robot_sf/sensor/social_graph_observation.py` - shared SocNav-to-graph adapter for
  graph/social-RL candidate screening; deployment-only and current-timestep by contract.
- Reference: `docs/dev/observation_contract.md`.

## Multi-Robot LiDAR
- Toggle robot-in-robot ray detection with `LidarScannerSettings.detect_other_robots`.
- Detection is enabled by default and affects LiDAR ranges only (no schema key changes).

## Grid & Observation Configuration
- Grid config: `RobotSimulationConfig.grid_config` and `use_occupancy_grid`.
- Include grid in observations: `include_grid_in_observation`.
- Grid metadata keys are flattened with `occupancy_grid_meta_*` prefix.
- Reference: `robot_sf/nav/occupancy_grid.py` and `robot_sf/gym_env/robot_env.py`.

## Force Flags (Pedestrian Physics)
- Static obstacle forces: `peds_have_static_obstacle_forces`.
- Pedestrian-robot repulsion: `peds_have_robot_repulsion` (mirrors `sim_config.prf_config.is_active`).
- Legacy alias: `peds_have_obstacle_forces` (deprecated).

## Map Selection
- Deterministic choice: set `config.map_id` to a key in `config.map_pool`.
- Random choice: leave `map_id=None`.

## Render Representation
- `SimulationView` supports per-entity render mode toggles:
- `robot_render_mode`, `ped_render_mode`, `ego_ped_render_mode` (`circle` or `sprite`).
- Optional sprite paths:
- `robot_sprite_path`, `ped_sprite_path`, `ego_ped_sprite_path`.
- Default fallback remains circle rendering when sprite loading fails.

## Scenario Splits
- Helper: `robot_sf/training/scenario_split.py` (`split_scenarios`).
- CLI: `scripts/tools/split_scenarios.py`.

## Rewards
- Defaults in `robot_sf/gym_env/reward.py`.
- Override by passing `reward_func` into environment factories.

## Artifacts & Output
- Canonical artifact root: `output/` (override with `ROBOT_SF_ARTIFACT_ROOT`).
- Recordings: `output/recordings/`.
- Benchmarks: `output/benchmarks/`.
- Telemetry: `output/telemetry/`.
- Helper utilities: `robot_sf/common/artifact_paths.py`.
