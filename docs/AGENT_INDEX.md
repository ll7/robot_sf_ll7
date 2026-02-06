# Agent Index

Quick pointers for agents and automation working in the Robot SF repo. This index favors
entry points, contracts, and where to change behavior.

## Training Entry Points
- `scripts/training/train_expert_ppo.py` - primary structured PPO training (expert + evaluation).
- `scripts/training/collect_expert_trajectories.py` - dataset collection for imitation pipelines.
- `scripts/training/pretrain_from_expert.py` - behavioral cloning pretrain from expert data.
- `scripts/training/train_ppo_with_pretrained_policy.py` - PPO fine-tuning from pretrain.
- `scripts/training_ppo.py` - legacy single-policy PPO entry point.
- `scripts/training_ped_ppo.py` - legacy pedestrian PPO entry point.
- `examples/advanced/16_imitation_learning_pipeline.py` - end-to-end pipeline walkthrough.

## Benchmark Entry Points
- `robot_sf/benchmark/cli.py` - benchmark CLI implementation (console entrypoint).
- `scripts/run_social_navigation_benchmark.py` - run standard benchmark flows.
- `scripts/classic_benchmark_full.py` - full classic interactions benchmark run.
- `scripts/benchmark_planner.py` - planner benchmarking.
- `robot_sf/benchmark/full_classic/orchestrator.py` - programmatic full-classic runner.

## Observation Modes
- `ObservationMode.DEFAULT_GYM` - stacked `drive_state` + `rays` observation dict.
- `ObservationMode.SOCNAV_STRUCT` - structured SocNav observations (flattened for SB3).
- Reference: `docs/dev/observation_contract.md`.

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
