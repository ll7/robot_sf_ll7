# Quickstart: Accelerate PPO Training with Expert Trajectories

This guide summarises how to execute the expert PPO workflow, generate expert trajectories, and launch imitation-enhanced PPO training. All commands assume the uv-managed environment is active (`source .venv/bin/activate`).

## 1. Train & Approve Expert Policy
1. Sync dependencies: `uv sync`
2. Launch expert training: `uv run python scripts/training/train_expert_ppo.py --config configs/training/expert_ppo.yaml`
3. Monitor TensorBoard: `uv run tensorboard --logdir output/runs/expert_ppo`
4. Evaluate and approve once metrics meet thresholds; record `policy_id` and artefact manifest.

## 2. Collect Expert Trajectories
1. Run recorder: `uv run python scripts/training/collect_expert_trajectories.py --policy-id <policy_id> --episodes 200`
2. Validate dataset: `uv run python scripts/validation/validate_trajectory_dataset.py --dataset output/benchmarks/expert_trajectories/<dataset_id>.npz`
3. Optional playback: `uv run python scripts/validation/playback_trajectory.py --dataset <dataset_id>`

## 3. Pre-Train and Fine-Tune PPO
1. Behavioural cloning warm start: `uv run python scripts/training/pretrain_from_expert.py --dataset <dataset_id> --config configs/training/bc_pretrain.yaml`
2. Online fine-tuning: `uv run python scripts/training/train_ppo_with_pretrained_policy.py --config configs/training/ppo_finetune.yaml`
3. Generate comparison report: `uv run python scripts/tools/compare_training_runs.py --group <run_group_id>`

## 4. Artefact Hygiene & Reporting
- Store models under `output/models/ppo_expert/` and `output/models/ppo_pretrained/`
- Trajectory datasets live under `output/benchmarks/expert_trajectories/`
- Publish summary in `output/reports/ppo_imitation/<timestamp>/summary.md`
- Update `docs/README.md` with links to new documentation pages once merged.
