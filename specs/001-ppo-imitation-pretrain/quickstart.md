# Quickstart: Accelerate PPO Training with Expert Trajectories

This guide summarizes how to execute the expert PPO workflow, generate expert trajectories, and launch imitation-enhanced PPO training. All commands assume the uv-managed environment is active (`source .venv/bin/activate`).

## Prerequisites

```bash
# One-time setup
uv sync --all-extras
source .venv/bin/activate

# Verify installation
uv run python -c "from robot_sf.gym_env.environment_factory import make_robot_env; print('Environment factory OK')"
uv run python -c "from stable_baselines3 import PPO; from imitation.algorithms import bc; print('Imitation library OK')"
```

## Pipeline Overview

**1. Train Expert Policy** → **2. Collect Trajectories** → **3. BC Pre-training** → **4. PPO Fine-tuning** → **5. Compare Results**

---

## 1. Train & Approve Expert Policy

### Run Expert Training
```bash
uv run python scripts/training/train_expert_ppo.py \
  --config configs/training/ppo_imitation/expert_ppo.yaml
```

**What happens:**
- Trains PPO policy until convergence criteria are met
- Evaluates performance against success rate and collision rate thresholds
- Writes expert policy manifest to `output/models/expert/<policy_id>.json`
- Saves checkpoint to `output/models/expert/<policy_id>.zip`
- Logs training run manifest to `output/imitation_reports/runs/<run_id>.json`

### Monitor Training (Optional)
```bash
# View TensorBoard logs
uv run tensorboard --logdir output/wandb/
```

### Verify Expert Policy
Check the manifest file to confirm metrics meet requirements:
```bash
cat output/models/expert/<policy_id>.json
# Look for: validation_state = "approved"
# Verify metrics.success_rate >= target and metrics.collision_rate <= threshold
```

---

## 2. Collect Expert Trajectories

### Record Episodes
```bash
uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id expert_traj_v1 \
  --policy-id ppo_expert_v1 \
  --episodes 200 \
  --seeds 42 43 44
```

**What happens:**
- Loads approved expert policy from `output/models/expert/<policy_id>.zip`
- Records 200 episodes using the expert policy
- Saves trajectory dataset to `output/benchmarks/expert_trajectories/expert_traj_v1.npz`
- Validates dataset integrity automatically
- Writes dataset manifest to `output/benchmarks/expert_trajectories/expert_traj_v1.json`

### Validate Dataset (Built-in)
Dataset validation runs automatically during collection. To re-validate:
```bash
uv run python scripts/validation/validate_trajectory_dataset.py --dataset expert_traj_v1
```

### Inspect Trajectories (Optional)
```bash
# Print statistics only
uv run python scripts/validation/playback_trajectory.py \
  --dataset-id expert_traj_v1 \
  --inspect-only

# Visual playback of specific episode
uv run python scripts/validation/playback_trajectory.py \
  --dataset-id expert_traj_v1 \
  --episode 0
```

---

## 3. Pre-Train via Behavioral Cloning

### Run BC Pre-training
```bash
uv run python scripts/training/pretrain_from_expert.py \
  --config configs/training/ppo_imitation/bc_pretrain.yaml
```

**What happens:**
- Loads trajectory dataset from `output/benchmarks/expert_trajectories/`
- Trains policy via behavioral cloning for configured epochs
- Saves pre-trained checkpoint to `output/models/expert/<policy_output_id>.zip`
- Writes training run manifest to `output/imitation_reports/runs/<run_id>.json`

### Example BC Config (configs/training/ppo_imitation/bc_pretrain.yaml)
```yaml
run_id: bc_pretrain_run_001
dataset_id: expert_traj_v1
policy_output_id: bc_policy_v1
bc_epochs: 10
batch_size: 32
learning_rate: 0.0003
random_seeds: [42, 43, 44]
```

---

## 4. Fine-Tune with PPO

### Run PPO Fine-tuning
```bash
uv run python scripts/training/train_ppo_with_pretrained_policy.py \
  --config configs/training/ppo_imitation/ppo_finetune.yaml
```

**What happens:**
- Loads pre-trained policy from `output/models/expert/<pretrained_policy_id>.zip`
- Continues training with PPO for configured timesteps
- Tracks convergence timesteps for sample-efficiency metrics
- Saves fine-tuned checkpoint to `output/models/expert/<run_id>_finetuned.zip`
- Writes training run manifest to `output/imitation_reports/runs/<run_id>.json`

### Example Fine-tuning Config (configs/training/ppo_imitation/ppo_finetune.yaml)
```yaml
run_id: ppo_finetune_run_001
pretrained_policy_id: bc_policy_v1
total_timesteps: 100000
random_seeds: [42, 43, 44]
learning_rate: 0.0001
```

---

## 5. Compare Training Runs

### Generate Comparison Report
```bash
uv run python scripts/tools/compare_training_runs.py \
  --group run_group_2025 \
  --baseline baseline_ppo_run \
  --pretrained pretrained_ppo_run
```

**What happens:**
- Loads training run manifests for baseline and pre-trained runs
- Computes sample-efficiency ratio (target: ≤0.70)
- Calculates convergence timestep reduction
- Compares final metrics between runs
- Saves comparison report to `output/imitation_reports/comparisons/<group_id>_comparison.json`
- Prints human-readable summary

**Example Output:**
```
======================================================================
Training Comparison Report: run_group_2025
======================================================================

Baseline Run: baseline_ppo_run
Pretrained Run: pretrained_ppo_run

Convergence Timesteps:
  Baseline:   1,000,000
  Pretrained:   650,000
  Reduction:    350,000 (35.0%)

Sample-Efficiency Ratio: 0.650
Target (≤0.70): ✓ PASS
======================================================================
```

---

## 4. Artifact Hygiene & Reporting

### Artifact Locations
- **Expert policies**: `output/models/expert/<policy_id>.zip`
- **Expert manifests**: `output/models/expert/<policy_id>.json`
- **Trajectory datasets**: `output/benchmarks/expert_trajectories/<dataset_id>.npz`
- **Dataset manifests**: `output/benchmarks/expert_trajectories/<dataset_id>.json`
- **Training run manifests**: `output/imitation_reports/runs/<run_id>.json`
- **Comparison reports**: `output/imitation_reports/comparisons/<group_id>_comparison.json`

### Quality Checks
All artifacts include:
- Deterministic seeds for reproducibility
- Git commit hash for traceability
- Full command-line invocation
- Timestamp and metadata

### Documentation Updates
- Pipeline workflow: `docs/dev_guide.md` § Imitation Learning Pipeline
- Quick links: `docs/README.md` § Getting Started
- Feature changelog: `CHANGELOG.md`
- This quickstart: `specs/001-ppo-imitation-pretrain/quickstart.md`

---

## Troubleshooting

### Common Issues

**Import Error: imitation not found**
```bash
# Ensure all extras are installed
uv sync --all-extras
```

**Dataset Not Found**
```bash
# Check dataset path resolution
ls output/benchmarks/expert_trajectories/
# Verify dataset_id matches filename
```

**Pre-trained Policy Not Found**
```bash
# Confirm BC pre-training completed successfully
ls output/models/expert/
# Check policy_output_id in BC config matches pretrained_policy_id in fine-tuning config
```

**Dry Run Mode**
All scripts support `--dry-run` flag for testing without actual training:
```bash
uv run python scripts/training/pretrain_from_expert.py --config <config> --dry-run
```

---

## Next Steps

- **Customize scenarios**: Modify `configs/training/ppo_imitation/expert_ppo.yaml`
- **Tune hyperparameters**: Adjust BC epochs, learning rates, batch sizes
- **Scale experiments**: Use multiple seeds for statistical validation
- **Automate pipelines**: Chain scripts in shell/Python orchestration
- **Monitor progress**: Integrate WandB or TensorBoard callbacks

See `docs/dev_guide.md` for complete development workflows and testing practices.
