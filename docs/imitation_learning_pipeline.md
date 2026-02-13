# Imitation Learning Pipeline (PPO Pre-training)

[← Back to Documentation Index](./README.md) | [← Back to Development Guide](./dev_guide.md)

## Overview

The project supports accelerating PPO training via behavioral cloning pre-training from expert trajectories. This pipeline enables sample-efficient training by warm-starting agents with expert demonstrations.

## Prerequisites

Install dependencies for the imitation workflow before running BC pre-training:

```bash
uv sync --group imitation
source .venv/bin/activate
```

When invoking commands with `uv run`, include `--group imitation`.

## Pipeline Workflow

**1. Train Expert Policy** → **2. Collect Trajectories** → **3. BC Pre-training** → **4. PPO Fine-tuning** → **5. Compare Results**

---

## Step-by-Step Guide

### Step 1: Train an Expert PPO Policy

Train a high-quality expert policy using standard PPO training:

```bash
# Train expert using configs/training/ppo_imitation/expert_ppo.yaml
uv run python scripts/training/train_expert_ppo.py --config configs/training/ppo_imitation/expert_ppo.yaml
```

**What happens:**
- Trains PPO policy until convergence criteria are met
- Evaluates performance against success rate and collision rate thresholds
- Writes expert policy manifest to `output/benchmarks/expert_policies/<policy_id>.json`
- Saves checkpoint to `output/benchmarks/expert_policies/<policy_id>.zip`

---

### Step 2: Collect Expert Trajectories

Record episodes using the approved expert policy:

```bash
# Record 200 episodes from approved expert policy
uv run python scripts/training/collect_expert_trajectories.py \
  --dataset-id expert_traj_v1 \
  --policy-id ppo_expert_v1 \
  --episodes 200 \
  --seeds 42 43 44
```

**What happens:**
- Loads approved expert policy
- Records specified number of episodes
- Saves trajectory dataset to `output/benchmarks/expert_trajectories/<dataset_id>.npz`
- Validates dataset integrity automatically
- Writes dataset manifest with metadata

---

### Step 3: Pre-train via Behavioral Cloning

Use expert trajectories to pre-train a new policy via behavioral cloning:

```bash
# Pre-train new policy using expert trajectories
uv run --group imitation python scripts/training/pretrain_from_expert.py \
  --config configs/training/ppo_imitation/bc_pretrain.yaml
```

The repository ships with `configs/training/ppo_imitation/bc_pretrain.yaml` as a
starting point—update `dataset_id` and `policy_output_id` before running manual
experiments. The automated example pipeline writes run-specific configs to
`output/tmp/imitation_pipeline/` so you can keep the checked-in YAML focused on
reusable defaults.

**What happens:**
- Loads trajectory dataset
- Trains policy via behavioral cloning (supervised learning)
- Saves pre-trained checkpoint to `output/models/expert/<policy_output_id>.zip`
- Writes training run manifest with metrics

---

### Step 4: Fine-tune with PPO

Continue training the pre-trained policy using PPO for online improvement:

```bash
# Fine-tune the pre-trained policy
uv run python scripts/training/train_ppo_with_pretrained_policy.py \
  --config configs/training/ppo_imitation/ppo_finetune.yaml
```

Like the BC step, a default `ppo_finetune.yaml` lives under
`configs/training/ppo_imitation/`. Adjust `pretrained_policy_id` and
`total_timesteps` for manual runs; the orchestration example generates a
temporary config automatically.

**What happens:**
- Loads pre-trained policy checkpoint
- Continues training with PPO in the environment
- Tracks convergence timesteps for sample-efficiency metrics
- Saves fine-tuned checkpoint
- Writes training run manifest

---

### Step 5: Compare Training Runs

Analyze the sample-efficiency improvements:

```bash
# Generate comparison report (baseline vs pre-trained)
uv run python scripts/tools/compare_training_runs.py \
  --group run_group_2025 \
  --baseline baseline_ppo_run \
  --pretrained pretrained_ppo_run
```

**What happens:**
- Loads training run manifests for both runs
- Computes sample-efficiency ratio (target: ≤0.70)
- Calculates convergence timestep reduction
- Compares final metrics
- Saves comparison report and prints summary

---

## Validation & Utilities

### Validate Trajectory Dataset

```bash
# Validate trajectory dataset quality
uv run python scripts/validation/validate_trajectory_dataset.py --dataset expert_traj_v1
```

Checks dataset integrity, completeness, and quality metrics.

### Playback Trajectories

```bash
# Playback and inspect trajectories visually
uv run python scripts/validation/playback_trajectory.py --dataset-id expert_traj_v1 --episode 0

# Inspect dataset statistics without visualization
uv run python scripts/validation/playback_trajectory.py --dataset-id expert_traj_v1 --inspect-only
```

Visual inspection tool for debugging and understanding expert behavior.

---

## Key Benefits

- **Sample Efficiency**: Target ≤70% of baseline timesteps to convergence
- **Reproducibility**: All runs tracked with manifests, seeds, and metadata
- **Validation**: Automated integrity checks and quality status tracking
- **Traceability**: Complete artifact lineage from expert → dataset → pre-trained → fine-tuned

---

## Artifact Locations

All artifacts are stored under the canonical `output/` directory:

- **Expert policies**: `output/benchmarks/expert_policies/<policy_id>.zip`
- **Expert manifests**: `output/benchmarks/expert_policies/<policy_id>.json`
- **Trajectory datasets**: `output/benchmarks/expert_trajectories/<dataset_id>.npz`
- **Dataset manifests**: `output/benchmarks/expert_trajectories/<dataset_id>.json`
- **Training run manifests**: `output/imitation_reports/runs/<run_id>.json`
- **Comparison reports**: `output/imitation_reports/comparisons/<group_id>_comparison.json`

---

## Configuration Files

Example configuration files are located in `configs/training/ppo_imitation/`:

- `expert_ppo.yaml` - Expert PPO training configuration
- `optuna_expert_ppo.yaml` - Config-first Optuna launcher settings for expert PPO sweeps
- `bc_pretrain.yaml` - Behavioral cloning pre-training configuration
- `ppo_finetune.yaml` - PPO fine-tuning configuration

### Expert PPO overrides

`train_expert_ppo.py` also supports two optional fields in the expert config:

- `ppo_hyperparams`: Overrides Stable-Baselines3 PPO kwargs (e.g., `learning_rate`,
  `n_steps`, `batch_size`, `n_epochs`, `ent_coef`, `clip_range`, `target_kl`).
- `best_checkpoint_metric`: Which eval metric to track for saving the best checkpoint
  (default: `eval_episode_return`; minimize `collision_rate`/`comfort_exposure`, maximize others).
- `snqi_weights`: Optional path to SNQI weights JSON used for canonical SNQI calculation.
- `snqi_baseline`: Optional path to SNQI baseline med/p95 JSON used for normalization.
- `env_factory_kwargs.reward_name`: Optional named reward (`simple`, `punish_action`,
  `snqi_step`) passed to `make_robot_env` for training/eval environments.
- `env_factory_kwargs.reward_kwargs`: Optional keyword arguments for the selected reward.

`snqi_step` metadata notes (for reward fidelity):
- `near_misses`: Exact per-step robot-pedestrian threshold event.
- `force_exceed_events`: Exact per-step count of pedestrians above comfort force threshold.
- `jerk_mean`: Running finite-difference proxy from robot motion (step-level approximation).
- `comfort_exposure`: Per-step normalized proxy (`force_exceed_events / pedestrian_count`).

Best checkpoints are written to:
`output/benchmarks/expert_policies/checkpoints/<policy_id>/<policy_id>_best.zip`

### Optuna sweeps (config-first)

Use the launcher config to keep study settings reproducible:

```bash
uv run python scripts/training/launch_optuna_expert_ppo.py \
  --config configs/training/ppo_imitation/optuna_expert_ppo.yaml
```

Objective modes:
- `best_checkpoint`
- `final_eval`
- `last_n_mean`
- `auc`
- `episodic_snqi` (uses full episode-level SNQI records and falls back to aggregated metrics when logs are missing)

Safety-gated Optuna selection is available via:

- `constraint_collision_rate_max`: Require `collision_rate <= threshold`
- `constraint_comfort_exposure_max`: Optional additional comfort gate
- `constraint_handling`: `penalize` (default) or `prune` infeasible trials

When constraints are active, each trial stores feasibility metadata in Optuna
`user_attrs` and the run logs a feasible/infeasible trial summary.

---

## Detailed Workflow Documentation

For comprehensive step-by-step instructions with examples and troubleshooting, see:

**[Quickstart Guide](../specs/001-ppo-imitation-pretrain/quickstart.md)**

This guide includes:
- Prerequisites and setup
- Detailed command examples
- Configuration file templates
- Troubleshooting common issues
- Advanced usage patterns

---

## Technical Implementation

### Configuration Dataclasses

The pipeline uses typed configuration dataclasses from `robot_sf.training.imitation_config`:

- `BCPretrainingConfig` - Behavioral cloning settings
- `PPOFineTuningConfig` - Fine-tuning parameters

### Metrics and Analysis

Enhanced metrics support from `robot_sf.benchmark.summary`:

- `compute_sample_efficiency_delta()` - Computes efficiency ratios
- `bootstrap_metric_confidence()` - Bootstrap confidence intervals
- `aggregate_training_metrics_with_bootstrap()` - Statistical aggregation

### Integration Testing

Comprehensive integration tests in `tests/integration/test_ppo_pretraining_pipeline.py` validate:

- End-to-end pipeline execution
- Configuration structure correctness
- Artifact creation and lineage
- Comparative metrics computation

---

## Related Documentation

- [Development Guide](./dev_guide.md) - Main development reference
- [Feature Specification](../specs/001-ppo-imitation-pretrain/README.md) - Technical specification
- [Data Model](../specs/001-ppo-imitation-pretrain/data-model.md) - Artifact schemas
- [Implementation Plan](../specs/001-ppo-imitation-pretrain/plan.md) - Design decisions
- [CHANGELOG](../CHANGELOG.md) - Feature 001 release notes

---

## Quick Reference

### Complete Pipeline (One-Liner Sequence)

```bash
# 1. Train expert
uv run python scripts/training/train_expert_ppo.py --config configs/training/ppo_imitation/expert_ppo.yaml

# 2. Collect trajectories
uv run python scripts/training/collect_expert_trajectories.py --dataset-id expert_v1 --policy-id ppo_expert_v1 --episodes 200

# 3. Pre-train with BC
uv run --group imitation python scripts/training/pretrain_from_expert.py --config configs/training/ppo_imitation/bc_pretrain.yaml

# 4. Fine-tune with PPO
uv run python scripts/training/train_ppo_with_pretrained_policy.py --config configs/training/ppo_imitation/ppo_finetune.yaml

# 5. Compare results
uv run python scripts/tools/compare_training_runs.py --group my_experiment --baseline baseline_run --pretrained pretrained_run
```

### Common Validation Commands

```bash
# Validate dataset
uv run python scripts/validation/validate_trajectory_dataset.py --dataset expert_v1

# Inspect statistics
uv run python scripts/validation/playback_trajectory.py --dataset-id expert_v1 --inspect-only

# Visual playback
uv run python scripts/validation/playback_trajectory.py --dataset-id expert_v1 --episode 0
```

---

## Support and Contributions

For questions, issues, or contributions related to the imitation learning pipeline:

1. Check the [quickstart guide](../specs/001-ppo-imitation-pretrain/quickstart.md) for common issues
2. Review [integration tests](../tests/integration/test_ppo_pretraining_pipeline.py) for usage examples
3. Consult the [feature specification](../specs/001-ppo-imitation-pretrain/README.md) for design rationale
4. Open an issue on GitHub with the `imitation-learning` label

---

**Last Updated**: Feature 001 implementation (2025)
