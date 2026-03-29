# Issue 708 Main-Based PPO Retrain Campaign

## Goal

Prepare a `main`-based branch that is ready to submit one canonical, from-scratch PPO retraining job
on SLURM and then evaluate the resulting checkpoint against the current promoted PPO baseline.

This branch is intentionally campaign-oriented:
- one canonical training config
- one canonical SLURM submission script
- one explicit post-training evaluation path

It is not a multi-variant sweep branch.

## Canonical Inputs

### Training config

- `configs/training/ppo/expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml`

### Current promoted PPO baseline

- model id: `ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200`
- registry source: `model/registry.yaml`
- benchmark-facing baseline config: `configs/baselines/ppo_issue_576_br06_v2_15m.yaml`

### SLURM entrypoint

- `SLURM/Auxme/issue_708_train_ppo_full.sl`

### Scenario surfaces

- training surface: `configs/scenarios/classic_interactions_francis2023.yaml`
- deterministic promotion eval surface: `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`
  - `70` unique maintained positive scenarios
  - `47` classic/francis benchmark scenarios + `23` maintained atomic scenarios
  - excludes the intentionally invalid validation fixture set
- secondary regression gate: `configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml`

## Locked Campaign Semantics

### Reward

- reward name: `route_completion_v3`
- reward profile: success-priority carry-forward from the strongest recent predictive-foresight PPO line
- primary decision signal remains benchmark-facing `success` + `collision`
- `SNQI` is diagnostic and promotion-supporting, not the override criterion

### Observation contract

- `observation_mode: socnav_struct`
- occupancy grid enabled and included in dict observations
- predictive foresight enabled with `predictive_proxy_selected_v2_full`
- predictive foresight must be applied through the PPO env override path and expose
  `predictive_*` observation keys during a real env reset
- no XL ego-conditioned foresight in this issue

### Robot contract

- differential-drive training only
- `max_linear_speed: 3.0`
- `max_angular_speed: 1.0`
- backwards motion allowed

### Training scope

- from scratch only
- no `resume_model_id`
- no alternate reward branch
- no holonomic branch

## Execution Path

### Local dry-run validation

```bash
uv run python scripts/training/train_ppo.py \
  --config configs/training/ppo/expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml \
  --dry-run \
  --log-level WARNING
```

### SLURM dry-run command

```bash
scripts/dev/sbatch_use_max_time.sh --dry-run SLURM/Auxme/issue_708_train_ppo_full.sl
```

### Canonical SLURM submission

```bash
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/issue_708_train_ppo_full.sl
```

Optional overrides:

```bash
ISSUE708_LOG_LEVEL=WARNING \
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/issue_708_train_ppo_full.sl
```

```bash
AUXME_RESULTS_DIR=/path/to/slurm-results \
scripts/dev/sbatch_use_max_time.sh SLURM/Auxme/issue_708_train_ppo_full.sl
```

## Expected Artifacts

During the SLURM job:
- remote artifact root inside the job: `${ROBOT_SF_ARTIFACT_ROOT}`
- rsynced result root after job exit:
  - default: `output/slurm/issue708-job-<jobid>/`
  - override: `${AUXME_RESULTS_DIR}`

Key outputs to inspect after training:
- expert policy checkpoint:
  - `benchmarks/expert_policies/ppo_expert_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.zip`
- best checkpoint family:
  - `benchmarks/expert_policies/checkpoints/ppo_expert_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch/`
- expert manifest:
  - `benchmarks/expert_policies/ppo_expert_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.json`
- training run manifest:
  - `benchmarks/ppo_imitation/runs/<run_id>.json`
- eval timeline:
  - `benchmarks/ppo_imitation/eval_timeline/<run_id>.json`
- eval by scenario:
  - `benchmarks/ppo_imitation/eval_by_scenario/<run_id>.json`
- performance summary:
  - `benchmarks/ppo_imitation/perf/<run_id>.json`
- W&B logs:
  - `wandb/`

## Post-Training Evaluation

### 1. Policy analysis for the trained checkpoint

Use the final expert checkpoint path from the SLURM output root.

```bash
uv run python scripts/tools/policy_analysis_run.py \
  --training-config configs/training/ppo/expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml \
  --policy ppo \
  --model-path <issue708_checkpoint_path> \
  --seed-set eval \
  --max-seeds 3 \
  --output output/benchmarks/issue708_policy_analysis \
  --video-output output/recordings/issue708_policy_analysis \
  --all
```

### 2. Promotion-grade candidate evaluation against the maintained benchmark surface

This is the canonical comparison command because it resolves the latest W&B candidate checkpoint,
builds the temporary PPO algo-config automatically, runs policy analysis, and runs the benchmark gate.

```bash
uv run python scripts/tools/evaluate_latest_ppo_candidate.py \
  --wandb-entity ll7 \
  --wandb-project robot_sf \
  --wandb-group issue-708-ppo-from-scratch \
  --training-config configs/training/ppo/expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml \
  --seed-set eval \
  --max-seeds 3 \
  --output-root output/benchmarks/issue708_latest_candidate_eval \
  --benchmark-workers 1 \
  --benchmark-horizon 120 \
  --benchmark-dt 0.1 \
  --log-level INFO
```

Outputs from this step:
- `output/benchmarks/issue708_latest_candidate_eval/latest_model_selection.json`
- `output/benchmarks/issue708_latest_candidate_eval/policy_analysis/`
- `output/benchmarks/issue708_latest_candidate_eval/benchmark/`
- `output/benchmarks/issue708_latest_candidate_eval/promotion_report.json`

### 3. Issue-596 atomic regression gate

Run the trained checkpoint directly on the atomic suite as a secondary sanity gate.

```bash
SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python scripts/tools/policy_analysis_run.py \
  --scenario configs/scenarios/sets/atomic_navigation_minimal_full_v1.yaml \
  --policy ppo \
  --model-path <issue708_checkpoint_path> \
  --seed-set eval \
  --max-seeds 1 \
  --output output/benchmarks/issue708_issue596_atomic_gate \
  --video-output output/recordings/issue708_issue596_atomic_gate \
  --all
```

## Promotion Checklist

- [ ] Full SLURM training run completed from the issue-708 config.
- [ ] Best or final checkpoint path is recorded.
- [ ] Policy analysis was run on the resulting checkpoint.
- [ ] Benchmark-gate comparison was run against the maintained classic/francis surface.
- [ ] Issue-596 atomic gate was run as a secondary sanity check.
- [ ] Promotion recommendation is written conservatively using success/collision first and SNQI second.

## Decision Rule

Use this ordering when writing the recommendation:

1. benchmark-facing `success`
2. benchmark-facing `collision`
3. `max_steps` / timeout behavior
4. `SNQI`, comfort, and weakest-scenario diagnostics

Do not promote a checkpoint that is clearly worse on success/collision just because `SNQI` looks better.

## Notes

- deterministic eval now uses the full maintained eval surface with fixed seeds
- with `5` fixed seeds and `70` scenarios, set `evaluation_episodes: 350` so each checkpoint covers
  the full scenario-seed cross product once
- If the latest W&B checkpoint is not the intended issue-708 run, narrow the `evaluate_latest_ppo_candidate.py`
  selection further with `--wandb-name-prefix` or stricter tags.
- If the full run proves unstable or unexpectedly weak, capture that as a follow-up issue instead of broadening
  this branch into a multi-variant tuning campaign.
