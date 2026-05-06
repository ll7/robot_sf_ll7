# Issue 1024 H500 PPO Retrain

## Scope

Issue: https://github.com/ll7/robot_sf_ll7/issues/1024

This note records the first bounded retrain for issue #1024: one PPO run cloned from the current
promoted issue-791 PPO leader, expanded to the broad all-available scenario surface, and aligned
with the h500 per-scenario horizon recommendations from PR #1025.

## Source Recipe

- Previous PPO leader: `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`
- Registry entry: `model/registry.yaml`
- Source config:
  `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml`
- W&B source run: `ll7/robot_sf/ibo3aqus`
- Recorded leader evidence: `docs/context/issue_791_promotion_campaign_128k_256k.md`

The issue #1024 config keeps the leader's reward curriculum, large-capacity `grid_socnav`
extractor, predictive-foresight env overrides, `num_envs: 22`, and `worker_mode: subproc`. It
changes the training budget to 12M timesteps and changes the scenario surface.

## Scenario Surface

- Training config:
  `configs/training/ppo/ablations/expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env22.yaml`
- Scenario config:
  `configs/scenarios/sets/ppo_all_available_training_v1_h500_schedule.yaml`
- Base scenario surface:
  `configs/scenarios/sets/ppo_all_available_training_v1.yaml`
- Horizon source from PR #1025:
  `configs/policy_search/scenario_horizons_h500.yaml`

`ppo_all_available_training_v1_h500_schedule.yaml` includes the broad issue-856 all-available PPO
surface and applies `scenario_overrides_by_name` to the 48 classic/francis h500 scenarios using the
PR #1025 `recommended_horizon_steps`. The remaining 42 non-h500 entries keep their existing
per-scenario `simulation_config.max_episode_steps` values.

To support this without duplicating the full 90-scenario manifest, `robot_sf/training/scenario_loader.py`
now supports `scenario_overrides_by_name` after include expansion. It fails closed for unknown
targets and for duplicate names only when the duplicated name is one of the override targets.

## Pre-Submit Proof

Commit used for submission: `961ae2f6` (`experiment: add issue 1024 h500 PPO retrain config`).

Commands run:

```sh
uv run ruff format --check robot_sf/training/scenario_loader.py tests/training/test_scenario_loader.py
uv run ruff check robot_sf/training/scenario_loader.py tests/training/test_scenario_loader.py
git diff --check
```

Config-load proof:

- `policy_id=ppo_expert_issue_1024_all_available_h500_schedule_12m_env22`
- `total_timesteps=12000000`
- `num_envs=22`
- `scenario_count=90`
- `h500_verified=48`
- `non_h500_count=42`
- `horizon_min=102 horizon_max=600`

Full pytest was intentionally not run on the Auxme login node; `local.machine.md` reserves heavy
tests/training for SLURM or compute nodes.

## SLURM Run

Partition preflight on 2026-05-06 selected `a30`:

```text
partition=a30 qos=a30-gpu free_gpu=8 pending=0 slots_left=2 score=850
```

Dry-run command:

```sh
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env22.yaml \
  --job-name rsf-1024-h500 \
  --wandb-policy require \
  --dry-run \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Submitted command:

```sh
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env22.yaml \
  --job-name rsf-1024-h500 \
  --wandb-policy require \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Job details:

- SLURM job: `12350`
- Job name: `rsf-1024-h500`
- Partition/QOS: `a30` / `a30-gpu`
- Node: `auxme-imech172`
- Allocation: 24 CPUs, 96G memory, 1 A30 GPU
- Time limit: `1-12:00:00`
- SLURM log: `output/slurm/12350-issue791-reward-curriculum.out`
- Wrapper scratch output root: `/tmp/luttkule/12350/results`

Startup proof from the SLURM log confirms:

```text
policy_id=ppo_expert_issue_1024_all_available_h500_schedule_12m_env22
scenario_config=.../configs/scenarios/sets/ppo_all_available_training_v1_h500_schedule.yaml
total_timesteps=12000000
requested_num_envs=22 num_envs=22 worker_mode=subproc
```

## Follow-Up Boundary

Wait for job `12350` to finish before deciding on further retrains, evaluation promotion, or
registry updates. Treat `output/` and `/tmp/luttkule/12350/results` as worktree-local until the
resulting checkpoint and summaries are promoted through the repository's durable artifact path.
