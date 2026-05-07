# Issue 1024 H500 PPO Retrain

## Scope

Issue: https://github.com/ll7/robot_sf_ll7/issues/1024

This note records the first bounded retrains for issue #1024: PPO runs cloned from the current
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
changes the training budget to 12M timesteps and changes the scenario surface. It also keeps
`randomize_seeds: true`, so each submitted retrain uses randomized scenario-generation seeds.

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

## SLURM Runs

### Job 12350: A30 First Run

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

Status check on 2026-05-06 14:16 CEST:

- Job `12350` was still `RUNNING`.
- The first scheduled evaluation at step `524288` completed.
- Early eval metrics were `success_rate=0.457`, `collision_rate=0.529`, `snqi=-1.44`.
- Live throughput was about `179 fps`, which matches prior a30 issue-791 PPO throughput.
- `MaxRSS` had reached about `80.8G` decimal under the `96G` allocation.

### Job 12351: Cancelled L40S Env22 Replica

The same config was also submitted to `l40s` on 2026-05-06 because the a30 run was expected to take
substantially longer than the prior l40s issue-791 leader.

Partition snapshot before submission:

```text
partition  qos        total_gpu alloc_gpu free_gpu running  pending  user_running slots_left score
a30        a30-gpu    12        6         6        3        0        1            1          625
l40s       l40s-gpu   8         6         2        3        0        0            2          250
```

Dry-run command:

```sh
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env22.yaml \
  --partition l40s \
  --qos l40s-gpu \
  --job-name rsf-1024-h500-l40s \
  --wandb-policy require \
  --dry-run \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Submitted command:

```sh
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env22.yaml \
  --partition l40s \
  --qos l40s-gpu \
  --job-name rsf-1024-h500-l40s \
  --wandb-policy require \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Job details:

- SLURM job: `12351`
- Job name: `rsf-1024-h500-l40s`
- Partition/QOS: `l40s` / `l40s-gpu`
- Node: `auxme-imech091`
- Allocation: 24 CPUs, 96G memory, 1 L40S GPU
- Time limit: `3-00:00:00`
- SLURM log: `output/slurm/12351-issue791-reward-curriculum.out`
- Wrapper scratch output root: `/tmp/luttkule/12351/results`

Startup proof from the SLURM log confirms:

```text
randomize_seeds enabled; ignoring provided seeds for training.
policy_id=ppo_expert_issue_1024_all_available_h500_schedule_12m_env22
scenario_config=.../configs/scenarios/sets/ppo_all_available_training_v1_h500_schedule.yaml
total_timesteps=12000000
requested_num_envs=22 num_envs=22 worker_mode=subproc randomize_seeds=True
```

Outcome:

- Job `12351` was intentionally cancelled after `00:12:25`.
- Reason: replace it with a higher-throughput l40s run using 32 CPUs, 128G memory, and `num_envs: 30`.
- `sacct` state: `CANCELLED by 1010`.

### Job 12352: Active L40S Env30 Replacement

Config:

`configs/training/ppo/ablations/expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env30_l40s.yaml`

This config keeps the same all-available h500 schedule, reward curriculum, large-capacity
`grid_socnav` extractor, 12M-step budget, and randomized training seeds as job `12350`, but raises
`num_envs` from 22 to 30 for a 32-CPU l40s allocation.

Config-load proof:

- `policy_id=ppo_expert_issue_1024_all_available_h500_schedule_12m_env30_l40s`
- `total_timesteps=12000000`
- `num_envs=30`
- `randomize_seeds=True`
- `scenario_count=90`

Dry-run command:

```sh
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env30_l40s.yaml \
  --partition l40s \
  --qos l40s-gpu \
  --job-name rsf-1024-h500-l40s-e30 \
  --wandb-policy require \
  --sbatch-arg --cpus-per-task=32 \
  --sbatch-arg --mem=128G \
  --dry-run \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Submitted command:

```sh
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env30_l40s.yaml \
  --partition l40s \
  --qos l40s-gpu \
  --job-name rsf-1024-h500-l40s-e30 \
  --wandb-policy require \
  --sbatch-arg --cpus-per-task=32 \
  --sbatch-arg --mem=128G \
  SLURM/Auxme/issue_791_reward_curriculum.sl
```

Job details:

- SLURM job: `12352`
- Job name: `rsf-1024-h500-l40s-e30`
- Partition/QOS: `l40s` / `l40s-gpu`
- Node: `auxme-imech091`
- Allocation: 32 CPUs, 128G memory, 1 L40S GPU
- Time limit: `3-00:00:00`
- SLURM log: `output/slurm/12352-issue791-reward-curriculum.out`
- Wrapper scratch output root: `/tmp/luttkule/12352/results`

Startup proof from the SLURM log confirms:

```text
randomize_seeds enabled; ignoring provided seeds for training.
policy_id=ppo_expert_issue_1024_all_available_h500_schedule_12m_env30_l40s
scenario_config=.../configs/scenarios/sets/ppo_all_available_training_v1_h500_schedule.yaml
total_timesteps=12000000
requested_num_envs=30 num_envs=30 worker_mode=subproc randomize_seeds=True
```

Outcome:

- Job `12352` completed successfully on 2026-05-06 after `08:56:43` wall time.
- W&B run: <https://wandb.ai/ll7/robot_sf/runs/n5oxm4rk>
- Final eval at `12,000,000` steps:
  - `success_rate=0.857`
  - `collision_rate=0.129`
  - `snqi=0.0389`
- Best checkpoint by in-run `success_rate`:
  - step `6,291,456`
  - `success_rate=0.900`
  - `collision_rate=0.100`
  - `snqi=0.134`
- Local synced report root:
  `output/slurm/issue791-reward-curriculum-job-12352/benchmarks/ppo_imitation/`

Interpretation:

- This is a useful checkpoint-selection result, not a promotion result.
- The retrain learned the all-available h500 schedule, but its best checkpoint still trails the
  current issue-791 PPO leader's registered best eval on the maintained eval surface
  (`success_rate=0.929`, `collision_rate=0.071`, `SNQI=0.353`).
- The best checkpoint is stronger than the final checkpoint; any follow-up eval should start from
  the `6,291,456`-step best-success checkpoint, not the final model.
- Remaining failures are concentrated in constrained interaction cases, especially doorway,
  merging, double-bottleneck, and narrow-hallway/narrow-doorway scenarios.
- The main insight is that the previously trained issue-791 leader appears to scale better to the
  longer h500 horizons than a fresh broad retrain on long-horizon scenarios. Follow-up work should
  therefore favor evaluating or warm-starting from the previous leader before spending more compute
  on fresh PPO retrains.

## Follow-Up Boundary

Job `12352` is complete and already provides the main issue insight. Job `12350` remains useful as
an A30/env22 comparison point, but the issue/PR does not need to stay open waiting for it unless the
maintainer wants a same-config hardware/control comparison in the same branch.

Treat `output/`, `/tmp/luttkule/12350/results`, and `/tmp/luttkule/12352/results` as worktree-local
until resulting checkpoints and summaries are promoted through the repository's durable artifact
path.
