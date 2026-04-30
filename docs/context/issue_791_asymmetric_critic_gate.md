# Issue 791 Asymmetric Critic Gate

## Goal

Implement the asymmetric-critic slice of issue 791: keep the shared SocNav + grid actor surface,
but add a critic-only privileged state vector built from the live observation payload plus
episode/state metadata.

## What Changed

- `robot_sf/gym_env/robot_env.py`
  - Added `asymmetric_critic` support to the observation space and runtime observation payload.
  - The privileged state is attached as `critic_privileged_state`.
- `robot_sf/training/ppo_policy.py`
  - Added `AsymmetricGridSocNavPolicy` for separate actor/critic feature extractors.
- `scripts/training/train_ppo.py`
  - Added config-driven policy selection and fail-closed validation for incompatible settings.
- `robot_sf/gym_env/environment_factory.py`
  - Threaded `asymmetric_critic` through the robot/image factory path.
- Tests
  - Added coverage for privileged-state extraction, env attachment, and policy selection.
- Config
  - Added `configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_stage1.yaml`.

## Canonical Config

- `configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_stage1.yaml`

## Validation

Run the targeted tests after the code changes are finalized:

```bash
uv run pytest tests/test_grid_socnav_extractor.py -q
uv run pytest tests/test_socnav_env_integration.py -q
uv run pytest tests/training/test_train_expert_ppo_contract.py -q
```

## Follow-Up

- Keep the asymmetric critic limited to the grid + SocNav policy path for now.
- Treat the attention-head slice as a separate follow-up change.

## Promotion Retry Status (2026-04-30)

Job **12190** was the latest long-horizon asymmetric-critic retry from this worktree. It ran on
`pro6000` for about 12h38m and was cancelled at `2026-04-30T10:30:31`.

Evidence:

- SLURM log: `output/slurm/12190-issue791-asymmetric-critic.out`
- Synced partial state:
  `output/slurm/issue791-asymmetric-critic-job-12190/wandb/wandb/run-20260429_215459-p94y7354/run-p94y7354.wandb`
- Scheduler state:
  `12190|robot-sf-policy-asymmetric-critic-pro6000|CANCELLED by 1010|0:0|pro6000|12:37:58`

Interpretation: this is useful startup/progress evidence only. It is **not** completed benchmark or
promotion evidence because no completed run JSON, eval-by-scenario artifact, or checkpoint manifest
was synced under `output/slurm/issue791-asymmetric-critic-job-12190/benchmarks/`.

Next retry should use the eval-aligned 10M config on `a30` or `l40s`, not `pro6000`, so the wall
time is aligned with the observed 10M runtime envelope:

```bash
scripts/dev/sbatch_auxme_issue791.sh \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_asymmetric_critic_promotion_10m_env22_eval_aligned.yaml \
  --job-name robot-sf-issue791-asymcrit \
  SLURM/Auxme/issue_791_asymmetric_critic.sl
```

Current queue boundary on 2026-04-30: `scripts/dev/auxme_partition_status.sh` reported
`slots_left=0` for both `a30` and `l40s`, and the reliable helper refused to submit with
`No admissible partition: user slot limit reached on all candidates.` Re-run the status helper
before submitting; do not add another pending issue-791 job while both QoS profiles are saturated.
