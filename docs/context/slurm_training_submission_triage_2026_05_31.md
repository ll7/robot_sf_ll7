# SLURM Training Submission Triage 2026-05-31

Date: 2026-05-31
Worktree: `../robot_sf_ll7.worktrees/slurm-training-20260531`
Branch: `slurm-training-20260531`
Base: `origin/main` at `aad6ba12`

## Summary

This pass used the SLURM campaign suitability gate to look for training jobs worth submitting from a
clean latest-main worktree.

No long-running training job remained queued from this pass. Two candidate issue-852 fixed-seed PPO
replicas were briefly submitted, proved the issue-791 wrapper can now bypass the broken `srun`
path, then were canceled after current context showed the same configs already completed as jobs
`12176` and `12177` with recorded metrics.

## Submitted Then Canceled

| Job | Config | Initial status | Final status | Reason |
| --- | --- | --- | --- | --- |
| `12665` | `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_seed231_fixed.yaml` | failed immediately | `FAILED` | Existing wrapper used `srun`; Auxme returned `Unable to confirm allocation` before training. |
| `12666` | `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_seed1337_fixed.yaml` | failed immediately | `FAILED` | Same `srun` allocation-handshake failure. |
| `12667` | seed 231 fixed config | started training startup | `CANCELLED by 1010` | Canceled to avoid duplicate spend after historical completed job `12177` was found. |
| `12668` | seed 1337 fixed config | started training startup | `CANCELLED by 1010` | Canceled to avoid duplicate spend after historical completed job `12176` was found. |

Jobs `12667` and `12668` reached the training startup summary after the wrapper change and were
running directly inside the batch allocation, so the immediate `srun` startup blocker is fixed for
the reward-curriculum wrapper in this worktree.

## Not Submitted

- Issue #1470, #1472, #1474, and #1475 launch-packet validators pass, but the issue text and
  handoff notes still block submission on concrete durable artifact aliases, exact execution
  commits, or concrete training/collection command surfaces.
- Issue #1490 / #1506 predictive-v2 expansion is parent-blocked after the #1543 audit; do not run
  the four-way matrix until a revised predictive-v2 transfer hypothesis is selected.
- Issue #1108 is artifact rescue only. Do not launch fresh training from that issue unless it is
  superseded by a clean rerun issue.
- Issue #852 fixed-seed PPO replicas and the later bounded single-factor queue-fill jobs have
  historical completed SLURM jobs recorded in `docs/context/issue_791_promotion_campaign_128k_256k.md`.

## Validation

- `uv sync --all-extras` completed in the clean worktree.
- `train_ppo.py --help`, `bash -n` for the issue-791 wrapper and submit helpers, and `git diff
  --check` passed before the wrapper commit.
- Launch-packet validators returned `status=valid` for learned-risk, shielded-PPO,
  oracle-imitation, and ORCA-residual packets; they remain blocked by their issue-level artifact
  and command-surface gates.

## Follow-Up

The issue-791 reward-curriculum wrapper fix was committed locally as:

`f32677bb fix: let issue 791 run without srun by default`

Consider opening a small PR for that wrapper fix if issue-791 style jobs may be submitted again.
