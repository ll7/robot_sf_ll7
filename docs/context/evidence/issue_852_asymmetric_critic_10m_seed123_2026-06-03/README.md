# Issue #852 Asymmetric-Critic-Only 10M Seed-123 Run

This directory records compact, reviewable provenance for the issue #852 single-factor
asymmetric-critic-only 10M run submitted through `goal-slurm-experiment`.

Status: completed attribution/diagnostic training evidence. This is not paper-grade evidence until
the local artifacts are promoted or otherwise tied to a durable retrieval path.

## Submission

- Issue: #852
- Job: `12719` (`gse-852-asym10m`)
- Failed predecessor: `12718`, immediate `srun` symbol failure before training
- Partition/QoS: `a30` / `a30-gpu`
- Node: `auxme-imech254`
- Worktree: `../robot_sf_ll7.worktrees/gse-852-asym10m-s123`
- Branch: `gse-852-asym10m-s123`
- Submit commit: `9d5f04df`
- Base commit: `f698470a`
- Config:
  `configs/training/ppo/ablations/single_factor/asymmetric_critic_only_10m_env22_seed123.yaml`
- Launcher: `SLURM/Auxme/issue_791_asymmetric_critic.sl`
- W&B run: <https://wandb.ai/ll7/robot_sf/runs/y59octg5>
- Local Slurm log: gitignored local output from job `12719`; see `SHA256SUMS`
  for source-artifact checksums.
- Synced local artifact root: gitignored job-local benchmark output from job
  `12719`; see `SHA256SUMS` for source-artifact checksums.

Local `output/` paths are ignored and non-durable. W&B is the durable external pointer observed at
completion; do not rely on this worktree-local output root in future checkouts.

## Result Summary

- SLURM accounting: `COMPLETED`, exit code `0:0`
- Runtime: 18h05m05s
- Total timesteps: 10,000,000
- Seed: `123`
- Train throughput: about 157.61 env steps/sec
- Eval schedule: every 524,288 steps plus final 10M evaluation
- Best checkpoint: step 7,864,320 by success rate
- Best success rate: 0.8714
- Best collision rate: 0.1286
- Best SNQI: 0.1241
- Final success rate: 0.8429
- Final collision rate: 0.1571
- Final SNQI: -0.0040
- Mean success rate across eval checkpoints: 0.7279
- Mean collision rate across eval checkpoints: 0.2707

Interpretation: the asymmetric-critic-only 10M seed-123 run is a strong single-factor result. It
does not meet the strict convergence target, but it materially changes the issue-852 attribution
story because it reaches the leader-relevant band without the attention head or reward curriculum.
The final checkpoint remains close to the best checkpoint, although the best checkpoint is better
on success, collision, and SNQI.

## Wrapper Repair

The first submission, job `12718`, failed immediately with:

```text
srun: symbol lookup error: srun: undefined symbol: slurm_msg_set_r_uid
```

The active run used a committed launcher repair that makes `srun` opt-in for
`SLURM/Auxme/issue_791_asymmetric_critic.sl`, matching the already-proven reward-curriculum
wrapper behavior. Job `12719` then ran directly inside the batch allocation and completed.

## Warnings

The log includes recurring warnings that should remain caveats for downstream interpretation:

- `uni_campus_big.svg` obstacle path `obstacle_13` has a self-intersection warning.
- Some scenarios have no robot routes and generated straight-line routes.
- Scenario switching reported observation-space bound mismatches and reused initial bounds for
  compatibility.

## Files

- `summary.json`: compact metrics, paths, and artifact status for this run.
- `single_factor_10m_seed123_triad.json`: compact W&B-backed comparison against the attention-only
  and reward-curriculum-only seed-123 10M siblings.
- `SHA256SUMS`: checksums for key ignored local artifacts.

## Seed-123 10M Triad Comparison

W&B-backed comparison on 2026-06-03 found that the attention-only seed-123 10M arm outperformed
both the reward-curriculum-only and asymmetric-critic-only arms on best success, collision, and
SNQI:

| Job | Arm | Best success | Best collision | Best SNQI | Best step | Final success | W&B |
|----:|-----|-------------:|---------------:|----------:|----------:|--------------:|-----|
| 12233 | attention-only | 0.8857 | 0.1143 | 0.1513 | 9.96M | 0.8714 | [8t3zd8sr](https://wandb.ai/ll7/robot_sf/runs/8t3zd8sr) |
| 12234 | reward-curriculum-only | 0.8714 | 0.1286 | 0.0889 | 8.91M | 0.8429 | [sg0iwfal](https://wandb.ai/ll7/robot_sf/runs/sg0iwfal) |
| 12719 | asymmetric-critic-only | 0.8714 | 0.1286 | 0.1241 | 7.86M | 0.8429 | [y59octg5](https://wandb.ai/ll7/robot_sf/runs/y59octg5) |

Interpretation: the asymmetric-critic-only run remains meaningful single-factor attribution
evidence, especially because it reaches the leader-relevant band without attention or reward
curriculum. However, this comparison does not support spending the next robustness allocation on
`asymmetric_critic_only_10m_env22_seed231`; the better next training candidate is the attention-only
seed-231 replication.
