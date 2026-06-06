# Issue #852 Queue-Fill Single-Factor Batch

This directory records compact, reviewable provenance for the completed jobs from the
issue #852 queue-fill batch submitted from `../robot_sf_ll7.worktrees/gse-queue-fill-20260605`.

Status: completed diagnostic training evidence for five jobs. The still-running A30
asymmetric seed-231 job `12751` is intentionally not preserved here as a completed result.

## Submission

- Issue: #852
- Worktree: `../robot_sf_ll7.worktrees/gse-queue-fill-20260605`
- Branch: `gse-queue-fill-20260605`
- Submit/base commit: `5faaa318d609f87730757d7fbda65b799178b5c5`
- Launchers: `SLURM/Auxme/issue_791_reward_curriculum.sl` and
  `SLURM/Auxme/issue_791_asymmetric_critic.sl`
- Intent: fill both `a30` and `l40s` queues with bounded issue-852 single-factor jobs while
  preserving intermediate and completed results.

Local `output/` paths are ignored and non-durable. W&B run URLs below are the durable
external pointers observed from the completed logs; `SHA256SUMS` records hashes for the
source local artifacts used to build this compact evidence bundle.

## Completed Results

| Job | Arm | Seed | Partition | Runtime | Best success | Best collision | Best SNQI | Best step | Final success | W&B |
|----:|-----|-----:|-----------|--------:|-------------:|---------------:|----------:|----------:|--------------:|-----|
| 12750 | reward_curriculum_only 10M | 231 | a30 | 16:20:34 | 0.9143 | 0.0857 | 0.2632 | 6815744 | 0.8571 | [8sq89hfw](https://wandb.ai/ll7/robot_sf/runs/8sq89hfw) |
| 12752 | reward_curriculum_only 1M | 231 | a30 | 01:49:28 | 0.5714 | 0.4143 | -0.8894 | 1000000 | 0.5714 | [13oeq9wv](https://wandb.ai/ll7/robot_sf/runs/13oeq9wv) |
| 12753 | reward_curriculum_only 10M | 1337 | l40s | 08:01:04 | 0.8429 | 0.1571 | -0.0534 | 6291456 | 0.8429 | [qex6vida](https://wandb.ai/ll7/robot_sf/runs/qex6vida) |
| 12754 | asymmetric_critic_only 10M | 1337 | l40s | 09:10:52 | 0.8429 | 0.1571 | -0.0397 | 7864320 | 0.8286 | [s4bdmdlc](https://wandb.ai/ll7/robot_sf/runs/s4bdmdlc) |
| 12755 | asymmetric_critic_only 1M | 1337 | l40s | 01:02:18 | 0.5143 | 0.4857 | -1.0600 | 917504 | 0.5000 | [7fg4zth4](https://wandb.ai/ll7/robot_sf/runs/7fg4zth4) |

## Interpretation

- Strongest completed result: job `12750`, reward-curriculum-only 10M seed 231, with
  best-success `0.9143`, best collision `0.0857`, and best SNQI `0.2632` at step `6815744`.
- The seed-1337 10M pair is close but lower: reward curriculum reached best success `0.8429`;
  asymmetric critic reached best success `0.8429`.
- The 1M jobs are useful low-budget diagnostics only. They are not converged policy evidence.
- Treat this as diagnostic single-factor training evidence, not paper-grade benchmark proof.

## In-Flight Follow-Up

Job `12751` (`gse-852-asym231`) was still running during this preservation pass. It should get
a follow-up evidence entry after completion. Its live W&B run observed in the log was
<https://wandb.ai/ll7/robot_sf/runs/wu9vx2tq>.

## Files

- `summary.json`: structured metrics, W&B pointers, source paths, and artifact classification.
- `SHA256SUMS`: hashes for key ignored local source artifacts and checkpoint zips.

## Caveats

- Raw logs, JSONL episodes, videos, checkpoints, and W&B caches remain ignored local output.
- Known warnings in sibling logs include `uni_campus_big.svg` obstacle self-intersection and
  generated straight-line robot routes for scenarios without route files.
- Use saved best checkpoints for comparison when they differ from the final checkpoint.
