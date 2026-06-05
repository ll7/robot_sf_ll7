# Issue #852 Attention-Only 10M Seed-231 Run 2026-06-04

This directory records compact, reviewable provenance for the issue #852 attention-only 10M
seed-231 replication run submitted after the seed-123 10M single-factor triad favored the
attention-only arm.

Status: completed replication/diagnostic training evidence. This is not paper-grade promotion
evidence until the result is compared against the leader-family replicas and final artifact
provenance is reviewed.

## Submission

- Issue: #852
- Job: `12730` (`gse-852-attn231`)
- Partition/QoS: `a30` / `a30-gpu`
- Node: `auxme-imech254`
- Worktree: `../robot_sf_ll7.worktrees/gse-852-asym10m-s123`
- Branch: `gse-852-asym10m-s123`
- Submit commit: `ddcf48a2`
- Config:
  `configs/training/ppo/ablations/single_factor/attention_only_10m_env22_seed231.yaml`
- Launcher: `SLURM/Auxme/issue_791_attention_head.sl`
- W&B run: <https://wandb.ai/ll7/robot_sf/runs/28w3leex>
- Local Slurm log: gitignored local output from job `12730`; see `SHA256SUMS` for
  source-artifact checksums.
- Synced local artifact root: gitignored job-local benchmark output from job `12730`; see
  `SHA256SUMS` for source-artifact checksums.

Local `output/` paths are ignored and non-durable. W&B is the durable external pointer observed at
completion; do not rely on this worktree-local output root in future checkouts.

## Result Summary

- SLURM accounting: `COMPLETED`, exit code `0:0`
- Runtime: 14h57m03s
- Total timesteps: 10,000,000 target; final log table reached 10,317,824 including segment overshoot
- Seed: `231`
- Train throughput: about 191.16 env steps/sec mean; W&B final `time/fps` about 192.66
- Eval schedule: every 524,288 steps plus final 10M evaluation
- Best checkpoint: final 10M checkpoint by success rate
- Best success rate: 0.8857
- Best collision rate: 0.1143
- Best SNQI: 0.1439
- Best eval return: 22.31
- Final success rate: 0.8857
- Final collision rate: 0.1143
- Final SNQI: 0.1439
- Mean success rate across eval checkpoints: 0.8179
- Mean collision rate across eval checkpoints: 0.1800

Interpretation: this is useful replication evidence for the attention-only single-factor arm. It
matches the seed-123 attention-only best success and collision, with slightly lower but still strong
SNQI. It does not meet the strict convergence target because collision remains above 0.05, so it
should not be promoted on its own.

## Comparison To Seed-123 Attention-Only

| Seed | Job | Best success | Best collision | Best SNQI | Best step | W&B |
|-----:|----:|-------------:|---------------:|----------:|----------:|-----|
| 123 | 12233 | 0.8857 | 0.1143 | 0.1513 | 9.96M | [8t3zd8sr](https://wandb.ai/ll7/robot_sf/runs/8t3zd8sr) |
| 231 | 12730 | 0.8857 | 0.1143 | 0.1439 | 10.0M | [28w3leex](https://wandb.ai/ll7/robot_sf/runs/28w3leex) |

## Warnings

The log includes recurring warnings already seen in sibling runs and still relevant as caveats:

- `uni_campus_big.svg` obstacle path `obstacle_13` has a self-intersection warning.
- Some scenarios have no robot routes and generated straight-line routes.
- Scenario switching reported observation-space bound mismatches and reused initial bounds for
  compatibility.

## Files

- `summary.json`: compact metrics, paths, and artifact status for this run.
- `SHA256SUMS`: checksums for key ignored local artifacts.
