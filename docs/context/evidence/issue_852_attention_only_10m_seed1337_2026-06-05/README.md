# Issue #852 Attention-Only 10M Seed-1337 Run

This directory records compact, reviewable provenance for the issue #852 attention-only 10M
seed-1337 run submitted to complete the three-seed attention-only variance band after seeds 123 and
231 matched on best success/collision.

Status: completed replication/diagnostic training evidence. This is not paper-grade promotion
evidence because the strict convergence target still fails on collision.

## Submission

- Issue: #852
- Job: `12746` (`gse-852-attn1337`)
- Partition/QoS: `a30` / `a30-gpu`
- Node: `auxme-imech172`
- Worktree: `../robot_sf_ll7.worktrees/gse-852-asym10m-s123`
- Branch: `gse-852-asym10m-s123`
- Submit commit: `93a8039a`
- Handoff commit: `53ebea6c`
- Config:
  `configs/training/ppo/ablations/single_factor/attention_only_10m_env22_seed1337.yaml`
- Launcher: `SLURM/Auxme/issue_791_attention_head.sl`
- W&B run: <https://wandb.ai/ll7/robot_sf/runs/rjq8tpim>
- Local Slurm log:
  `output/slurm/12746-issue791-attention-head.out`
- Synced local artifact root:
  `output/slurm/issue791-attention-head-job-12746/`

Local `output/` paths are ignored and non-durable. W&B is the durable external pointer observed at
completion; do not rely on this worktree-local output root in future checkouts.

## Result Summary

- SLURM accounting: `COMPLETED`, exit code `0:0`
- Runtime: 16h44m30s
- Total timesteps: 10,000,000 target; final log table reached 10,317,824 including segment overshoot
- Seed: `1337`
- Train throughput: about 169.59 env steps/sec mean; W&B final `time/fps` about 172.10
- Eval schedule: every 524,288 steps plus final 10M evaluation
- Best checkpoint: final 10M checkpoint by success rate
- Best success rate: 0.8714
- Best collision rate: 0.1286
- Best SNQI: 0.0544
- Best eval return: 21.91
- Final success rate: 0.8714
- Final collision rate: 0.1286
- Final SNQI: 0.0544
- Mean success rate across eval checkpoints: 0.7950
- Mean collision rate across eval checkpoints: 0.2043

Interpretation: this is a weaker third attention-only seed than seeds 123 and 231. It still reaches
the leader-relevant band, but it does not match the prior two seeds (`0.8857` success / `0.1143`
collision) and has notably lower SNQI. The three-seed band supports attention-only as a strong
single-factor mechanism, but not as a converged promotion candidate.

## Three-Seed Attention-Only Band

| Seed | Job | Best success | Best collision | Best SNQI | Best step | W&B |
|-----:|----:|-------------:|---------------:|----------:|----------:|-----|
| 123 | 12233 | 0.8857 | 0.1143 | 0.1513 | 9.96M | [8t3zd8sr](https://wandb.ai/ll7/robot_sf/runs/8t3zd8sr) |
| 231 | 12730 | 0.8857 | 0.1143 | 0.1439 | 10.0M | [28w3leex](https://wandb.ai/ll7/robot_sf/runs/28w3leex) |
| 1337 | 12746 | 0.8714 | 0.1286 | 0.0544 | 10.0M | [rjq8tpim](https://wandb.ai/ll7/robot_sf/runs/rjq8tpim) |

## Warnings

The log includes recurring warnings already seen in sibling runs and still relevant as caveats:

- `uni_campus_big.svg` obstacle path `obstacle_13` has a self-intersection warning.
- Some scenarios have no robot routes and generated straight-line routes.
- Scenario switching reported observation-space bound mismatches and reused initial bounds for
  compatibility.

## Files

- `summary.json`: compact metrics, paths, and artifact status for this run.
- `SHA256SUMS`: checksums for key ignored local artifacts.
