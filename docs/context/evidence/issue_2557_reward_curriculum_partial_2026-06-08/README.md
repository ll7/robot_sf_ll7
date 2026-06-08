# Issue #2557 Reward-Curriculum Seed Replica Partial Evidence

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2557>

This bundle preserves compact evidence from the first four completed real 10M-step seed-replica
runs for the Issue #2557 reward-curriculum follow-up. It is partial training evidence only, not a
complete seed-batch result: seeds 503, 504, and 507 were still running, and seeds 508-516 were
pending, when this bundle was created on 2026-06-08.

## Claim Boundary

- Evidence tier: partial training evidence.
- Not claimed: full 16-seed batch result, paper-grade sufficiency, or final Issue #2557 conclusion.
- Raw artifacts excluded from git: model zips, episode JSONL, W&B binary logs, and full Slurm logs.
- Durable external artifacts: W&B run pages linked in `seed_summary.csv` and `seed_summary.json`.

## Source Runs

The source worktree was
`/home/luttkule/git/robot_sf_ll7.worktrees/issue-2557-gse-night-seeds` at commit
`0e0f98a4fefe9faedc3eed31f2085af9618914e3`. Each job used
`configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`, `total_timesteps=10000000`,
`dry_run=False`, deterministic evaluation seeds, and the default SNQI implementation.

| Seed | Job | Partition | W&B run | Final success | Final collision | Final SNQI |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 501 | 12767 | a30 | <https://wandb.ai/ll7/robot_sf/runs/xu1ygfhp> | 0.914286 | 0.085714 | 0.263960 |
| 502 | 12768 | a30 | <https://wandb.ai/ll7/robot_sf/runs/usp2vcrl> | 0.914286 | 0.085714 | 0.275673 |
| 505 | 12771 | l40s | <https://wandb.ai/ll7/robot_sf/runs/7t40agqo> | 0.871429 | 0.128571 | 0.071857 |
| 506 | 12772 | l40s | <https://wandb.ai/ll7/robot_sf/runs/76r3v1jb> | 0.885714 | 0.114286 | 0.067319 |

Aggregate over these four final 10M-checkpoint rows:

- Mean success rate: 0.896429.
- Mean collision rate: 0.103571.
- Mean SNQI: 0.169702.
- Mean evaluation episode return: 22.249741.

## Files

- `seed_summary.csv`: compact per-seed final-checkpoint table plus W&B links and source checksums.
- `seed_summary.json`: typed JSON equivalent with provenance metadata.
- `SHA256SUMS`: checksums for the tracked compact files in this bundle.

The source run summaries, final eval timelines, and source config copies are identified by SHA-256
inside `seed_summary.csv`/`seed_summary.json`; the large local model and episode artifacts are not
mirrored into git.
