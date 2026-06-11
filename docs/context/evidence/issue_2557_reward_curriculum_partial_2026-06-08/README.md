# Issue #2557 Reward-Curriculum Seed Replica Expanded Partial Evidence

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2557>

This bundle preserves compact evidence from the first fourteen completed real 10M-step seed-replica
runs for the Issue #2557 reward-curriculum follow-up. It is expanded partial training evidence only,
not a complete seed-batch result: seeds 509, 510, 511, 512, 513, 514, 515, 516, 523, and 524 were
running or pending when this bundle was refreshed on 2026-06-10.

## Claim Boundary

- Evidence tier: expanded partial training evidence.
- Not claimed: full seed-batch result, paper-grade sufficiency, or final Issue #2557 conclusion.
- Raw artifacts excluded from git: model zips, episode JSONL, W&B binary logs, and full Slurm logs.
- Durable external artifacts: W&B run pages linked in `seed_summary.csv` and `seed_summary.json`.

## Source Runs

The source branches/worktrees were:

- `issue-2557-gse-night-seeds` at commit `0e0f98a4fefe9faedc3eed31f2085af9618914e3` for seeds 501-508.
- `issue-2557-gse-topup-20260608` at commit `c7896d14e22a7ee604ab3fc07196c18dbfe86654` for seeds 517-522.

Each job used `configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml`,
`total_timesteps=10000000`, `dry_run=False`, deterministic evaluation seeds, and the default SNQI
implementation.

| Seed | Job | Partition | W&B run | Final success | Final collision | Final SNQI |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 501 | 12767 | a30 | <https://wandb.ai/ll7/robot_sf/runs/xu1ygfhp> | 0.914286 | 0.085714 | 0.263960 |
| 502 | 12768 | a30 | <https://wandb.ai/ll7/robot_sf/runs/usp2vcrl> | 0.914286 | 0.085714 | 0.275673 |
| 503 | 12769 | a30 | <https://wandb.ai/ll7/robot_sf/runs/pj5qb1w8> | 0.900000 | 0.100000 | 0.231309 |
| 504 | 12770 | a30 | <https://wandb.ai/ll7/robot_sf/runs/2b6h821r> | 0.814286 | 0.185714 | -0.071680 |
| 505 | 12771 | l40s | <https://wandb.ai/ll7/robot_sf/runs/7t40agqo> | 0.871429 | 0.128571 | 0.071857 |
| 506 | 12772 | l40s | <https://wandb.ai/ll7/robot_sf/runs/76r3v1jb> | 0.885714 | 0.114286 | 0.067319 |
| 507 | 12773 | l40s | <https://wandb.ai/ll7/robot_sf/runs/fe3ll0gb> | 0.828571 | 0.171429 | -0.020554 |
| 508 | 12774 | l40s | <https://wandb.ai/ll7/robot_sf/runs/k15wku6n> | 0.914286 | 0.085714 | 0.215544 |
| 517 | 12795 | a30 | <https://wandb.ai/ll7/robot_sf/runs/z7mebaii> | 0.885714 | 0.114286 | 0.143497 |
| 518 | 12796 | a30 | <https://wandb.ai/ll7/robot_sf/runs/uhyn9zop> | 0.885714 | 0.114286 | 0.141420 |
| 519 | 12797 | a30 | <https://wandb.ai/ll7/robot_sf/runs/4fec6gh4> | 0.885714 | 0.114286 | 0.101817 |
| 520 | 12798 | a30 | <https://wandb.ai/ll7/robot_sf/runs/sg3faqx1> | 0.885714 | 0.114286 | 0.228044 |
| 521 | 12799 | l40s | <https://wandb.ai/ll7/robot_sf/runs/2rqd6n5o> | 0.885714 | 0.114286 | 0.127681 |
| 522 | 12800 | l40s | <https://wandb.ai/ll7/robot_sf/runs/qdtlz7tk> | 0.914286 | 0.085714 | 0.264641 |

Aggregate over these fourteen final 10M-checkpoint rows:

- Mean success rate: 0.884694.
- Mean collision rate: 0.115306.
- Mean SNQI: 0.145752.
- Mean evaluation episode return: 21.800575.

## Files

- `seed_summary.csv`: compact per-seed final-checkpoint table plus W&B links and source checksums.
- `seed_summary.json`: typed JSON equivalent with provenance metadata.
- `SHA256SUMS`: checksums for the tracked compact files in this bundle.

The source run summaries, final eval timelines, and source config copies are identified by SHA-256
inside `seed_summary.csv`/`seed_summary.json`; the large local model and episode artifacts are not
mirrored into git.
