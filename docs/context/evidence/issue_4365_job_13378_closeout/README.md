# Issue #4365 Job 13378 Diagnostic Closeout

Evidence status: **diagnostic-only; artifact-integrity blocked for promotion**.

Slurm job `13378` completed all six configured arms, but the shared campaign
output root contains episode rows appended across multiple attempts and public
commits. This record preserves the recoverable final-commit observation and the
reason the official aggregate must not be used for planner ranking, a
paper-facing claim, or a record-breaking claim.

## Frozen identity

- Public commit: `a596a33b27acc835a6c1b399bd36cd2699792437`.
- Campaign: `issue4365_h600_hybrid_vs_orca_s30_run_20260704`.
- Scenario matrix hash: `152eba3969a9`.
- Slurm state: `COMPLETED`, exit `0:0`, elapsed `14:28:14`.
- Original payload: 63 files. Remote and retrieved SHA-256 multisets matched
  before analysis outputs were generated.
- Inventory: [`SHA256SUMS.original`](SHA256SUMS.original).
- Machine-readable closeout: [`summary.json`](summary.json).
- Private operations record:
  [`4e4f22af77f586ff021b62a550cf877491c07689`](https://github.com/ll7/robot_sf_ll7-private-ops/commit/4e4f22af77f586ff021b62a550cf877491c07689),
  job `13378`.

The raw 1.4 GiB payload is intentionally not committed. Its retrieval locator
and terminal state are retained in private operations; this tracked packet
preserves the compact interpretation and complete original checksum inventory.
The exact private-operations submission commit was not recorded and remains an
explicit provenance gap.

## Integrity finding

The campaign summaries declare 1,440 episodes per planner. The episode JSONL
files contain 8,640 rows for each non-PPO arm and 4,195 PPO rows. The five
8,640-row files contain six 1,440-row commit blocks; PPO contains partial prior
blocks plus one complete final block. The exact final-public-commit slice has
1,440 unique scenario/seed pairs per planner (48 scenarios by 30 seeds).

This materially changes the PPO result: the contaminated official aggregate is
59.52% success / 37.83% collision, while the final-commit slice is 71.74% /
26.46%. The official aggregate is therefore ineligible for comparison or
promotion.

## Recoverable diagnostic observation

The final-commit slice uses `metrics.success` for success and
`outcome.collision_event` for collision. `metrics.collisions` is not treated as
authoritative because it differs from the outcome event in the final PPO slice.

| Planner | Success | Collision |
| --- | ---: | ---: |
| Hybrid v3 continuous | 77.15% | 14.86% |
| PPO | 71.74% | 26.46% |
| ORCA | 68.06% | 28.33% |

Seed-block percentile bootstrap diagnostics (30 seed blocks, 30,000 resamples,
seed 123) place hybrid v3 continuous versus ORCA at `+9.10` percentage points
success, 95% interval `[+6.46, +11.67]`, and `-13.47` points collision,
`[-15.83, -10.97]`. Versus PPO, the differences are `+5.42` points success,
`[+2.71, +8.06]`, and `-11.60` points collision, `[-13.61, -9.58]`.

These are diagnostic intervals, not a preregistered inferential result. The
Social Navigation Quality Index (SNQI) contract fails, required Automated
Mobility Validation (AMV) coverage fields are missing, adapter/native execution
comparability is limited, independent clean-machine and real-world validation
are absent, and hybrid v3 continuous mean episode wall time is about 13.6 times
ORCA's. None of those boundaries is repaired by retaining this packet.

## Code follow-up

Issue [#5449](https://github.com/ll7/robot_sf_ll7/issues/5449) fixes the
seed-default resume-identity mismatch that allowed duplicate append and makes
the analyzer consume relocated self-contained campaigns without a symlink.
That implementation prevents recurrence; it does not retroactively repair this
frozen result.
