# Slurm job 13378 frozen-campaign closeout

## Disposition

Job `13378` completed successfully at the scheduler level and contains a useful
diagnostic signal, but the official campaign aggregate is **not eligible for
paper-facing or record-breaking claims**. The campaign reused one output root
across resume attempts, so the episode JSONL files contain append history from
multiple public commits. No rerun was launched during this closeout.

The exact final-public-commit slice is complete and supports a bounded
hypothesis: on this simulator-only h600/S30 surface, hybrid v3 continuous has a
meaningful success/safety separation from ORCA and PPO. That slice is retained
as diagnostic evidence only; it does not repair or supersede the official
campaign report.

## Scheduler and provenance

- Slurm: `COMPLETED`, exit `0:0`, elapsed `14:28:14`.
- Start/end: `2026-07-12T09:52:11+02:00` to
  `2026-07-13T00:20:25+02:00`.
- Public commit: `a596a33b27acc835a6c1b399bd36cd2699792437`.
- Campaign ID: `issue4365_h600_hybrid_vs_orca_s30_run_20260704`.
- Scenario matrix hash recorded by the payload: `152eba3969a9`.
- The 63 original remote files and the 63 retrieved files have identical sorted
  SHA-256 multisets. Analyzer-generated reports were created only after this
  check.
- The private operations ledger did not contain the submission row while the
  job was running. Queue commit `59ee646bfdb33738d860ab2b9bdde49c8d4e54bc`
  records the attempt post hoc, but the exact private-ops submission commit is
  unknown and remains blank in the reconstructed ledger row.

## Frozen artifact integrity

The campaign summary declares 1,440 written episodes per planner. The episode
files instead contain:

| Planner | JSONL rows | Final-commit rows | Unique final scenario/seed pairs |
|---|---:|---:|---:|
| hybrid v1 | 8,640 | 1,440 | 1,440 |
| hybrid v2 | 8,640 | 1,440 | 1,440 |
| hybrid v3 static | 8,640 | 1,440 | 1,440 |
| hybrid v3 continuous | 8,640 | 1,440 | 1,440 |
| ORCA | 8,640 | 1,440 | 1,440 |
| PPO | 4,195 | 1,440 | 1,440 |

The five 8,640-row files contain six 1,440-row commit blocks. PPO contains
partial earlier blocks plus a complete final block. Every final block covers
48 scenarios by 30 seeds exactly once, and the last 1,440 rows equal the latest
row for every scenario/seed pair. Binary outcome signatures were stable across
repeated attempts for every planner, but continuous metadata differs.

This contaminates the official PPO aggregate materially: the campaign table
reports 59.52% success and 37.83% collision, while the clean final slice is
71.74% success and 26.46% collision. The official multi-attempt aggregate must
not be used for planner ranking or scientific promotion.

## Final-commit diagnostic slice

Means and 95% seed-block percentile bootstrap intervals use 30 seed blocks and
30,000 resamples with seed 123. These intervals are diagnostic and are not a
substitute for the preregistered inference contract.

| Planner | Success | Collision | Normalized time |
|---|---:|---:|---:|
| hybrid v3 continuous | 77.15% [74.72, 79.31] | 14.86% [13.19, 16.53] | 0.460 [0.442, 0.479] |
| hybrid v3 static | 75.97% [73.40, 78.40] | 20.56% [18.89, 22.22] | 0.454 |
| hybrid v1 | 75.35% [72.85, 77.71] | 21.53% [19.79, 23.19] | 0.458 |
| hybrid v2 | 75.35% [72.92, 77.64] | 21.11% [19.38, 22.78] | 0.458 |
| PPO | 71.74% [70.14, 73.26] | 26.46% [25.07, 27.92] | 0.456 |
| ORCA | 68.06% [65.76, 70.35] | 28.33% [26.18, 30.63] | 0.520 [0.504, 0.536] |

Paired seed-block differences for hybrid v3 continuous:

- versus ORCA: success `+9.10` percentage points, 95% interval
  `[+6.46, +11.67]`; collision `-13.47` points
  `[-15.83, -10.97]`; normalized time `-0.060`
  `[-0.078, -0.041]`.
- versus PPO: success `+5.42` percentage points
  `[+2.71, +8.06]`; collision `-11.60` points
  `[-13.61, -9.58]`; normalized-time difference `+0.004` with an interval
  crossing zero.

The metric schema check uses `metrics.success` for success and
`outcome.collision_event` for collision. `metrics.collisions` differs from the
outcome event in the final PPO slice and is not used as the authoritative
collision field.

## Claim boundaries and remaining risks

- `paper_facing` is false and the required AMV coverage dimensions are absent.
- The SNQI contract fails (`rank_alignment_spearman = -0.942857`); SNQI values
  and ranking claims are excluded.
- Hybrid arms use adapter execution while PPO is native. Several AMMV proxy
  feasibility fields are false, so those proxies do not certify comparable
  physical execution.
- Runtime cost is large: mean episode wall time is about 18.26 s for hybrid v3
  continuous versus 1.34 s for ORCA; p95 is 53.23 s versus 3.30 s.
- There is no real-world or independent clean-machine validation in this
  payload. Scenario-difficulty output is ORCA-only because ORCA is the sole core
  planner in that analysis and is not a cross-planner consensus.
- The repository-local analyzer initially rejected the retrieved location
  because a repo-relative `episodes_path` was prepended to the supplied campaign
  root. Analysis succeeded through a read-only canonical-path symlink view; no
  frozen benchmark file was rewritten.

Final status: **credible diagnostic, artifact-integrity blocked for promotion**.
Retain the frozen payload and this clean-slice analysis, but do not cite the
official aggregate as a completed S30 result and do not restart this campaign
from this closeout.
