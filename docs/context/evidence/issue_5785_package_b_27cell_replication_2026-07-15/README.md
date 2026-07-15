<!-- AI-GENERATED (robot_sf#5785, 2026-07-15) - NEEDS-REVIEW -->
# Issue #5785 Package B 27-cell replication evidence 2026-07-15

Plain-language summary: this bundle conserves the executed Package B 27-cell diagnostic
result (Issue #3079 / PR #5710) which had lost its durable artifact anchor. The original
`output/adversarial/issue_3079_package_b/` artifacts were not recoverable from any recorded
surface, so the result was reproduced exactly on CPU at the recorded execution commit with
the recorded manifest identity, then durably registered with full provenance. It reports
27 executed cells (3 samplers x 3 budgets x 3 seeds) producing 42 certified/replayable valid
failures (random=24, optuna=18, coordinate=0). It is not paper-facing evidence.

- Schema (report): `adversarial-sampler-comparison.v3`
- Schema (confirmation): `adversarial-package-b-confirmation.v1`
- Execution commit: `7ec582b81cdcb871fb4fcb47700338194e7617d5`
- Manifest SHA-256: `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04` (match)
- Report gate status: `ready_for_empirical_review`
- Evidence tier: `targeted replication / conservation`
- Result classification: `diagnostic-only`
- Claim boundary: exact reproduction and durable registration of a previously-reported CPU
  diagnostic only. It runs the real CPU `pysocialforce` evaluator and produces certified,
  replayable valid failures; it is not simulator-realism evidence, not sim-to-real evidence,
  and not paper-facing evidence. All 42 certified failures remain `not_confirmed` pending
  independent-seed confirmation, deterministic replay, and stable mechanism attribution.

## Scope

- Executed cells: `27` (samplers `random, coordinate, optuna` x budgets `16, 32, 64` x seeds `1101, 2202, 3303`)
- Total certified/replayable valid failures: `42`
- By sampler: `random=24, optuna=18, coordinate=0`
- Fallback / degraded rows: `0`
- Held-out-family yield: `not_evaluated_narrow_archive` (caveat preserved from manifest)

## Aggregate certified failures per cell

| Sampler | Budget | Seed | Certified | Replayable | Replay rate | First failure iter |
|---|---:|---:|---:|---:|---:|---:|
| random | 16 | 1101 | 1 | 1 | 1.0 | 12 |
| coordinate | 16 | 1101 | 0 | 0 | - | - |
| optuna | 16 | 1101 | 0 | 0 | - | - |
| random | 16 | 2202 | 1 | 1 | 1.0 | 3 |
| coordinate | 16 | 2202 | 0 | 0 | - | - |
| optuna | 16 | 2202 | 3 | 3 | 1.0 | 9 |
| random | 16 | 3303 | 0 | 0 | - | - |
| coordinate | 16 | 3303 | 0 | 0 | - | - |
| optuna | 16 | 3303 | 3 | 3 | 1.0 | 2 |
| random | 32 | 1101 | 2 | 2 | 1.0 | 12 |
| coordinate | 32 | 1101 | 0 | 0 | - | - |
| optuna | 32 | 1101 | 0 | 0 | - | - |
| random | 32 | 2202 | 5 | 5 | 1.0 | 3 |
| coordinate | 32 | 2202 | 0 | 0 | - | - |
| optuna | 32 | 2202 | 3 | 3 | 1.0 | 9 |
| random | 32 | 3303 | 1 | 1 | 1.0 | 25 |
| coordinate | 32 | 3303 | 0 | 0 | - | - |
| optuna | 32 | 3303 | 3 | 3 | 1.0 | 2 |
| random | 64 | 1101 | 6 | 6 | 1.0 | 12 |
| coordinate | 64 | 1101 | 0 | 0 | - | - |
| optuna | 64 | 1101 | 0 | 0 | - | - |
| random | 64 | 2202 | 6 | 6 | 1.0 | 3 |
| coordinate | 64 | 2202 | 0 | 0 | - | - |
| optuna | 64 | 2202 | 3 | 3 | 1.0 | 9 |
| random | 64 | 3303 | 2 | 2 | 1.0 | 25 |
| coordinate | 64 | 3303 | 0 | 0 | - | - |
| optuna | 64 | 3303 | 3 | 3 | 1.0 | 2 |

## Exclusions (fallback / degraded / non-native)

- Excluded rows: `0`
- Reasons: `none`

Per the issue #691 benchmark fallback policy, excluded rows never contribute to the result
metrics above. No fallback or degraded candidate execution occurred in this replication.

## Reproduction and conservation

- Original artifacts unrecoverable from recorded surfaces (Mac, imech156-u, imech036,
  imech039, archives, storage, receipts).
- Exact replication at commit `7ec582b81...` with manifest identity SHA-256
  `9f174f06...` (verified match). Empirical CPU command and orchestrator from PR #5710.
- Result agrees with the historical `42 failures (random 24, optuna 18, coordinate 0)`.
- CPU runtime: 198s (recorded ~4m27s; CPU variance expected, not an exact target).

## Files

- `report.json`: orchestrator durable report (27 rows, full metrics).
- `confirmation.json`: censored confirmation sidecar (artifact-bound to report.json).
- `comparison_table.md`: durable Package-B comparison table.
- `replication_summary.json`: pipeline summary payload.
- `SHA256SUMS`: checksums for the four durable summary artifacts.
- `candidate_replay_SHA256SUMS.txt`: SHA-256 of all 4761 candidate/replay artifacts in the
  frozen replay tree (full replayable set every reported count regenerates from).
- `provenance.md`: execution commit, environment/lock identity, and conservation decision.
- `README.md`: this human-readable summary.
