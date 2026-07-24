<!-- AI-GENERATED (robot_sf#5785, 2026-07-15) - NEEDS-REVIEW -->
# Issue #5785: Package B 27-Cell Diagnostic Summary (2026-07-15)

Plain-language summary: this tracked bundle preserves a self-consistent summary of a Package B
CPU diagnostic associated with Issue #3079 and PR #5710. The committed report contains 27 cells
(3 samplers x 3 budgets x 3 seeds) and records 42 certified/replayable failure counts
(random=24, optuna=18, coordinate=0). The focused test regenerates those counts from the committed
report only. The 4,761 candidate/replay file inventory is registered in `candidate_replay_SHA256SUMS.txt`
as portable names and producer-recorded digest assertions only; the referenced bytes and execution
logs are unavailable. This bundle maintains the diagnostic-only claim boundary and is not
paper-facing evidence. Issue #6131 remains blocked on a durable raw-artifact archive.

## PR #6185 Scope and Residual Ownership

PR #6185 is a partial preparatory change. It adds a fail-closed retrieval and
verification helper, but it does not complete or close Issue #6131.

Issue #6131 remains open and owns publication of the original raw-tree archive
and stdout/stderr logs, committed portable retrieval metadata, and clean-checkout
verification of all 4,761 inventory entries against the retrieved bytes.

- Schema (report): `adversarial-sampler-comparison.v3`
- Schema (confirmation): `adversarial-package-b-confirmation.v1`
- Execution commit: `7ec582b81cdcb871fb4fcb47700338194e7617d5`
- Manifest SHA-256: `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04` (match)
- Report gate status: `ready_for_empirical_review`
- Evidence tier: `diagnostic-only summary`
- Result classification: `diagnostic-only`
- Claim boundary: internal consistency of the committed 27-cell summary and its recorded sampler
  totals only. `certified` and `replayable` are source-report fields, not independently reverified
  raw-artifact claims in this bundle. It is not simulator-realism, sim-to-real, benchmark, or
  paper-facing evidence. All 42 recorded failures remain `not_confirmed` pending deterministic
  replay, independent-seed confirmation, and stable mechanism attribution.

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

## Recorded execution lineage

- The producer recorded CPU execution commit `7ec582b81cdcb871fb4fcb47700338194e7617d5`, manifest SHA-256 `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04`,
  and runtime 198s. The tracked config still matches that manifest digest.
- The committed summary agrees with the historical `42 failures (random 24, optuna 18,
  coordinate 0)` count.
- The 4,761 candidate/replay file inventory is registered in `candidate_replay_SHA256SUMS.txt` as
  portable names and producer-recorded digests. It is not a byte-verification result.

## Raw Artifact Retrieval Status (Issue #6131)

- Status: `blocked`; no durable archive URI/content address contains the original raw tree and
  stdout/stderr logs.
- Producer public commit head: `7ec582b81cdcb871fb4fcb47700338194e7617d5`
- Manifest SHA-256: `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04`
- Total candidate/replay inventory entries: `4761`
- Inventory status: `unavailable_raw_bytes`; syntax, uniqueness, and portable-name checks alone
  are not raw-byte verification.
- Bounded recovery command attempted:
  `python scripts/tools/run_adversarial_package_b.py --manifest configs/adversarial/issue_3079_package_b_budget_matched.yaml --repo-root . --empirical`
- Recovery result: `96` matching files, `4,594` missing files, and `71` changed files; the attempt
  did not reproduce the producer archive and its captured stdout/stderr are local-only diagnostics.
- Fail-closed verification command (returns non-zero until a pinned archive is committed):
  `python scripts/tools/verify_package_b_raw_artifacts.py --bundle docs/context/evidence/issue_5785_package_b_27cell_replication_2026-07-15`
- Unblock condition: publish an HTTPS archive with its SHA-256, archive layout, and SHA-256 entries
  for both logs in `raw_artifact_bundle.json`; then retrieve and byte-verify all 4,761 entries.
- Preservation boundary: diagnostic-only summary (27 cells, 42 certified/replayable valid failures:
  random=24, optuna=18, coordinate=0). Not paper-facing.

## Files

- `report.json`: committed 27-row diagnostic summary with portable raw-tree identifiers.
- `confirmation.json`: censored sidecar bound to the committed `report.json` bytes.
- `comparison_table.md`: diagnostic rendering of the committed report fields.
- `replication_summary.json`: pipeline summary payload.
- `SHA256SUMS`: checksums for the four committed summary artifacts.
- `candidate_replay_SHA256SUMS.txt`: producer-recorded digest inventory for 4,761 candidate/replay
  files. Its relative names are portable identifiers, not retrievable bytes.
- `provenance.md`: recorded execution lineage, blocked retrieval status, and the explicit
  preservation boundary.
- `README.md`: this human-readable summary.

Related work: Refs #5785, #6095, and #6131.
