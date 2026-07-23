# Diagnostic Summary Provenance — Issue #5785 (Package B 27-Cell Result)

Claim boundary: this note records producer-reported lineage for the committed 27-cell summary. It
does not prove exact reproduction or preservation of the unavailable candidate/replay bytes and
logs. The summary is diagnostic-only and not paper-facing evidence.

## Preservation decision

- Bounded recovery search of the recorded surfaces (Mac, imech156-u, imech036, imech039,
  orphan/worktree archives, private artifact storage, and execution receipts) found no
  recoverable raw candidate/replay tree or execution logs.
- The committed report and digest inventory are retained as compact diagnostic metadata. They are
  not a substitute for the missing bytes; Issue #6131 owns retrieval and second-reader proof.

## PR #6185 Scope and Residual Ownership

PR #6185 is a partial preparatory change. It supplies fail-closed retrieval and
verification behavior only; it does not complete or close Issue #6131.

Issue #6131 remains open and owns the original archive and stdout/stderr publication,
the committed portable retrieval metadata, and verification of all 4,761 inventory
entries against bytes retrieved from a clean checkout.

## Reproduction identity

- Execution commit: `7ec582b81cdcb871fb4fcb47700338194e7617d5`
  (matches PR #5710; is an ancestor of current `origin/main`).
- Manifest path: `configs/adversarial/issue_3079_package_b_budget_matched.yaml`
- Manifest SHA-256 (recorded in issue #5785): `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04`
- Current tracked manifest SHA-256: `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04`
  -> config identity match only.
- Empirical command: `python scripts/tools/run_adversarial_package_b.py --manifest <3079 yaml> --repo-root . --empirical --fail-closed`
- CPU runtime (recorded in PR #5710): ~4m27s.
- CPU runtime (this replication): 198s. Within CPU-variance of the recorded value; the
  historical `~4m27s` is not treated as an exact target, only the cell population and counts.

## Environment / lock identity

- Python: 3.13.13
- numpy 2.4.6, pygame 2.6.1 (SDL 2.28.4), shapely 2.1.2, optuna 4.9.0, pandas 3.0.3,
  scipy 1.17.1, numba 0.66.0
- Full resolved dependency set: uv-locked (`uv pip freeze` -> SHA-256
  `08b526b5582f724ae68c1b32625a40dd4955912690abb45dbaae66afdc35e63b`).
- No Slurm, no GPU, no CARLA group. CPU `pysocialforce` simulator only.

## Recorded diagnostic summary

- 27 executed cells (3 samplers x 3 budgets x 3 seeds).
- 42 certified/replayable valid failures: random=24, optuna=18, coordinate=0.
- The source report records replayable == certified in every cell. The missing replay bytes mean
  this PR does not independently reverify that label.
- 0 fallback / degraded candidate rows.
- report gate status: `ready_for_empirical_review`.
- Confirmation sidecar: all 42 certified failures remain `not_confirmed` (awaiting
  independent-seed confirmation, deterministic replay, and stable mechanism attribution).
  No failure is silently counted as a confirmed discovery.

## Preservation boundary

- The committed bundle contains the summary artifacts and the
  `candidate_replay_SHA256SUMS.txt` checksum manifest.
- The 4,761-file candidate/replay tree digest inventory is registered in
  `candidate_replay_SHA256SUMS.txt` as portable names and producer-recorded digests. The file has
  no raw bytes, so its digest strings cannot be independently checked against the original files.
- Issue #6131 now fails closed on missing raw bytes, archive retrieval, archive checksums, log
  checksums, and changed files. Its durable-archive acceptance condition remains unmet.

## Raw Artifact Retrieval Status (Issue #6131)

- Status: `blocked`; a durable raw-tree/log archive and its content address have not been found.
- Producer public commit head: `7ec582b81cdcb871fb4fcb47700338194e7617d5`
- Manifest SHA-256: `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04`
- Total candidate/replay inventory entries: `4761`
- Inventory verification status: `unavailable_raw_bytes`; the inventory was checked only for
  portable paths, 64-character SHA-256 syntax, and uniqueness.
- Bounded recovery command attempted:
  `python scripts/tools/run_adversarial_package_b.py --manifest configs/adversarial/issue_3079_package_b_budget_matched.yaml --repo-root . --empirical`
- Recovery result: `96` matching files, `4,594` missing files, and `71` changed files. The current
  source cannot recreate the producer bytes, and its stdout/stderr are local-only diagnostics.
- Fail-closed verifier command:
  `python scripts/tools/verify_package_b_raw_artifacts.py --bundle docs/context/evidence/issue_5785_package_b_27cell_replication_2026-07-15`
- A clean checkout cannot retrieve or byte-verify this evidence until a durable archive is published.
  The required metadata schema pins an HTTPS archive SHA-256, archive layout, and both stdout/stderr
  log SHA-256 digests before it extracts the archive and verifies all 4,761 bytes.

## Disagreement handling

The committed summary agrees with the historical `42 failures (random 24, optuna 18,
coordinate 0)` count. This is a summary-level consistency statement, not independent raw-artifact
confirmation.

## Claim boundary

Diagnostic summary only. NOT paper-facing. This bundle preserves the committed report and digest
inventory, not the executed raw result. Independent replay, independent-seed confirmation,
mechanism attribution, and held-out-family yield remain unresolved.

## Files in this bundle

- `report.json`: committed 27-row diagnostic summary with portable raw-tree identifiers.
- `confirmation.json`: censored confirmation sidecar (artifact-bound to report.json).
- `comparison_table.md`: diagnostic rendering of the report fields.
- `replication_summary.json`: pipeline summary payload.
- `SHA256SUMS`: checksums for the four committed summary artifacts above.
- `candidate_replay_SHA256SUMS.txt`: producer-recorded SHA-256 inventory for 4,761 candidate/replay
  files. Relative names are portable identifiers, not retrievable or byte-verified evidence.
- `provenance.md`: this file.
- `README.md`: human-readable diagnostic-summary boundary.

Related work: Refs #5785, #6095, and #6131.
