# Reproduction Provenance — Issue #5785 (Package B 27-Cell Conservation)

Replication of the executed Package B 27-cell diagnostic result originally reported by
PR #5710 (Issue #3079), recovered by exact CPU re-execution after the original
worktree-local artifacts were found unrecoverable from the recorded surfaces.

## Conservation decision

- Bounded recovery search of the recorded surfaces (Mac, imech156-u, imech036, imech039,
  orphan/worktree archives, private artifact storage, output receipts) found NO recoverable
  copy of the raw Package B candidate/replay tree. The compact claim therefore lacked a durable
  artifact anchor.
- Exact replication was performed at the recorded execution commit and manifest identity.

## Reproduction identity

- Execution commit: `7ec582b81cdcb871fb4fcb47700338194e7617d5`
  (matches PR #5710; is an ancestor of current `origin/main`).
- Manifest path: `configs/adversarial/issue_3079_package_b_budget_matched.yaml`
- Manifest SHA-256 (recorded in issue #5785): `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04`
- Manifest SHA-256 (verified at execution commit): `9f174f067d23efd374c019702168213a27085dfffa1b0b5bc10adafaa9614e04`
  -> IDENTITY MATCH.
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

## Result (this replication)

- 27 executed cells (3 samplers x 3 budgets x 3 seeds).
- 42 certified/replayable valid failures: random=24, optuna=18, coordinate=0.
- replayable == certified in every cell (replay_success_rate 1.0 where failures exist).
- 0 fallback / degraded candidate rows.
- report gate status: `ready_for_empirical_review`.
- Confirmation sidecar: all 42 certified failures remain `not_confirmed` (awaiting
  independent-seed confirmation, deterministic replay, and stable mechanism attribution).
  No failure is silently counted as a confirmed discovery.

## Preservation boundary

- The committed bundle contains the summary artifacts and the
  `candidate_replay_SHA256SUMS.txt` checksum manifest.
- The 4,761-file candidate/replay tree and execution stdout/stderr are not present in this
  checkout, and this bundle has no durable URI or registry entry for them.
- The checksum manifest is a durable integrity record, not proof that the separately preserved
  tree is retrievable. Issue #5785 remains blocked on publishing or registering that tree and the
  captured execution logs.

## Disagreement handling

The replication AGREES with the historical `42 failures (random 24, optuna 18, coordinate 0)`.
No disagreement was observed. Had any count disagreed with the historical report, that
disagreement would have been reported as evidence under the issue's claim boundary, not
normalized away.

## Claim boundary

Diagnostic / local nominal evidence only. NOT paper-facing. This bundle conserves the
executed result; it does NOT authorize paper-facing promotion or treat the replication as the
lost original run. Independent-seed confirmation, mechanism attribution, and held-out-family
yield review (issue #5785 step 5) remain deferred to a separate interpretive step.

## Files in this bundle

- `report.json`: orchestrator durable report (27 rows, full metrics).
- `confirmation.json`: censored confirmation sidecar (artifact-bound to report.json).
- `comparison_table.md`: durable Package-B comparison table.
- `replication_summary.json`: pipeline summary payload.
- `SHA256SUMS`: checksums for the four durable summary artifacts above.
- `candidate_replay_SHA256SUMS.txt`: SHA-256 of all 4,761 candidate/replay artifacts in the
  frozen raw replay tree (the full replayable set that every reported count regenerates from).
  The tree is not included in this checkout; see the preservation boundary above.
- `provenance.md`: this file.
- `README.md`: human-readable conservation summary.
