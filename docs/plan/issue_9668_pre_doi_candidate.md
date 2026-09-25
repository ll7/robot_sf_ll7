# 0.0.8 evaluation before publication approval

## Goal and boundary

Run and verify the 0.0.8 scientific matrix before the author authorizes a tag or
Zenodo record. A candidate is not a release or a publication bundle. The frozen
0.0.7 archive and all raw 0.0.8 episode rows remain unchanged when publication
identity is bound later.

## Evidence and ownership

- Issue #9668 requires the same 14 configurations, 48 scenarios, seeds 111–140,
  horizon 600 and 0.1-second step as 0.0.7; the predecessor archive is pinned by
  SHA-256 `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`.
- The dedicated 0.0.8 campaign template has SHA-256
  `cd749189831cc6cd940aed695687b52a476f02e0c11f4afeb792e84861ee2774`.
  A static parser comparison found all 14 algorithm-config identities and five
  checkpoint references unchanged from 0.0.7. Its receipt is in the task-owned
  `.git/codex-agent-runs/issue9668-evidence/stage0-efaa9bed/` directory.
- `run_camera_ready_benchmark.py` and `camera_ready/` own raw execution;
  `release_protocol.py` owns source and scientific identity;
  `check_release_metric_equivalence.py` owns predecessor comparison;
  `finalize_benchmark_data_v008.py` owns derivative publication packaging.

## Candidate contract

1. An explicit pre-DOI mode admits only the exact dedicated template bytes at
   the exact clean source. It resolves the matrix, config hashes, model-registry
   pins, staged checkpoint receipt, and exact-source runtime smoke receipt before
   execution. DOI and release-tag
   fields are **absent** from candidate artifacts; unresolved template slots
   never become apparent publication coordinates. Existing runner modes and
   v1 output bytes retain their existing behavior.
2. Candidate custody records the source SHA, campaign-template SHA, ordered
   planner config paths and hashes, checkpoint identities, scenario and seed
   hashes, all 20,160 expected cell keys, producer sidecars, and each raw
   `episodes.jsonl` digest. A separate scientific config hash ignores only
   publication identity fields. It is versioned and used only by this mode.
   Before execution, all 14 arms and five staged checkpoint references must
   match the SHA-pinned 0.0.7 archive's campaign manifest, including registry
   digests. After execution, the arm-level checkpoint and runtime bundle
   digests and their hash sources are compared again; a staging/smoke pair
   that agrees only with itself is insufficient.
3. Acceptance uses the strict full-matrix and fallback/degraded exclusions,
   then the 0.0.7 predecessor comparator at absolute tolerance `1e-12` and
   robot-force validation. A mismatch or missing custody stops the candidate.
   No candidate status is described as a published benchmark release.
4. After author approval, the source-bound resolver obtains real DOI values.
   The finalizer copies the accepted producer and binds publication metadata
   only in that copy. It verifies every raw row digest and source/config/seed
   declaration against the pre-DOI receipt before and after binding, re-runs
   full release acceptance, then exports the bundle. The publication hash is
   separate from the earlier scientific hash. A source, config, row, or model
   pin drift blocks promotion; no rows are silently rerun or rewritten.

## Proof and stop rule

- Test negative cases: placeholder DOI leakage, changed source/config/model
  pin, missing or changed raw row/sidecar, incomplete matrix, degraded row,
  duplicate key, stale receipt, and DOI-bound copy changing a raw digest.
- Test a tiny synthetic candidate to DOI-bound derivative fixture, asserting
  original row bytes and scientific hash are identical. Do not use evaluation
  seeds in tests or local verification.
- Run focused tests, Ruff and formatting; then exact-head full PR readiness and
  independent scientific review before any candidate launch. The full 20,160
  row equivalence report and frozen publication archive are required before a
  0.0.8 claim. If a gate fails, classify the attempt and stop.
