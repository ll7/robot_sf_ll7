# Issue #5952 Evidence-Registry Ratchet Baseline Reconciliation

Issue: <https://github.com/ll7/robot_sf_ll7/issues/5952>

## Goal

Restore the downward ratchet at `scripts/dev/evidence_registry_ratchet.py` as a
trustworthy clean-main provenance signal. The #5275/#5317 baseline
(`scripts/validation/evidence_registry_baseline.json`, committed 2026-07-11) had
drifted: twelve evidence files merged after the baseline gained 48 findings the
ratchet did not grandfather, so `--check` failed on clean `origin/main` even
though the strict linter reported zero active findings (the new codes were
already in the strict-CI exclusion policy).

## Implementation

This is a successor slice to #5275/#5317 (which built the ratchet and the
initial 359-finding baseline). It does not duplicate them; it reconciles the
baseline, records per-file review, and adds the clean-main regression guard that
was missing.

* Refreshed `scripts/validation/evidence_registry_baseline.json` via
  `evidence_registry_ratchet.py --write-baseline` so it reconciles with the
  tracked evidence files on clean `origin/main` (407 findings across 85 files).
  The counts are machine-generated and reproduce exactly from `--write-baseline`.
* Added `scripts/validation/evidence_registry_baseline_review.yaml`: a per-file
  remediate-or-baseline disposition record for every file added since the
  #5275/#5317 baseline. All twelve files use the same legacy-debt pattern as the
  original 359 findings (artifact-provenance and campaign-provenance gaps that
  require recovering external or historical provenance), and all are
  dispositioned `baseline`. None admitted a trivial in-repo repair, and "deleting
  artifacts" / global suppression were explicitly out of scope.
* Updated `docs/context/evidence/evidence_registry_strict_ci_policy.yaml`
  justification counts and decision summary to match the refreshed baseline.
* Hardened `evidence_registry_ratchet.py --report` to map a missing/malformed
  pre-rendered report to the infra-error exit code (2) instead of an uncaught
  traceback, matching the documented exit-code contract.

## Per-file disposition (48 findings across 12 files)

All twelve files were reviewed against their source. Every introduced code is
already in the strict-CI exclusion policy, so no active strict-linter finding
was introduced (strict mode stays at zero active findings). The largest deltas
are `uncommitted_artifact_missing_location` (+30) and
`hash_without_artifact_path` (+7); both are the external/private/untracked
artifact-provenance pattern. One `artifact_hash_mismatch` (the bare filename
`README.md` in `issue_3078.../artifact_manifest.yaml`) is an ambiguous-path
provenance gap — the manifest intends the manifest-directory README whose hash
matches, but the linter resolves the bare name repo-root-relative — not a
corrupted artifact. Full per-file detail lives in
`scripts/validation/evidence_registry_baseline_review.yaml`.

## Clean-main regression guard (issue DoD #4)

The original drift went undetected because nothing ran `--check` on main. Two
guards now prevent recurrence:

* `tests/dev/test_evidence_registry_ratchet.py` asserts the committed baseline
  (a) passes `--check` against the *live* registry on a clean checkout,
  (b) reproduces byte-for-byte from `--write-baseline` (no hand-edits), and
  (c) covers every post-#5317 baseline file with an explicit disposition in the
  review companion. It also covers the pure ratchet logic (aggregation, gate
  semantics, CLI roundtrip, failure on clean-file regression and per-code
  increase) and the strict policy's zero-active-findings contract.
* `.github/workflows/evidence-registry-ratchet.yml` runs `--check` as a
  **blocking** gate on PRs that touch `docs/context/evidence/**` or the
  ratchet/baseline/policy surfaces, plus a weekly schedule. (Blocking, not
  `continue-on-error`, because the baseline is now reconciled and every tracked
  file is reviewed; a failure means the PR introduced a net-new integrity
  finding that the baseline does not grandfather.)

To refresh after intentional cleanup or a reviewed new-evidence merge:

```bash
uv run python scripts/dev/evidence_registry_ratchet.py --write-baseline
# then add/adjust the per-file disposition in
# scripts/validation/evidence_registry_baseline_review.yaml and update
# prior_baseline_files_with_findings + the refreshed counts.
```

## Boundary

This slice reconciles the baseline and prevents recurrence; it does NOT
remediate the underlying legacy provenance gaps (dangling historical commits,
external/private/untracked artifacts, missing campaign provenance). Those remain
owned by the disposition packet as backfill-required debt. It does not change
evidence semantics, delete artifacts, or suppress findings globally — the
strict-CI code-level exclusion policy is unchanged in which codes it excludes.

## Validation

```bash
uv run python scripts/dev/evidence_registry_ratchet.py --check
uv run python scripts/tools/lint_evidence_registry.py --strict \
  --strict-exclusion-policy docs/context/evidence/evidence_registry_strict_ci_policy.yaml
uv run pytest tests/dev/test_evidence_registry_ratchet.py -v
uv run ruff check scripts/dev/evidence_registry_ratchet.py tests/dev/test_evidence_registry_ratchet.py
```
