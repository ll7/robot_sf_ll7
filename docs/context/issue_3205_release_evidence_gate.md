# Issue #3205 Release Evidence Reproduction Gate

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/3205>

## Claim Boundary (Read First)

- This adds a **reproduction-verification gate**, not new benchmark evidence. It proves that a
  tagged release's promoted tables regenerate byte-identically from the released canonical rows.
- "Reproduction" here means **regenerating the promoted tables from the released report rows**; it
  does **not** re-run the benchmark campaign. A `PASS` snapshot means the release is independently
  reproducible from a clean checkout and therefore safe to cite, not that any new result was produced.
- Evidence status: **smoke evidence** for the gate itself (synthetic-fixture tests) plus one
  **nominal** end-to-end reproduction of the existing `0.0.2` release.

## What This Adds

- `scripts/dev/release_evidence_gate.py` — a single fail-closed command that:
  1. verifies the release archive SHA-256 against the expected value (archive mode);
  2. regenerates the dissertation artifact bundle from the canonical report rows via the existing
     `scripts/tools/benchmark_publication_bundle.py dissertation-bundle` contract;
  3. compares every regenerated artifact's SHA-256 against the tracked reference manifest;
  4. emits a citable `release_evidence_snapshot.v1` JSON and exits non-zero on any archive mismatch,
     missing artifact, or checksum drift (fail-closed).
- `tests/tools/test_release_evidence_gate.py` — exercises the real generator through a synthetic
  fixture: faithful reproduction → `PASS`; tampered rows, missing artifact, and corrupted archive →
  fail-closed.

This composes the existing release/validation tooling (`benchmark_publication_bundle.py`,
`validate-dissertation-bundle`, the `0.0.2` table bundle) into one repeatable gate. The previous path
was a manual multi-step shell chain recorded in `docs/context/issue_2689_release_evidence_handoff_2026_06_15.md`;
this turns that chain into a testable, single-command contract that any future tagged release can run.

## Validation Evidence

- `uv run pytest tests/tools/test_release_evidence_gate.py` → 5 passed (on imech036 shared venv).
- `ruff check` / `ruff format` → clean.
- **End-to-end reproduction of release `0.0.2`** (genuine run, exit 0, `status: PASS`):
  - archive SHA-256 matched `64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90`;
  - all 3 promoted tables (`tab_results_overview`, `tab_robot_sf_release_planner_results`,
    `tab_release_failure_count_slices`) regenerated with matching checksums;
  - snapshot copied to `docs/context/evidence/issue_3205_release_evidence_gate/snapshot_0_0_2.json`.

## Reproduction Command

```bash
gh release download 0.0.2 --repo ll7/robot_sf_ll7 --pattern '*publication_bundle.tar.gz' --dir /tmp/rel
EVID=docs/context/evidence/issue_2686_release_0_0_2_table_bundle
uv run python scripts/dev/release_evidence_gate.py \
  --archive /tmp/rel/paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz \
  --expected-archive-sha256 64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90 \
  --artifact-spec $EVID/artifact_spec.json \
  --reference-manifest $EVID/artifact_manifest.json \
  --source-commit f7ebdcae2375d085e925213197a75a386e26a79c \
  --release-tag 0.0.2 --doi 10.5281/zenodo.19563812 \
  --out output/release_evidence/snapshot_0_0_2.json
```

## Downstream / Follow-up

- This is the reusable gate the successor-release plan (#3081) and the living-sync work depend on:
  each future tag can now emit a `PASS` snapshot that downstream paper-facing consumers cite.
- Next increment (separate issue/PR): wire the gate into the release workflow so a snapshot is
  produced automatically at tag time, and extend it to figures (not just tables) where canonical
  source rows exist.
