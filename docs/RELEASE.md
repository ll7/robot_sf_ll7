# Benchmark Release Checklist

This checklist covers the approved S30/H600 benchmark-data release. It is a
different release lane from the Robot SF software/package release: the
benchmark-data tag identifies an immutable campaign contract, while the
software tag identifies installable source. Do not infer a package version from
the benchmark-data tag, or reuse a software-release DOI for benchmark data.

The current campaign contract is the 14-arm, differential-drive matrix in
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended_post1.yaml`.
The bounded one-scenario/one-seed preflight and runtime smoke is tracked by
`configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml`.
The smoke is execution evidence only: the Social Navigation Quality Index
(SNQI) remains advisory and has no planner-ranking authority.

## Before Running

- confirm the target branch/tag is the intended immutable code state
- confirm the approved S30/H600 full-release manifest is the one paired with the
  campaign config above; do not substitute the historical v1 seven-planner/S3
  manifest
- confirm the bounded smoke manifest is correct:
  - `configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml`
- confirm manifest hashes still match referenced config and assets
- confirm benchmark fallback policy is fail-closed for benchmark mode
- confirm a fresh Zenodo concept is reserved for the benchmark-data record; the
  historical concepts `10.5281/zenodo.19482025` and
  `10.5281/zenodo.19563812` must not be reused
- confirm SNQI is documented as advisory/no-ranking, including when calibration
  reports a warning
- classify smoke artifacts before handoff: raw episode files remain worktree-local
  ignored cache, while the manifest/config hashes and compact summary are the
  tracked provenance surface; record source commit, command, seed, and campaign
  root in the release evidence

## Preflight

Run:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml \
  --mode preflight
```

Verify:

- manifest validation status is `valid`
- preflight artifacts were written
- matrix summary reflects all 14 planner arms, one smoke scenario, one seed,
  H600, and differential-drive kinematics
- smoke output is labeled `runtime-smoke`; it is not full benchmark evidence

## Release Execution

First stage every referenced checkpoint into durable shared storage and persist
the exact admission receipt:

```bash
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_extended_post1.yaml \
  --stage \
  --report-path output/release/checkpoints/staging_receipt.json
```

Require `submit_safe=true`. Then run:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest <approved-s30-h600-full-release-manifest.yaml> \
  --label release \
  --checkpoint-receipt output/release/checkpoints/staging_receipt.json
```

Verify:

- process exit code is zero
- `benchmark_success` is `true`
- `release/release_manifest.resolved.json` exists
- `release/release_result.json` exists
- required campaign artifacts exist
- publication bundle archive, manifest, and checksums exist

From the extracted bundle root, verify the payload checksums with:

```bash
sha256sum -c checksums.sha256
```

The publication preflight also requires `release_result.json` and the rebuilt
`campaign_summary.json` to agree on status, validity, and cardinality. If episode
runtime commits differ from the publication commit, record
`provenance.commit_reconciliation` with the runtime commit list, publication
commit, and a plain-language explanation. Goal-reached plus timeout rows require
`provenance.goal_timeout_boundary` with timing evidence or an explicit exclusion
note.

## Publication

Upload the generated bundle using:

- `docs/benchmark_camera_ready_release.md`

## Version Alignment (single source of truth: the software git tag)

The package version is derived automatically from the git tag by `hatch-vcs`
(release tags are plain `X.Y.Z`, e.g. `0.0.2`; release-candidate tags are
`rcX.Y.Z`, e.g. `rc0.0.3`). Do not hardcode a version in `pyproject.toml`.

When cutting a **full release** `X.Y.Z`:

- bump `CITATION.cff` `version:` to the new `X.Y.Z` (it tracks the latest full
  release tag; the benchmark release-protocol context stays in the title/abstract)
- run the alignment guard locally before tagging:

  ```bash
  uv run python scripts/dev/check_version_alignment.py
  ```

- push the tag; the `release-functional-badge` workflow re-runs this guard
  (gating) so `pyproject`, the built package, and `CITATION.cff` cannot drift

The guard also runs advisory (non-gating) on every CI run via the `lint` phase
of `scripts/dev/ci_driver.sh`.

Benchmark-data release tags are intentionally separate. They carry the frozen
S30/H600 config and manifest identity, but they do not change `pyproject.toml`,
`CITATION.cff`, or the package version. Publish benchmark data only after the
fresh Zenodo concept, checksums, claim boundary, and full-run evidence have been
reviewed.

## Archive and Citation

- ensure `CITATION.cff` remains current (aligned to the latest full release tag)
- keep the release tag and release asset URL stable
- replace DOI placeholder only when a real DOI exists
