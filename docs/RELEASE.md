# Benchmark Release Checklist

This checklist covers benchmark publication releases only.

## Before Running

- confirm the target branch/tag is the intended immutable code state
- confirm the canonical manifest is correct:
  - `configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml`
- confirm manifest hashes still match referenced config and assets
- confirm benchmark fallback policy is fail-closed for benchmark mode

## Preflight

Run:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml \
  --mode preflight
```

Verify:

- manifest validation status is `valid`
- preflight artifacts were written
- matrix summary reflects the intended paper-facing planner set and kinematics

## Release Execution

Run:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml \
  --label release
```

Verify:

- process exit code is zero
- `benchmark_success` is `true`
- `release/release_manifest.resolved.json` exists
- `release/release_result.json` exists
- required campaign artifacts exist
- publication bundle archive, manifest, and checksums exist

## Publication

Upload the generated bundle using:

- `docs/benchmark_camera_ready_release.md`

## Version Alignment (single source of truth: the git tag)

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

## Archive and Citation

- ensure `CITATION.cff` remains current (aligned to the latest full release tag)
- keep the release tag and release asset URL stable
- replace DOI placeholder only when a real DOI exists