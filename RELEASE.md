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

## Archive and Citation

- ensure `CITATION.cff` remains current
- keep the release tag and release asset URL stable
- replace DOI placeholder only when a real DOI exists
