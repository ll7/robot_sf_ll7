# Benchmark Release Protocol v0.1

This document defines the benchmark release model used for immutable,
paper-facing benchmark artifacts in `robot_sf_ll7`.

## Scope

This protocol covers the benchmark release process only. It does not declare the
repository or Python package to be `1.0`.

Three version concepts are intentionally separate:

1. Repository/package version
   - source tree and installable software lifecycle
2. Benchmark protocol version
   - release-process contract for benchmark publication artifacts
3. Benchmark release id / tag
   - one immutable benchmark artifact set built from a frozen config

The current benchmark protocol is:

- `benchmark_protocol_version: 0.1.0`
- maturity: `pre-1.0`

## Versioning Policy

Benchmark release versioning is independent from `pyproject.toml`.

- Patch:
  - documentation repair
  - provenance repair
  - release tooling bugfix
  - no intended metric/contract change
- Minor:
  - comparable benchmark contract extension or clarification
  - for example: new provenance fields, stricter release validation, additional
    reproducibility metadata
- Major:
  - non-comparable benchmark change
  - for example: changed scenario suite, seed policy, kinematics contract,
    metric contract, planner set, or SNQI normalization basis

While the release process is still evolving, benchmark releases remain in the
`0.x.y` line.

## Canonical First Release Unit

The first formal release unit is the existing paper-facing matrix:

- campaign config:
  - `configs/benchmarks/paper_experiment_matrix_v1.yaml`
- release manifest:
  - `configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml`

This is the benchmark contract we treat as frozen for release purposes. Broader
exploratory matrices are not benchmark releases.

## Release Manifest

The release manifest is a thin wrapper over existing benchmark tooling. It does
not replace the camera-ready execution stack.

Canonical fields:

- benchmark protocol version
- release id and release tag
- canonical campaign config path and SHA-256
- scenario matrix path and SHA-256
- seed policy
- SNQI asset paths and SHA-256
- planner keys and planner-group expectations
- kinematics contract
- required campaign artifacts
- repository URL and DOI placeholder
- citation/checklist references

## Benchmark Claim Artifact

The BenchmarkClaim artifact is the reviewable boundary for paper-facing
benchmark statements. It does not replace release manifests or publication
bundles: release manifests define the frozen execution contract, publication
bundles package the durable files, and BenchmarkClaim records the compact
machine-checkable evidence for one claim.

Use:

```bash
uv run robot_sf_bench claim \
  --claim-id <claim-id> \
  --statement "<paper-facing benchmark statement>" \
  --scenario-matrix <scenario-matrix.yaml> \
  --scenario-matrix-sha256 <sha256> \
  --policy-metadata <policy-metadata.json> \
  --training-episodes <training-episodes.jsonl> \
  --validation-episodes <validation-episodes.jsonl> \
  --final-benchmark-episodes <final-benchmark-episodes.jsonl> \
  --aggregate-report <aggregate-summary.json> \
  --dependency-group dev \
  --output-json <benchmark-claim.json>
```

The command writes a `schema_version: benchmark_claim.v1` JSON artifact. The
artifact keeps training, validation, and final benchmark episodes distinct, and
fails closed when the scenario matrix hash, policy artifact hashes, or
schema/version markers are missing. Final benchmark episodes are required;
training and validation episodes are optional provenance inputs and must not be
silently substituted for final benchmark evidence.

## Release Entrypoint

Use:

```bash
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v1_release_v0_1.yaml
```

The release entrypoint:

1. validates the manifest,
2. runs preflight through the existing camera-ready stack,
3. runs the canonical campaign,
4. fails closed if `benchmark_success` is false,
5. injects benchmark-release provenance into campaign artifacts,
6. exports a publication bundle only for benchmark-valid runs,
7. writes archival release metadata under `<campaign_root>/release/`.

The entrypoint is intentionally a release wrapper, not a second benchmark
execution engine.

`release/release_result.json` preserves the wrapped campaign semantics in the
top-level `status`, `status_reason`, `benchmark_success`, and `exit_code`
fields. Release-wrapper state is recorded separately under
`release_status`, `release_status_reason`, `release_benchmark_success`, and
`release_exit_code`. Exit code `3` means the campaign finished as
`accepted_unavailable_only`: still non-success and still fail-closed.

## Release Outputs

Each successful release writes:

- camera-ready campaign artifacts under `output/benchmarks/camera_ready/<campaign_id>/`
- release metadata:
  - `release/release_manifest.resolved.json`
  - `release/release_result.json`
- publication bundle under `output/benchmarks/publication/`

The campaign summary now carries benchmark-release provenance:

- `benchmark_protocol_version`
- `benchmark_release_id`
- `benchmark_release_tag`
- `benchmark_release_manifest_path`
- `benchmark_release_manifest_sha256`
- `canonical_release_config`

## Related Documents

- `docs/benchmark_camera_ready.md`
- `docs/benchmark_camera_ready_release.md`
- `docs/benchmark_artifact_publication.md`
- `docs/benchmark_release_reproducibility.md`
- `docs/RELEASE.md`
