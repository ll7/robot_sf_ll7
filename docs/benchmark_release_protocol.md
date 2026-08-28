# Benchmark Release Protocol v0.1

This document defines the benchmark release model used for immutable,
paper-facing benchmark artifacts in `robot_sf_ll7`.

The current approved benchmark-data surface is the S30/H600 release: 14
planner arms, 48 scenarios, 30 seeds, horizon 600, and differential-drive
kinematics. The historical seven-planner/S3 manifest remains available for
reproduction of old artifacts only; it is not the current release contract.

## Scope

This protocol covers the benchmark release process only. It does not declare the
repository or Python package to be `1.0`.

Benchmark-data and software releases are separate. A benchmark-data tag freezes
the campaign config, manifest, assets, and evidence; a software tag freezes the
installable package. A benchmark-data release must use a fresh Zenodo concept,
not an existing software or historical benchmark concept.

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

## Current Canonical Release Unit

The approved S30/H600 benchmark-data campaign is:

- campaign config:
  - `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml`
- publication-grade manifest:
  - `configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml`
- fresh Zenodo reservation:
  - concept DOI `10.5281/zenodo.22077447`
  - version DOI `10.5281/zenodo.22077448`
- source contract:
  - 14 planner arms, `paper_eval_s30`, `horizon: 600`, and
    `kinematics_matrix: [differential_drive]`
- bounded runtime smoke manifest:
  - `configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml`

The smoke selects one scenario (`francis2023_blind_corner`) and seed `111` but
retains all 14 arms and H600. It proves construction and runtime compatibility;
it is not a substitute for the full 20,160-episode benchmark-data release.
The historical v1/S3 manifest is a compatibility/reproduction input only.

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
- repository URL and distinct reserved concept/version DOI identities
- citation/checklist references

The publication-grade manifest uses
`schema_version: benchmark-release-manifest.v0.2`. In addition to the fields
above, it pins the exact latest-green base commit, expected 20,160 episode
identities, suite policy and route-certification hashes, resolved seeds
`111..140`, the `advisory_no_ranking` SNQI claim policy, direct Zenodo dataset
channel, and distinct fresh concept/version DOIs. `provenance.doi` must equal
the reserved version DOI.

The release claim boundary must also state:

- Social Navigation Quality Index (SNQI) is advisory only; calibration warnings
  do not authorize planner ranking
- runtime smoke output is diagnostic execution evidence, not full benchmark
  evidence
- software package/version claims are outside the benchmark-data manifest
- Zenodo uses a fresh concept; `10.5281/zenodo.19482025` and
  `10.5281/zenodo.19563812` are historical records and must not be reused

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
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml \
  --stage \
  --report-path output/release/checkpoints/runtime_smoke_staging_receipt.json
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml \
  --checkpoint-receipt output/release/checkpoints/runtime_smoke_staging_receipt.json
```

This command is the bounded smoke path. Use
`configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml` for the
full publication campaign; do not replace it with the historical
seven-planner/S3 manifest.

The release entrypoint:

1. validates the manifest,
2. rejects a missing, stale, config-mismatched, or non-submit-safe checkpoint receipt,
3. rejects same-campaign resume unless a fresh, hash-bound receipt classifies the
   interruption as infrastructure-only with the source/config/checkpoint inputs unchanged,
4. runs preflight through the existing camera-ready stack,
5. runs the canonical campaign,
6. fails closed unless the exact 14-arm, 48-scenario, 30-seed, H600 identity
   product succeeds once at one source commit with no fallback, degraded,
   failed, or unavailable evidence,
7. injects benchmark-release provenance into campaign artifacts,
8. exports a publication bundle only for benchmark-valid runs,
9. writes archival release metadata under `<campaign_root>/release/`.

The entrypoint is intentionally a release wrapper, not a second benchmark
execution engine.

For a no-campaign rehearsal of a v0.2 release, use the named `rehearsal` mode
with both the enforced-staged checkpoint receipt and the exact-source runtime
smoke receipt:

```bash
REHEARSAL_SOURCE_COMMIT="$(git rev-parse --verify HEAD)"
git diff --quiet
uv run python scripts/tools/run_benchmark_release.py \
  --manifest configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml \
  --mode rehearsal \
  --source-commit "$REHEARSAL_SOURCE_COMMIT" \
  --checkpoint-receipt output/release/checkpoints/staging_receipt.json \
  --runtime-smoke-receipt output/benchmarks/camera_ready/<smoke_id>/release/release_result.json
```

Rehearsal normalizes repository-relative inputs, verifies the checked-out
source/config/manifest identities, and reports startup, planner-roster,
checkpoint, and runtime-smoke admissions. It returns success only when all
admissions pass, while explicitly reporting `campaign_execution_status` as
`not_started`; it creates no campaign output, episode, publication bundle, or
scheduler submission. The runtime-smoke admission must carry the exact SHA-256
of its own embedded staged-checkpoint receipt. The release and runtime-smoke
receipts are validated independently; their wrapper hashes may differ because
their campaign-config bindings differ, but their checkpoint arm identities and
model-byte SHA-256 values must match. Allocation and resume options, including
`--resume-receipt-max-age-hours`, are rejected in this mode.
The canonical benchmark-data manifest is a historical compatibility manifest
without `source_sha`, so `--source-commit` is required and must be an exact
40-character SHA equal to the clean checked-out `HEAD`. A manifest-declared
`source_sha`, when present, remains authoritative; an explicit argument that
disagrees with it is rejected.

`release/release_result.json` preserves the wrapped campaign semantics in the
top-level `status`, `status_reason`, `benchmark_success`, `exit_code`,
`campaign_execution_status`, `evidence_status`, and `row_status_summary`
fields. Release-wrapper state is recorded separately under
`release_status`, `release_status_reason`, `release_benchmark_success`, and
`release_exit_code`. Exit code `3` means the campaign finished as
`accepted_unavailable_only`: still non-success and still fail-closed.

`benchmark_success` now means the campaign produced fully valid benchmark
evidence, not merely that the core planner subset completed. Accepted
unavailable/excluded rows therefore keep `benchmark_success=false` even when
the campaign execution finished cleanly. Distinguish the axes as follows:

- `campaign_execution_status`
  - `completed`: the campaign finished without unexpected row failures
  - `failed`: malformed payloads or unexpected failed rows
  - `interrupted`: unexpected failure with fewer executed rows than expected
- `evidence_status`
  - `valid`: every executed row produced benchmark-valid evidence
  - `partial`: at least one successful evidence row plus accepted unavailable rows
  - `blocked`: only accepted unavailable/excluded rows were produced
  - `invalid`: unexpected failures or malformed payloads
- `row_status_summary`
  - `successful_evidence_rows`
  - `accepted_unavailable_rows`
  - `unexpected_failed_rows`
  - `fallback_or_degraded_rows`

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
