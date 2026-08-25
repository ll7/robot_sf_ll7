# Camera-Ready Benchmark Release Workflow

## Purpose

This runbook describes publication of a validated benchmark-data campaign bundle
as a GitHub release asset with checksum and manifest verification. It is not a
software/package release procedure. The current benchmark-data target is the
14-arm S30/H600 matrix; the seven-planner/S3 instructions in historical
artifacts must not be reused.

For the full benchmark release protocol, start with:

- `docs/benchmark_release_protocol.md`
- `docs/benchmark_release_reproducibility.md`

The command in this document is the publication/upload step after a benchmark
release run has already produced a valid publication bundle.

Before publication, run the bounded smoke manifest
`configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_2.yaml`
and retain its result as runtime evidence. A smoke pass is not full benchmark
evidence. Social Navigation Quality Index (SNQI) is advisory/no-ranking for
this release, including when calibration emits a warning.

## Prerequisites

- Completed camera-ready campaign output containing:
  - `reports/campaign_summary.json`
  - `publication_bundle.archive_path`
  - `publication_bundle.checksums_path`
  - `publication_bundle.manifest_path`
- A final release bundle whose `publication_manifest.json` declares
  `release_metadata.schema_version: benchmark-release-publication-metadata.v1`
  and includes the resolved release manifest/result, citation metadata, exact
  Zenodo metadata, generated rights/provenance statement, and pinned SNQI
  weights/baseline for cold verification.
- An immutable release identity recorded in the resolved manifest and release
  result: exact tag, source SHA, campaign-config SHA, and bundle SHA-256.
- `gh` CLI authenticated for repository upload.

## Recommended Tag Naming

Use an immutable benchmark-data tag that carries the H600/S30 identity:

- `paper-matrix-v2-h600-s30-<commit-sha12>`

Keep software/package tags (for example, `0.0.3`) in their separate release
lane. Do not derive a package version from the benchmark-data tag.

## Zenodo Boundary

The benchmark-data publication requires a fresh Zenodo concept and a new
version DOI after the final bundle is validated. Do not reuse historical
concepts `10.5281/zenodo.19482025` or `10.5281/zenodo.19563812`, and do not
assume GitHub-to-Zenodo automation is enabled. Until a real record exists, keep
the manifest DOI as a pending placeholder.

The direct Zenodo path is deliberately separate from the GitHub release path.
Disable the repository's GitHub-to-Zenodo webhook immediately before publishing
the GitHub Release and leave it disabled. Do not use the webhook to create or
update this benchmark-data deposition: unrelated software or model releases
must not contaminate the benchmark concept.

## Command Path

1. Dry-run validation + command plan:

```bash
uv run python scripts/tools/publish_camera_ready_release.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --repo ll7/robot_sf_ll7 \
  --tag <release_tag> \
  --output-json output/benchmarks/camera_ready/<campaign_id>/reports/release_publish_plan.json
```

2. Execute asset upload (first publication: create the draft release first):

```bash
uv run python scripts/tools/publish_camera_ready_release.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --repo ll7/robot_sf_ll7 \
  --tag <release_tag> \
  --create-draft \
  --expected-source-sha <exact-40-char-source-sha> \
  --execute-upload
```

The upload helper uploads into a draft GitHub Release only. Use `--create-draft`
(which requires `--expected-source-sha`) to create the missing tag-targeted draft
before the first upload; it fails closed when the tag already exists at a
different target or a non-draft release is present, and skips creation when an
exact-SHA draft already exists. Dry-run (without `--execute-upload`) prints the
planned `gh release create`/`gh release upload` commands without touching GitHub.
The helper never reserves, uploads to, or publishes Zenodo. Use the direct
Zenodo CLI for the reserved deposition after the bundle has passed the
independent cold check:

```bash
# Keep this file outside Git with mode 0600; never print its contents.
ZENODO_TOKEN_FILE=/home/<user>/.config/robot-sf/zenodo.token
ZENODO_STATE=output/release/zenodo-deposition.json
ZENODO_MANIFEST=configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml
ZENODO_METADATA=configs/benchmarks/releases/benchmark_data_release_s30_h600_zenodo_metadata.json

uv run robot-sf release zenodo reserve \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --metadata "$ZENODO_METADATA"

# If this unpublished draft still exists but its local state was lost, recover
# that exact deposition instead of reserving a second DOI. This performs one
# authenticated read and fails closed on any manifest, metadata, DOI, or state drift.
uv run robot-sf release zenodo recover \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --manifest "$ZENODO_MANIFEST" \
  --metadata "$ZENODO_METADATA" \
  --deposition-id <existing-unpublished-deposition-id>

uv run robot-sf release zenodo upload \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  output/benchmarks/publication/<campaign_id>_publication_bundle.tar.gz

uv run robot-sf release zenodo verify \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --metadata "$ZENODO_METADATA"

# Run only after acceptance and independent cold verification pass.
uv run robot-sf release zenodo publish \
  --token-file "$ZENODO_TOKEN_FILE" \
  --state "$ZENODO_STATE" \
  --metadata "$ZENODO_METADATA"
```

`reserve` must return a fresh concept and version DOI; freeze those values in
the release manifest before the immutable execution point. Use `recover` only
when that exact unpublished draft still exists and the credential-free local
state was lost; it never reserves or mutates a deposition and refuses published
or mismatched drafts. `upload` must send
the byte-identical bundle used for GitHub. `verify` is read-only and must check
the title, dataset type, GPL-3.0-only license, creator union, exact source tag,
and concept/version DOI distinction. `publish` is irreversible; never run it
for a draft with missing files, unaccepted rows, or an unresolved DOI.

Disable the specific GitHub-to-Zenodo webhook through repository settings or
the approved GitHub API operation immediately before GitHub publication. Confirm
its effective state with the release doctor and retain only webhook id/state in
the operator receipt; never put a token or authorization header in that receipt.

## Validation Checklist

- `release_publish_plan.json` contains expected paths and URLs.
- `checksums.sha256` is non-empty and references bundle files.
- Release page contains archive + checksums + manifest assets.
- Campaign summary contains URL placeholders:
  - `release_url`
  - `release_asset_url`
  - `doi_url`

## Cold verification and exact identity checks

Perform these checks from a clean temporary directory that contains neither
the build output nor the source worktree:

1. Download the GitHub Release archive and its checksum/manifest assets. Extract
   the archive and run `sha256sum -c checksums.sha256` from the bundle root.
2. Confirm that `payload/release/release_manifest.resolved.json` and
   `payload/release/release_result.json` agree on release id, tag, source SHA,
   acceptance status, and 20,160 episode identities. No fallback, degraded,
   failed, or unavailable row may be treated as evidence.
3. Confirm that the metadata roles in `publication_manifest.json` point to
   files inside the archive and that every declared digest matches. The citation,
   Zenodo metadata, rights/provenance statement, and both pinned SNQI assets
   must be present. Raw episode and component-metric files remain in the
   payload; `output/` remains working storage, not a citation target.
4. Confirm the tag points to the exact source SHA recorded by the release result
   and that the frozen manifest/config hashes match the bundle. A tag or DOI
   mismatch is a release blocker, not a documentation warning.
5. Download the Zenodo file independently after upload (and again after
   publication), extract it into a second clean directory, and repeat steps
   1–4. Compare the GitHub and Zenodo archive SHA-256 values byte-for-byte.
6. Verify that Zenodo reports the reserved version DOI and parent concept, the
   exact title/type/license/creators, and the source-tag relation. Keep the
   readback receipts with the private durable-artifact record.

These checks are independent of the local build directory. A successful local
`export` or draft upload alone is not benchmark evidence or a publication.

## Paper Ingestion Links

After upload, reference:

- release URL from `release_url`
- archive URL from `release_asset_url`
- DOI URL from `doi_url`
