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

## Command Path

1. Dry-run validation + command plan:

```bash
uv run python scripts/tools/publish_camera_ready_release.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --repo ll7/robot_sf_ll7 \
  --tag <release_tag> \
  --output-json output/benchmarks/camera_ready/<campaign_id>/reports/release_publish_plan.json
```

2. Execute asset upload:

```bash
uv run python scripts/tools/publish_camera_ready_release.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --repo ll7/robot_sf_ll7 \
  --tag <release_tag> \
  --execute-upload
```

## Validation Checklist

- `release_publish_plan.json` contains expected paths and URLs.
- `checksums.sha256` is non-empty and references bundle files.
- Release page contains archive + checksums + manifest assets.
- Campaign summary contains URL placeholders:
  - `release_url`
  - `release_asset_url`
  - `doi_url`

## Paper Ingestion Links

After upload, reference:

- release URL from `release_url`
- archive URL from `release_asset_url`
- DOI URL from `doi_url`
