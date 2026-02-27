# Camera-Ready Benchmark Release Workflow

## Purpose

This runbook publishes the campaign publication bundle as a GitHub release asset
with checksum and manifest verification.

## Prerequisites

- Completed camera-ready campaign output containing:
  - `reports/campaign_summary.json`
  - `publication_bundle.archive_path`
  - `publication_bundle.checksums_path`
  - `publication_bundle.manifest_path`
- `gh` CLI authenticated for repository upload.

## Recommended Tag Naming

Use immutable paper-facing tags:

- `camera-ready-v1.0.0`
- `paper-matrix-v1-<date>`

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
