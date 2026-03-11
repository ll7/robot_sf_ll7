# Benchmark Artifact Publication (DOI-Ready)

## Purpose

This guide defines a publication-safe artifact policy for benchmark outputs and
documents the reproducible export path for camera-ready artifacts.

Use this when you need public, stable references for papers based on
`robot_sf_ll7` benchmark runs.

## Public Artifact Policy

| Channel | What belongs there | Why |
| --- | --- | --- |
| Git repository (`main`) | Schemas, docs, scripts, compact metadata examples | Reviewable, versioned, lightweight |
| GitHub Release assets | Export bundles (`*.tar.gz`) with checksums + manifest | Immutable per tag, easy to download |
| Zenodo (linked to GitHub release) | Final camera-ready bundle snapshot with DOI | Citable long-term DOI endpoint |

Rules:
- Do not require private repository paths for paper evidence.
- Paper references should point to GitHub release URLs and/or Zenodo DOI links.
- Keep generated large artifacts out of source control unless intentionally tiny
  and review-critical.

## Bundle Format (v1)

Export bundles are produced by
`scripts/tools/benchmark_publication_bundle.py export` and contain:

- `payload/`: run files selected for publication.
- `publication_manifest.json`:
  - `schema_version: benchmark-publication-bundle.v1`
  - provenance (run id, optional run meta/manifests)
  - publication channel metadata (repo URL, release tag, DOI string)
  - per-file metadata (`path`, `size_bytes`, `sha256`, `kind`)
- `checksums.sha256`: SHA-256 checksums for payload files.
- `<bundle_name>.tar.gz`: archive for release upload.

## Command Path (Reproducible)

1. Measure current benchmark artifact sizes (optional but recommended):

```bash
uv run python scripts/tools/benchmark_publication_bundle.py size-report \
  --benchmarks-root output/benchmarks \
  --output-json docs/context/issue_499_artifact_size_report_2026-02-16.json
```

2. Export a publication bundle for one run:

```bash
uv run python scripts/tools/benchmark_publication_bundle.py export \
  --run-dir output/benchmarks/<run_dir> \
  --out-dir output/benchmarks/publication \
  --bundle-name <run_dir>_publication_bundle \
  --release-tag vX.Y.Z \
  --doi 10.5281/zenodo.<record-id>
```

This is the canonical command path required for publication exports.

## DOI-Capable Release Flow

1. Create a Git tag/release for the code state.
2. Upload the exported `*.tar.gz` bundle as a GitHub release asset.
3. Ensure Zenodo-GitHub integration is enabled for the repository.
4. Trigger/archive the release in Zenodo and obtain DOI.
5. Update paper references to the Zenodo DOI and release asset URL.

## Citation-Ready URL Templates

- Release page:
  - `https://github.com/ll7/robot_sf_ll7/releases/tag/<tag>`
- Release asset:
  - `https://github.com/ll7/robot_sf_ll7/releases/download/<tag>/<bundle>.tar.gz`
- DOI:
  - `https://doi.org/10.5281/zenodo.<record-id>`

## Retention Policy

- Release assets: retain for all paper-referenced tags.
- Zenodo records: immutable archive of camera-ready evidence.
- Local `output/` artifacts: keep latest working sets; prune transient runs that
  are not release candidates.

## Current Size Snapshot (2026-02-16)

Measured with:
- `scripts/tools/benchmark_publication_bundle.py size-report --include-videos`
- report artifact: `docs/context/issue_499_artifact_size_report_2026-02-16.json`

Observed distribution across 38 discovered run directories:
- total bytes: min `2,664`, p50 `448,538`, p90 `744,413`, max `916,188`
- episode payload bytes: p50 `179,744`, p90 `435,719`
- aggregate payload bytes: p50 `153,187`, p90 `180,226`
- report payload bytes: p50 `155,916`, p90 `181,913`

## Related Files

- `robot_sf/benchmark/artifact_publication.py`
- `scripts/tools/benchmark_publication_bundle.py`
- `docs/context/issue_499_artifact_size_report_2026-02-16.json`
