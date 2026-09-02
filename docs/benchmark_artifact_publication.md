# Benchmark Artifact Publication (DOI-Ready)

## Purpose

This guide defines a publication-safe artifact policy for benchmark outputs and
documents the reproducible export path for the current S30/H600 benchmark-data
release. Benchmark-data publication and software/package release are separate
operations.

For the higher-level benchmark release process, see:

- `docs/benchmark_release_protocol.md`
- `docs/benchmark_release_reproducibility.md`

Use this when you need public, stable references for papers based on
`robot_sf_ll7` benchmark runs.

## Public Artifact Policy

| Channel | What belongs there | Why |
| --- | --- | --- |
| Git repository (`main`) | Schemas, docs, scripts, compact metadata examples | Reviewable, versioned, lightweight |
| GitHub Release assets | Export bundles (`*.tar.gz`) with checksums + manifest | Immutable per tag, easy to download |
| Zenodo (fresh concept) | Final S30/H600 benchmark-data bundle snapshot with DOI | Citable long-term DOI endpoint |

Rules:
- Do not require private repository paths for paper evidence.
- Paper references should point to GitHub release URLs and/or Zenodo DOI links.
- Reserve a fresh Zenodo concept for each benchmark-data release. Do not reuse
  historical concepts `10.5281/zenodo.19482025` or `10.5281/zenodo.19563812`.
- Social Navigation Quality Index (SNQI) is advisory/no-ranking for the current
  S30/H600 release; raw and component metrics remain separately reportable.
- A one-scenario/one-seed runtime smoke is diagnostic execution evidence, not a
  full benchmark result or software release.
- Keep generated large artifacts out of source control unless intentionally tiny
  and review-critical.

## Bundle Format (v2)

Export bundles are produced by
`scripts/tools/benchmark_publication_bundle.py export` and contain:

- `payload/`: run files selected for publication.
- `publication_manifest.json`:
  - `schema_version: benchmark-publication-bundle.v2`
  - provenance (run id, optional run meta/manifests)
  - publication channel metadata (repo URL, release tag, DOI string)
  - per-file metadata (`path`, `size_bytes`, `sha256`, `kind`)
- `checksums.sha256`: SHA-256 checksums for payload files.
- `<bundle_name>.tar.gz`: archive for release upload.

A derived-metadata erratum additionally publishes the bundle-local
`publication_manifest.json` and `checksums.sha256` as detached assets alongside
`publication_custody.json`. The custody receipt binds the complete archive
digest and the embedded erratum receipt without creating a self-checksum cycle.
All four assets must be uploaded under the same names and with identical bytes
to the GitHub and Zenodo successor drafts.

When the source campaign contains both
`release/release_manifest.resolved.json` and `release/release_result.json`, the
exporter also fails closed on missing release metadata and adds a
`release_metadata` block to `publication_manifest.json`. Its payload contains:

- the resolved release manifest and release result;
- the tracked `CITATION.cff` and the exact tracked Zenodo dataset metadata;
- a generated rights/provenance statement with the raw-artifact boundary; and
- checksummed SNQI weights and baseline under `payload/release_metadata/snqi/`.

Raw episode rows and component metrics remain eligible for the payload even when
videos are excluded. Large generated output remains out of Git and must be
promoted through the durable GitHub/Zenodo artifact path before it is cited.
The bundle records `campaign_output: durable-required` and
`local_output: working-storage-not-citation-target` so this distinction survives
a cold download.

There is no standalone publication-bundle schema validator CLI. For release handoff, inspect
`publication_manifest.json` for `schema_version: benchmark-publication-bundle.v2`, verify the
required `release_metadata` roles for a completed release, and run
`sha256sum -c checksums.sha256` from the bundle root before upload. The release runner invokes
the stricter publication preflight, including release-result reconciliation and SNQI consistency.

## Reusable Figure And Table IDs

Use `artifact_catalog.v1` when a publication bundle or benchmark report needs
stable semantic IDs for reusable figures and tables. The catalog records
`artifact_id`, source files, outputs, SHA-256 checksums, generation command,
generation commit, and claim boundary while keeping generated paths replaceable.

See `docs/artifact_catalog.md` and validate catalogs with:

```bash
uv run python scripts/validation/validate_artifact_catalog.py <catalog.yaml>
```

## Benchmark Artifact Compiler

Use `scripts/tools/compile_benchmark_artifacts.py` to turn an existing benchmark
campaign report directory into a compact, cataloged artifact pack for dissertation
draft tables, figures, captions, and export review:

```bash
uv run python scripts/tools/compile_benchmark_artifacts.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --output output/benchmarks/publication_candidates/<campaign_id> \
  --catalog-id <campaign_id>_artifacts
```

Expected output tree:

```text
artifact_catalog.yaml
captions.md
checksums.sha256
figures/
  planner_status_summary.pdf
  planner_status_summary.png
not_available_inputs.json
sources/
  reports/
    <copied source report inputs>
tables/
  campaign_table.csv
  campaign_table.md
  campaign_table.tex
  not_available_inputs.md
```

The compiler preserves fallback, degraded, failed, and `not_available` rows in
the generated campaign table. Missing optional report inputs are recorded in
`not_available_inputs.json` rather than silently omitted. The resulting
`artifact_catalog.yaml` should validate with the catalog command above before an
artifact pack is cited or exported.

Compiler output under `output/` is a local publication-candidate stage, not the
durable citation target. Treat its catalog rows as draft-ready only until the
source campaign and selected artifacts are promoted through a release asset, DOI,
tracked compact evidence copy, or another durable store with checksums. The
compiler's claim boundary is diagnostic-only unless that downstream durable proof
exists and the source benchmark campaign itself satisfies the relevant
camera-ready or paper-facing contract.

Compiler evidence map:

| Artifact id | Files | Caption/checksum surface | Boundary |
| --- | --- | --- | --- |
| `fig_planner_status_summary` | `figures/planner_status_summary.{png,pdf}` | `captions.md`, `checksums.sha256`, and `artifact_catalog.yaml` | Diagnostic status distribution; not metric or planner-quality evidence by itself. |
| `tab_campaign_table` | `tables/campaign_table.{csv,md,tex}` | `captions.md`, `checksums.sha256`, and `artifact_catalog.yaml` | Formatted campaign rows; preserves fallback, degraded, failed, and `not_available` caveats. |
| `tab_not_available_inputs` | `tables/not_available_inputs.md`, `not_available_inputs.json` | `captions.md`, `checksums.sha256`, and `artifact_catalog.yaml` | Missing optional compiler inputs; documents limitations instead of filling gaps. |

## Dissertation Figure/Table Export

Use `scripts/tools/benchmark_publication_bundle.py dissertation-bundle` when a
small set of figure/table candidates needs a dissertation-facing handoff without
turning local `output/` files into durable evidence. The command copies selected
artifacts into a disposable bundle and writes:

- `artifact_manifest.json`: JSON manifest with `schema_version:
  dissertation_artifact_bundle.v1`;
- `checksums.sha256`: checksums for files under `payload/artifacts/`;
- per-artifact rows with `artifact_id`, `source_path`, `source_artifact`,
  `output_path`, `sha256`, `source_commit`, `generation_command`,
  `caption_draft`, `claim_boundary`, `recommended_manuscript_use`,
  `fallback_degraded_summary`, and optionally `chapter_target` and
  `chapter_target_justification`.

The artifact spec is a compact JSON file so reviewers can see exactly which
figure/table candidates are being handed off:

```json
{
  "artifacts": [
    {
      "artifact_id": "tab_campaign_table",
      "source_path": "tables/campaign_table.md",
      "source_artifact": "release-backed campaign table candidate",
      "caption_draft": "Campaign table preserving fallback and degraded rows.",
      "claim_boundary": "Formatted table only; not new benchmark evidence.",
      "recommended_manuscript_use": "discussion",
      "fallback_degraded_summary": "Fallback and degraded rows remain visible.",
      "chapter_target": "Results, Section 4.2"
    }
  ]
}
```

Allowed `recommended_manuscript_use` values are `results`, `methodology`,
`discussion`, `outlook`, and `do-not-use`; unsupported values fail closed.

Optional `chapter_target` is a free-form dissertation chapter/section label
(e.g. `Results, Section 4.2` or `Limitations, Section 5.1`). Diagnostic-only
rows should target limitations, methodology, or future-work style sections
unless `chapter_target_justification` explicitly explains why another target
is appropriate.

Example command:

```bash
uv run python scripts/tools/benchmark_publication_bundle.py dissertation-bundle \
  --source-root output/benchmarks/publication_candidates/<campaign_id> \
  --out-dir output/dissertation_export \
  --bundle-name <campaign_id>_figure_table_bundle \
  --artifact-spec output/benchmarks/publication_candidates/<campaign_id>/dissertation_artifacts.json \
  --command "uv run python scripts/tools/compile_benchmark_artifacts.py --campaign-root output/benchmarks/camera_ready/<campaign_id> --output output/benchmarks/publication_candidates/<campaign_id>" \
  --commit "$(git rev-parse HEAD)"
```

This bundle is a provenance and review workflow only. It does not create new
benchmark evidence, dissertation prose, or paper-grade claims; durable use still
requires a release asset, DOI, tracked compact evidence copy, or another
explicit artifact pointer.

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

This is the canonical command path required for publication exports. Replace the release tag and DOI
placeholders before citing the bundle as paper-facing evidence.

## DOI-Capable Benchmark-Data Release Flow

1. Freeze the approved S30/H600 benchmark-data manifest and immutable benchmark-data tag.
2. Export the bundle and independently cold-verify its archive, manifest, checksums,
   release result, citation, Zenodo metadata, rights statement, and pinned SNQI assets.
3. Create a fresh Zenodo concept with `release zenodo reserve`; do not reuse a
   software or historical benchmark concept.
4. Disable the GitHub-to-Zenodo webhook and leave it disabled, then create a draft
   GitHub Release and upload the byte-identified `*.tar.gz` bundle plus checksum and
   manifest assets.
5. Upload the exact same archive to the reserved Zenodo draft, run read-only
   `release zenodo verify`, and independently compare GitHub/Zenodo downloads.
6. Publish the GitHub Release and Zenodo version only after all acceptance and
   cold-download checks pass. Immediately rerun the authenticated Zenodo
   verification against the published record and require a passing receipt
   before recording the version DOI and parent concept DOI.
7. Update paper references to the verified Zenodo DOI and benchmark-data asset URL.

The doctor and post-publication checks observe the hook list read-only. Their
receipt records an instantaneous disabled/absent state, but cannot prevent
reactivation or another state change in the time-of-check/time-of-use interval
around publication. Recheck immediately before publishing and preserve this
residual boundary in the release record; never describe the snapshot as a
permanent guarantee.

Software/package tags and their DOI records remain separate. This documentation
does not authorize publishing, credential use, webhook changes, or reusing an
existing Zenodo concept.

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
