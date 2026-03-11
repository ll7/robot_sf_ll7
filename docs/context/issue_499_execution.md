# Issue 499 Execution Notes: Public Benchmark Artifact Publication

Issue: <https://github.com/ll7/robot_sf_ll7/issues/499>

## Implemented

- Added reusable publication helpers in:
  - `robot_sf/benchmark/artifact_publication.py`
- Added CLI tool in:
  - `scripts/tools/benchmark_publication_bundle.py`

Provided capabilities:
- discover benchmark run directories (`discover_run_directories`)
- measure artifact size ranges across runs (`measure_artifact_size_ranges`)
- export DOI-ready bundle with:
  - payload
  - `publication_manifest.json` (`benchmark-publication-bundle.v1`)
  - `checksums.sha256`
  - compressed archive (`.tar.gz`)

## Tests Added

- `tests/benchmark/test_artifact_publication.py`
- `tests/tools/test_benchmark_publication_bundle.py`

Coverage focus:
- file discovery and video include/exclude behavior
- size-report schema and distribution output
- export manifest/checksum/archive generation
- CLI command behavior for `size-report` and `export`

## Publication Policy Docs

- Added `docs/benchmark_artifact_publication.md`:
  - public channel policy (Git vs Release assets vs Zenodo)
  - reproducible export command path
  - DOI-ready release flow
  - citation URL templates
  - retention policy

## Size Measurements

- Added generated report:
  - `docs/context/issue_499_artifact_size_report_2026-02-16.json`
- Generated via:
  - `uv run python scripts/tools/benchmark_publication_bundle.py size-report ...`

