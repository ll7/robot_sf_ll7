# Artifact Catalog v1

`artifact_catalog.v1` records stable semantic IDs for reusable figures and tables.
It separates the identity of an artifact from whatever generated path currently
stores the rendered file.

Use this catalog when a report, paper draft, or follow-up workflow needs to cite
the same figure or table across reruns.

## Required Fields

Each catalog has:

- `schema_version: artifact_catalog.v1`
- `catalog_id`: stable catalog identifier
- `artifacts`: one or more figure/table rows

Each artifact row has:

- `artifact_id`: stable semantic ID, such as `fig_benchmark_outcome_matrix`
- `artifact_kind`: `figure` or `table`
- `source_kind`: source class, such as `benchmark_campaign`
- `source_files`: tracked or durable source files with SHA-256 checksums
- `outputs`: generated files keyed by format, each with SHA-256 checksums
- `generation_command`: command that produced the outputs
- `generation_commit`: commit used for generation
- `claim_boundary`: what the artifact can and cannot support
- `caption_file`: optional checksummed caption/source text

The validator fails closed when IDs are duplicated, required files are missing,
checksums mismatch, or a durable reference points at local-only locations such as
`output/`, `.git/`, `.venv/`, `/tmp/`, or a worktree path.

## Example

```yaml
schema_version: artifact_catalog.v1
catalog_id: camera_ready_tables_v1
artifacts:
  - artifact_id: tab_planner_execution_modes
    artifact_kind: table
    source_kind: benchmark_campaign
    source_files:
      - path: reports/comparability_matrix.json
        sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    outputs:
      md:
        path: tables/tab_planner_execution_modes.md
        sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
      tex:
        path: tables/tab_planner_execution_modes.tex
        sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    generation_command: uv run python scripts/tools/export_benchmark_tables.py --catalog
    generation_commit: fbc7e125
    claim_boundary: Benchmark summary table; not raw episode evidence.
```

## Validation

Validate a catalog with:

```bash
uv run python scripts/validation/validate_artifact_catalog.py tests/fixtures/artifact_catalog/v1/valid_catalog.yaml
```

The checked-in fixture is intentionally tiny and exists to prove the schema and
validator contract, not to serve as benchmark evidence.

## Related Work

- `docs/dev/issues/figures-naming/design.md`
- `docs/benchmark_artifact_publication.md`
- GitHub issue #2007 tracks a future benchmark artifact compiler that can consume
  this catalog contract.
- GitHub issue #2008 introduced the first schema and validator slice.
