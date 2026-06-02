# Issue #2008 Artifact Catalog Contract (2026-06-01)

## Goal

Issue #2008 adds the first `artifact_catalog.v1` contract so reusable figures and
tables can be addressed by stable semantic IDs instead of regenerated paths.

## Contract

- Loader and typed metadata: `robot_sf/benchmark/artifact_catalog.py`
- JSON Schema: `robot_sf/benchmark/schemas/artifact_catalog.v1.json`
- Validator CLI: `scripts/validation/validate_artifact_catalog.py`
- Fixture catalog: `tests/fixtures/artifact_catalog/v1/valid_catalog.yaml`
- User-facing docs: `docs/artifact_catalog.md`

The validator fails on duplicate `artifact_id` values, missing source files,
missing output files, checksum mismatches, and local-only durable references such
as `output/`.

## Boundary

This is a schema and validation slice only. It does not choose final figure/table
IDs for future papers, rewrite historical figure folders, or implement full
benchmark table/figure generators.

The fixture catalog is contract proof only and is not benchmark evidence.

## Validation

Targeted validation for this slice:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest tests/benchmark/test_artifact_catalog.py -q
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/validation/validate_artifact_catalog.py tests/fixtures/artifact_catalog/v1/valid_catalog.yaml
python -m json.tool robot_sf/benchmark/schemas/artifact_catalog.v1.json
git diff --check
```
