# Issue #1413 Map Catalog Schema And Sync Checker

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1413>

## Goal

Issue #1413 implements the first capability-aware `maps/registry.yaml` schema from the
Issue #1348 design note and adds a local sync checker. This slice keeps runtime enforcement out of
scope; the catalog records parser-derived facts and reviewed capability fields so future runtime
resolvers can fail closed without guessing from filenames.

## Implemented Contract

`maps/registry.yaml` is now a v2 catalog with:

- top-level `schema: robot_sf.map_catalog.v2`,
- top-level `parser_version: parser-capability-metadata.v1`,
- per-row `source_sha256`,
- explicit capability booleans for robot runtime, pedestrian runtime, route-only behavior,
  obstacle-source use, and benchmark-candidate use,
- parser metadata from `SvgInspectionReport.capability_metadata`,
- reviewed `role`, `profile`, `limitations`, and `validation` fields.

`scripts/tools/generate_map_registry.py` refreshes computed fields from the SVG source while
preserving reviewed row fields when they already exist. Existing logical aliases such as
`classic_cross_trap` are preserved when they point to a real SVG path.

New rows do not default to `benchmark_candidate: true`; benchmark candidacy is treated as a
reviewed intent claim, while parser-derived robot/pedestrian runtime capabilities can be inferred.

`scripts/validation/verify_map_catalog.py` checks:

- v2 schema and parser version,
- duplicate `map_id` values,
- missing catalog paths,
- absolute or out-of-catalog paths that would make the registry machine-local,
- stale source hashes,
- stale parser metadata,
- unregistered SVG files under the selected map root,
- declared capability booleans that contradict parser-derived facts.

## Boundaries

The sync checker intentionally does not enforce map selection at runtime. Capability-aware scenario
or benchmark resolution remains the follow-up tracked by the runtime resolver issue from the
Issue #1348 split.

The catalog allows intentional path aliases because `classic_cross_trap` already points at
`svg_maps/classic_crossing.svg`. Duplicate path policy can become stricter later after aliases have
an explicit reviewed marker.

## Validation

Targeted proof for this slice:

```bash
uv run ruff check scripts/tools/generate_map_registry.py scripts/validation/verify_map_catalog.py tests/validation/test_verify_map_catalog.py
uv run pytest tests/validation/test_verify_map_catalog.py -q
uv run pytest tests/maps -q
uv run python scripts/tools/generate_map_registry.py --check --map-root maps/svg_maps --output maps/registry.yaml
uv run python scripts/validation/verify_map_catalog.py --registry maps/registry.yaml --scope all
uv run python scripts/validation/check_docs_proof_consistency.py --base origin/issue-1414-parser-capability-metadata
```

The catalog checker was also run against the full tracked `maps/svg_maps` tree and reported:

```text
Map catalog is synchronized: maps/registry.yaml
```
