# Issue #1348 Capability-Aware Map Catalog Design

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1348>

## Decision

Robot SF should treat map validity as capability-specific instead of enforcing one universal
`MapDefinition` construction rule. A map can be a valid asset, route-only fixture, obstacle source,
template, or benchmark candidate without satisfying every runtime capability.

Use a two-layer model:

1. A tracked human-reviewable catalog in `maps/registry.yaml`.
2. Optional generated parser metadata/cache artifacts keyed by source hash, parser version, and
   catalog schema version.

The SVG source remains canonical. Generated converted-map payloads are rebuildable artifacts, not
the source of truth, unless a future issue deliberately promotes a specific small fixture.

## Current Surfaces

- `maps/registry.yaml` currently stores only `version`, `map_id`, and `path`.
- `scripts/tools/generate_map_registry.py` regenerates the current v1 registry from
  `maps/svg_maps/**/*.svg` filenames and should become the migration point for source hashes and
  parser metadata.
- `robot_sf/training/scenario_loader.py` resolves scenario `map_id` references through
  `maps/registry.yaml`, honoring the `ROBOT_SF_MAP_REGISTRY` override. This is the first runtime
  consumer that should call a capability-aware resolver instead of treating registry resolution as
  only a path lookup.
- `robot_sf/maps/verification/map_inventory.py` discovers `maps/svg_maps/**/*.svg`, infers a few
  filename tags, and does not yet read `maps/registry.yaml`.
- `robot_sf/maps/verification/runner.py` applies structural rules and records `FactoryType.ROBOT`
  or `FactoryType.PEDESTRIAN`, but real environment instantiation is intentionally skipped.
- `scripts/validation/verify_maps.py` exposes the verification runner and can write JSON manifests.
- `robot_sf/nav/svg_map_parser.py` contains the parser-derived facts needed for cataloging:
  explicit robot/pedestrian spawn and goal zones, robot/pedestrian routes, route-only mode,
  synthetic route endpoint zones, obstacles, bounds, single-pedestrian markers, and warnings.

## Catalog Schema

Extend `maps/registry.yaml` to this shape:

```yaml
version: 2
schema: robot_sf.map_catalog.v2
parser_version: svg_map_parser.v1
maps:
  - map_id: classic_doorway
    path: svg_maps/classic_doorway.svg
    source_sha256: "<sha256>"
    source_mtime_utc: "2026-05-20T00:00:00Z"
    role: benchmark_candidate
    capabilities:
      robot_runtime: true
      pedestrian_runtime: true
      route_only: false
      obstacle_source: true
      benchmark_candidate: true
    profile: benchmark_candidate
    parser_metadata:
      width: 20.0
      height: 20.0
      bounds_count: 4
      obstacle_count: 7
      robot_spawn_zones: 1
      robot_goal_zones: 1
      ped_spawn_zones: 1
      ped_goal_zones: 1
      robot_routes: 1
      ped_routes: 1
      robot_route_only: false
      ped_route_only: false
      synthetic_robot_zones: 0
      synthetic_ped_zones: 0
    limitations: []
    validation:
      status: pass
      checked_at: "2026-05-20T00:00:00Z"
      rule_ids: []
```

Required top-level map fields:

- `map_id`: stable logical ID; unique across the catalog.
- `path`: repository-root-relative path under `maps/`.
- `source_sha256`: hash of the SVG source used for stale detection.
- `role`: one of `benchmark_candidate`, `robot_runnable`, `pedestrian_runnable`,
  `route_only`, `obstacle_only`, `template`, `example`, or `invalid`.
- `capabilities`: explicit booleans for call-site checks.
- `profile`: default validation profile.
- `parser_metadata`: parser-derived facts from the current SVG.
- `limitations`: stable labels such as `route_derived_zones`, `no_robot_routes`,
  `no_pedestrian_routes`, `obstacle_repair_applied`, `self_intersection_repaired`,
  `template_not_runtime`, or `missing_required_runtime_zones`.
- `validation`: latest catalog sync/verification status.

## Validation Profiles

Profiles define what a caller can require. They should not imply that every map must satisfy every
profile.

| Profile | Required Capabilities | Missing Features Are |
| --- | --- | --- |
| `asset` | readable SVG, parseable dimensions/bounds when applicable | limitations |
| `route_only` | at least one robot or pedestrian route, endpoint-derived zones allowed | limitations unless routes are malformed |
| `robot_runtime` | robot spawn source and goal source, robot route or route synthesis, bounds | fail-closed |
| `pedestrian_runtime` | pedestrian spawn source and goal source or single-ped markers, bounds | fail-closed |
| `benchmark_candidate` | robot runtime plus declared scenario/benchmark intent and no invalid parser warnings | fail-closed |
| `obstacle_only` | obstacles or bounds usable as geometry source | limitations for missing routes/zones |
| `template` | parseable example/template asset | limitations for runtime gaps |
| `invalid` | explicitly unusable until repaired | fail-closed for all runtime profiles |

Spawn/goal source should be recorded separately from capability:

- `explicit_zone`: explicit `robot_spawn_zone` / `robot_goal_zone` or pedestrian equivalents.
- `route_endpoint_derived`: parser generated small endpoint zones from route-only paths.
- `single_ped_marker`: single-pedestrian start/goal markers.
- `missing`: unavailable for that actor type.

Route-only maps with valid route paths should be classified as `route_only` or runnable for the
specific actor if the endpoint-derived zones are acceptable for the requested profile. They should
not be blanket-invalidated by missing explicit rectangles.

## Runtime Behavior

Add a capability-aware map resolver before benchmark/runtime execution:

```text
resolve_map(map_id, required_profile="benchmark_candidate")
  -> load catalog row
  -> verify source path/hash/parser schema are current
  -> verify capabilities satisfy required_profile
  -> return SVG path or generated cache pointer
```

If a caller requests a profile the map does not provide, fail closed with a message naming:

- requested profile,
- missing capability,
- source path,
- catalog row status,
- limitation labels.

Low-level parser tests and asset inspection tools may continue to call `convert_map(...)` directly
when they are testing parser behavior rather than runtime readiness.

The first consumer should be scenario loading: `robot_sf/training/scenario_loader.py` already
centralizes `map_id` lookup and supports `ROBOT_SF_MAP_REGISTRY`, so it can require
`robot_runtime` or `benchmark_candidate` depending on the caller without changing every scenario
YAML file at once.

## Registry Sync Check

Add a sync command, for example:

```bash
uv run python scripts/validation/verify_map_catalog.py --registry maps/registry.yaml --scope all
```

The sync check should fail on:

- catalog path missing on disk,
- SVG file under `maps/` missing from the catalog unless explicitly ignored,
- duplicate `map_id`,
- duplicate path with different `map_id` unless intentionally aliased,
- stale `source_sha256`,
- stale `parser_version` / catalog schema version,
- parser metadata mismatch,
- declared capability contradicted by parser facts,
- benchmark profile assigned to template/example/obstacle-only maps.

The check can warn, not fail, on unknown optional metadata while the catalog migrates.

`scripts/tools/generate_map_registry.py` should grow a mode that preserves reviewed fields while
refreshing computed fields:

```bash
uv run python scripts/tools/generate_map_registry.py \
  --map-root maps \
  --output maps/registry.yaml \
  --schema-version 2 \
  --preserve-reviewed-fields
```

The generator should not silently erase `role`, `capabilities`, `profile`, or reviewed
`limitations`. Computed fields such as `source_sha256`, parser metadata, and stale validation status
may be refreshed.

## Generated Cache Policy

Converted `MapDefinition` caches are optional and should be introduced only after the catalog is in
place. If used, store them under ignored `output/` by default, or a deliberate durable artifact store
for expensive campaign workflows.

Cache key:

```text
sha256(svg bytes + parser_version + map_catalog_schema + normalized parser options)
```

Cache metadata must include the source path, source hash, parser version, catalog schema version,
generated timestamp, and capability/profile used. A stale cache is a miss, not an error, unless a
workflow explicitly requires prehydrated artifacts.

## Follow-Up Implementation Split

The design should land first. Then split implementation into small issues:

1. Catalog schema and sync checker for `maps/registry.yaml`.
2. Parser metadata extractor for route-only/explicit-zone/synthetic-zone facts.
3. Capability-aware runtime/benchmark resolver.
4. Optional generated converted-map cache once the resolver contract is stable.

Created follow-up issues:

- Issue #1413: map catalog schema and sync checker.
- Issue #1414: parser-derived capability metadata.
- Issue #1415: capability-aware scenario/runtime resolver.
- Issue #1416: generated converted-map cache evaluation.

## Validation

Design validation for this note:

```bash
uv run python scripts/validation/verify_maps.py --scope all --mode local \
  --output output/validation/map_verification_issue1348_design.json
```

Expected interpretation: current verification passing is not enough to prove benchmark capability,
because environment instantiation is intentionally skipped. It is only evidence that existing
structural rules can be reused as one input to the catalog.

## Completion Boundary

This note satisfies the design spike for #1348. It does not implement catalog enforcement, generated
cache loading, or runtime resolver behavior; those belong in follow-up issues so reviewers can land
the schema without mixing it with call-site migration risk.
