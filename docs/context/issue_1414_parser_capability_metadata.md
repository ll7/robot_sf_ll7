# Issue #1414 Parser Capability Metadata

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1414>

## Decision

Parser-derived map capability facts are now exposed through
`robot_sf/maps/verification/svg_inspection.py`. The existing `inspect_svg(...)` API and
`scripts/validation/svg_inspect.py --json ...` CLI include a `capability_metadata` payload alongside
route summaries and inspection findings.

This keeps Issue #1414 scoped to metadata extraction. Catalog writes, registry sync enforcement, and
runtime scenario resolution remain follow-up work for Issues #1413 and #1415.

## Metadata Contract

The payload is versioned as `parser-capability-metadata.v1` and records:

- explicit and parsed robot/pedestrian spawn and goal zone counts,
- synthetic-zone deltas produced by parser fallback paths,
- robot and pedestrian route counts,
- obstacle, bounds, crowded-zone, and single-pedestrian marker counts,
- route-only mode booleans for robot and pedestrian routes,
- boolean capability hints such as explicit runtime zones, runtime routes, and obstacle presence,
- parser limitation/finding codes such as `PED_ROUTE_ONLY_MODE`.

Route-only maps are represented as having routes while lacking explicit pedestrian or robot runtime
zones. Obstacle-only/template-style SVGs are represented as obstacle sources without runtime routes
or explicit runtime zones.

## Validation

```bash
uv run pytest tests/maps/test_svg_inspection.py -q

uv run python scripts/validation/svg_inspect.py \
  maps/svg_maps/classic_crossing.svg \
  --json output/validation/issue1414_svg_inspect.json \
  --show-routes
```

The JSON smoke confirmed `classic_crossing.svg` emits `capability_metadata` with
`ped_route_only_mode=true`, explicit robot runtime zones, pedestrian routes, no explicit pedestrian
runtime zones, and `parser_limitation_codes=["PED_ROUTE_ONLY_MODE"]`.

## Follow-Up Boundary

Issue #1413 should consume this payload when adding the registry schema and sync checker.
Issue #1415 should decide which fields become fail-closed runtime resolver requirements.
