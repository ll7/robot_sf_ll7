# Issue 2728: Semantic Boundaries - Evidence (2026-06-14)

## Claim
Diagnostic-only proof that 2D semantic boundary metadata can be parsed from SVG
maps, carried through MapDefinition, and queried by downstream consumers without
affecting collision physics or planner behaviour.

## Evidence Classification
`diagnostic_only_not_benchmark` - This evidence proves the metadata plumbing
works. It does not make any benchmark, paper, or safety claim.

## What Was Validated
1. **Parser success**: `SemanticBoundary` dataclass is populated from SVG path
   labels with the `semantic_boundary_` prefix. Flags are parsed from `__`-separated
   tokens after the boundary name.
2. **Unsupported token fail-closed**: An unsupported token in a semantic boundary
   label raises `ValueError`.
3. **Robot route avoidance**: The authored robot route does not cross the
   `vehicle_blocking` separator, while a direct spawn-to-goal line would.
4. **Pedestrian emergence**: The pedestrian start position is within the declared
   `max_distance` of the `pedestrian_passable` `spawn_edge` boundary.

## Files
- `maps/svg_maps/issue_2728_semantic_boundaries.svg` - Fixture map
- `configs/scenarios/single/issue_2728_semantic_boundaries.yaml` - Scenario definitions
- `robot_sf/nav/nav_types.py` - `SemanticBoundary` dataclass
- `robot_sf/nav/map_config.py` - `MapDefinition.semantic_boundaries` field
- `robot_sf/nav/svg_map_parser.py` - Parser extension
- `scripts/validation/validate_semantic_boundaries_issue_2728.py` - Validation script
- `tests/test_semantic_boundaries_issue_2728.py` - Test suite

## Risks
- This is metadata-only. No collision physics, planner logic, or runtime
  behaviour was changed.
- Future consumers of `semantic_boundaries` must be implemented in follow-up work.
