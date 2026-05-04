# Issue #942 CARLA T0 MapDefinition Adapter

## Goal

Issue #942 adds a CARLA-free adapter from an already-certified Robot-SF `MapDefinition` to the
`carla-replay-export.v1` neutral payload. It is a child of issue #872 and is stacked on the issue
#930 schema, issue #934 builder, and issue #940 read-helper work.

## Decision

`build_export_payload_from_map_definition(...)` exports:

- the selected Robot-SF robot route as neutral robot start/goal,
- `single_pedestrians` as scripted pedestrian starts/routes,
- static obstacle polygons and map bounds under `static_geometry`,
- scenario identity, simulation settings, trajectory fields, and provenance through the existing
  schema-validated builder.

The adapter fails closed when the supplied `scenario_cert.v1` payload is excluded or has a status
outside the exportable set: `passed`, `valid`, `hard_but_solvable`, or `knife_edge`.

## Boundary

This is T0 neutral JSON construction only. It does not import CARLA, convert coordinates into a
CARLA world, run simulator replay, or claim transfer parity.

## Validation

Proof command for the implementation branch:

```bash
uv run pytest tests/carla_bridge/test_t0_export.py -q
```

The RED run failed with two missing-import failures for
`build_export_payload_from_map_definition`. After implementation, the focused CARLA-free test file
passed with `11 passed`.
