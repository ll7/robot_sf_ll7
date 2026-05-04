# Issue #984 CARLA T0 Export Manifest Schema

## Scope

Issue #984 adds a packaged JSON Schema for `carla-replay-export-manifest.v1` and exposes it through
`load_export_manifest_schema()`. Generated export manifests now validate against that schema in the
manifest reader after the existing clear Python-level checks for version, list shape, non-empty
scenario IDs, and non-empty payload paths.

The manifest schema captures the stable batch metadata contract:

- `schema_version: carla-replay-export-manifest.v1`
- `exports[]`
- each export entry's `scenario_id`
- each export entry's relative payload `path`

Unsafe absolute paths and parent-directory escapes remain path-semantics checks in
`resolve_export_manifest_payload_paths(...)`.

## Boundary

This change does not alter manifest file layout, load referenced payload contents, add manifest
schema CLI output, install CARLA, start CARLA, run replay, or compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_export_manifest_validates_against_schema -q
```

Failed before implementation because `load_export_manifest_schema` was not exported by
`robot_sf_carla_bridge`.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_export_manifest_validates_against_schema tests/carla_bridge/test_t0_export.py::test_export_manifest_schema_rejects_malformed_entries tests/carla_bridge/test_t0_export.py::test_read_export_manifest_round_trips_writer_output tests/carla_bridge/test_t0_export.py::test_read_export_manifest_rejects_invalid_manifest_shape -q
```

Passed after adding the manifest schema resource and loader: `4 passed in 13.38s`.
