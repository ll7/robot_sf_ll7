# Issue #994 CARLA Bridge Schema Catalog API

## Scope

Issue #994 adds `list_carla_bridge_schema_catalog()`, an import-safe helper that returns
deterministic JSON-safe metadata for CARLA bridge schemas. The catalog is versioned as
`carla-bridge-schema-catalog.v1` and lists:

- `availability`
- `t0_export_payload`
- `t0_export_manifest`
- `t0_batch_validation_summary`

Each entry reports the schema name, schema version, and public loader helper.

## Boundary

This change only reports schema metadata. It does not add a schema catalog CLI, change existing
schema contents, change CLI outputs, install CARLA, start CARLA, run replay, or compare simulator
metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_schema_catalog_lists_all_carla_bridge_contracts -q
```

Failed before implementation because `list_carla_bridge_schema_catalog` was not exported by
`robot_sf_carla_bridge`.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_schema_catalog_lists_all_carla_bridge_contracts tests/carla_bridge/test_t0_export.py::test_missing_carla_reports_not_available -q
```

Passed after adding the catalog helper: `2 passed in 13.08s`.
