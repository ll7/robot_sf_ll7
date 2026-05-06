# Issue #998 CARLA Bridge Schema Catalog Schema

## Scope

Issue #998 adds a packaged JSON Schema for the CARLA bridge schema catalog metadata returned by
`list_carla_bridge_schema_catalog()`. The public `load_schema_catalog_schema()` helper loads the
schema without importing CARLA, and the catalog output is validated against that schema in tests.

The schema contract covers the catalog version marker and the per-schema `name`, `loader`, and
`schema_version` fields exposed by issue #994. The final catalog now also lists the catalog schema
itself so `robot-sf-catalog-carla-schemas` remains complete after the new loader is added.

## Boundary

This change only adds a machine-readable schema for catalog metadata and makes that schema
discoverable from the catalog output. It does not change CARLA installation, replay execution,
scenario export, or simulator metric comparison.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_schema_catalog_validates_against_schema -q
```

Failed before implementation with `ImportError` because `load_schema_catalog_schema` did not
exist.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_schema_catalog_validates_against_schema tests/carla_bridge/test_t0_export.py::test_schema_catalog_lists_all_carla_bridge_contracts -q
```

Passed after adding the packaged schema and public loader: `2 passed in 13.01s`.
