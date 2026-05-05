# Issue #996 CARLA Bridge Schema Catalog CLI

## Scope

Issue #996 adds `robot-sf-catalog-carla-schemas`, a no-argument CLI that prints the
`list_carla_bridge_schema_catalog()` metadata as deterministic JSON. The command is packaged as a
project script and remains import-safe without CARLA installed.

Existing schema-specific CLIs remain unchanged.

## Boundary

This change only exposes schema catalog metadata through a command-line surface. It does not change
schema catalog contents, change individual schema CLIs, install CARLA, start CARLA, run replay, or
compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_catalog_carla_schemas_main_prints_catalog tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q
```

Failed before implementation because `catalog_carla_schemas_main` and the
`robot-sf-catalog-carla-schemas` project script were missing.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_catalog_carla_schemas_main_prints_catalog tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q
```

Passed after adding the CLI and script entry: `2 passed in 20.85s`.
