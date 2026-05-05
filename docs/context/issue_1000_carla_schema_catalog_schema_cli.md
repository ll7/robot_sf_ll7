# Issue #1000 CARLA Bridge Schema Catalog Schema CLI

## Scope

Issue #1000 extends `robot-sf-catalog-carla-schemas` with `--schema`, which prints the
`load_schema_catalog_schema()` JSON Schema as deterministic JSON. This keeps the schema catalog
CLI paired with the import-safe loader added by issue #998.

Existing no-argument catalog output remains unchanged and continues to print
`list_carla_bridge_schema_catalog()` metadata.

## Boundary

This change only exposes the existing schema catalog schema through the CLI. It does not change the
catalog metadata, schema contents, CARLA installation behavior, replay execution, scenario export,
or simulator metric comparison.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_catalog_carla_schemas_main_prints_schema -q
```

Failed before implementation because `--schema` was rejected as an unrecognized argument by
`catalog_carla_schemas_main`.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_catalog_carla_schemas_main_prints_schema tests/carla_bridge/test_t0_export_cli.py::test_catalog_carla_schemas_main_prints_catalog -q
```

Passed after adding schema mode: `2 passed in 12.72s`.

Broader local proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py tests/carla_bridge/test_t0_export.py::test_schema_catalog_validates_against_schema -q
BASE_REF=origin/main rtk scripts/dev/pr_ready_check.sh
```

The targeted CARLA bridge command passed with `21 passed in 13.57s`. The full PR readiness gate
passed with `3190 passed, 15 skipped, 3 warnings in 256.54s`.

## Artifact Decision

Validation refreshed ignored coverage output under `output/coverage/`. The CLI change does not
depend on generated artifacts, model checkpoints, benchmark bundles, or local cache state, so those
ignored files are disposable and are not promoted.
