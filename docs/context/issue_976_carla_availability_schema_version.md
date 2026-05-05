# Issue #976 CARLA Availability Schema Version

## Scope

Issue #976 adds `schema_version: "carla-availability.v1"` to CARLA bridge availability metadata.
The field is emitted by `check_carla_availability()` in both available and not-available cases and
therefore appears in `robot-sf-check-carla --json`.

Existing fields remain intact:

- `available`
- `status`
- `reason`
- `dependency`

## Boundary

This change only versions the local availability metadata contract. It does not install CARLA,
start CARLA, run replay, change dependency metadata, or compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_missing_carla_reports_not_available tests/carla_bridge/test_t0_export.py::test_importable_carla_reports_available tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_json_status -q
```

Failed before implementation because `schema_version` was absent from helper and CLI JSON metadata.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_missing_carla_reports_not_available tests/carla_bridge/test_t0_export.py::test_importable_carla_reports_available tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_json_status -q
```

Passed after adding the schema version field: `3 passed`.
