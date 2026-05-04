# Issue #974 CARLA Availability Boolean Field

## Scope

Issue #974 adds `available: bool` to CARLA bridge availability metadata. The existing
`status`, `reason`, and `dependency` fields remain intact for backwards-compatible text and script
consumers.

The field is emitted by `check_carla_availability()` and therefore also appears in
`robot-sf-check-carla --json` output. It lets setup scripts branch on a boolean instead of parsing
`status` strings.

## Boundary

This change only extends local availability metadata. It does not install CARLA, start CARLA, run
replay, change dependency metadata, or compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_missing_carla_reports_not_available tests/carla_bridge/test_t0_export.py::test_importable_carla_reports_available tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_json_status -q
```

Failed before implementation because the `available` key was absent from helper and CLI JSON
metadata.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export.py::test_missing_carla_reports_not_available tests/carla_bridge/test_t0_export.py::test_importable_carla_reports_available tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_json_status -q
```

Passed after adding the boolean field: `3 passed`.
