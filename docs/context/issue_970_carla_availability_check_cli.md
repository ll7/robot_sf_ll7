# Issue #970 CARLA Availability Check CLI

## Scope

Issue #970 exposes the CARLA bridge availability metadata through
`robot-sf-check-carla`. The CLI calls `check_carla_availability()` and reports the dependency,
status, and reason in either text or deterministic JSON form.

The command is intentionally CARLA-free: it reports whether the optional CARLA Python API is
importable without starting a simulator or importing replay code.

## Boundary

This change does not install CARLA, start CARLA, run replay, change dependency metadata, or compare
Robot-SF/CARLA metrics. It only provides a setup-check surface for future replay workflows.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_json_status tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_text_status tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q
```

Failed before implementation because `check_carla_availability_main` and the
`robot-sf-check-carla` project script were absent.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_json_status tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_text_status tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q
```

Passed after adding the CLI and script metadata: `3 passed`.
