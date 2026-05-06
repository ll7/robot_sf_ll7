# Issue #972 CARLA Availability CLI Require Mode

## Scope

Issue #972 extends `robot-sf-check-carla` with `--require`. In normal report-only mode, the CLI
continues to return `0` whether CARLA is available or not. In require mode, it returns `1` when the
availability status is not `available`, while preserving the same text or JSON output.

This gives CARLA-dependent CI/setup workflows a fail-closed check without requiring normal
Robot-SF or T0 export workflows to install CARLA.

## Boundary

This change does not install CARLA, start CARLA, run replay, change dependency metadata, or compare
simulator metrics. It only changes the process exit code policy for an explicit setup-check mode.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_require_fails_when_unavailable tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_require_passes_when_available -q
```

Failed before implementation because `--require` was not recognized.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_require_fails_when_unavailable tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_require_passes_when_available tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_json_status tests/carla_bridge/test_t0_export_cli.py::test_check_carla_availability_main_prints_text_status -q
```

Passed after adding require-mode exit handling: `4 passed`.
