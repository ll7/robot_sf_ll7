# Issue #982 CARLA T0 Export Schema CLI

## Scope

Issue #982 extends `robot-sf-export-carla-t0` with `--schema`, which prints the packaged
`carla-replay-export.v1` JSON Schema as deterministic JSON. The command returns `0` without
requiring a scenario file, output directory, commit provenance, CARLA installation, or replay
execution.

Existing export behavior remains intact: normal export mode still requires `--scenario-file`,
`--output-dir`, and `--robot-sf-commit`.

## Boundary

This change only exposes the existing local export schema through the CLI. It does not change the
schema contents, add manifest schema output, install CARLA, start CARLA, run replay, or compare
simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_export_t0_scenarios_main_prints_schema -q
```

Failed before implementation because `--schema` still required `--scenario-file`, `--output-dir`,
and `--robot-sf-commit`.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_export_t0_scenarios_main_prints_schema tests/carla_bridge/test_t0_export_cli.py::test_export_t0_scenarios_main_writes_records_and_prints_manifest -q
```

Passed after adding schema mode: `2 passed in 12.63s`.
