# Issue #992 CARLA T0 Batch Summary Schema CLI

## Scope

Issue #992 extends `robot-sf-validate-carla-t0-batch` with `--schema`, which prints the packaged
`carla-replay-export-batch-validation-summary.v1` JSON Schema as deterministic JSON. The command
returns `0` without requiring `--manifest`, loading payload files, installing CARLA, or running
replay.

Existing batch validation behavior remains intact:

- text mode still requires `--manifest` and reports the validated payload count,
- `--json` still requires `--manifest` and prints the versioned validation summary.

## Boundary

This change only exposes the existing local batch-summary schema through the CLI. It does not
change summary schema contents, change manifest or payload validation semantics, install CARLA,
start CARLA, run replay, or compare simulator metrics.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_prints_schema -q
```

Failed before implementation because `--schema` still required `--manifest`.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_prints_schema tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_prints_json_summary tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_loads_payloads_and_prints_count -q
```

Passed after adding schema mode: `3 passed in 12.69s`.
