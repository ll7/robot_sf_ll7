# Issue #966 CARLA T0 Batch Validation JSON Summary

## Scope

Issue #966 adds a machine-readable summary mode to the CARLA-free T0 batch validation CLI from
issue #964. The CLI still delegates validation to the issue #962 manifest payload loader, then can
emit a JSON object with:

- `manifest`: the validated manifest path as passed to the CLI,
- `payload_count`: number of loaded payloads,
- `scenario_ids`: scenario ids in manifest/load order.

The default text output remains available for humans and existing scripts.

## Boundary

This is a local validation summary only. It does not execute CARLA, compare simulator trajectories,
upload durable artifacts, or change the T0 export writer format.

## Validation

RED proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_prints_json_summary -q
```

Failed before implementation because `--json` was not recognized by the batch validation CLI.

GREEN proof:

```bash
rtk uv run pytest tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_prints_json_summary tests/carla_bridge/test_t0_export_cli.py::test_validate_t0_export_batch_main_loads_payloads_and_prints_count -q
```

Passed after adding the JSON summary branch: `2 passed`.
