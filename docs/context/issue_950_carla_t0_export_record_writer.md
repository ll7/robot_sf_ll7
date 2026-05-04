# Issue #950 CARLA T0 Export Record Writer

## Goal

Issue #950 adds a CARLA-free writer for the ordered export records returned by
`build_export_payloads_from_scenario_file(...)`. It is a child of issue #872 and is stacked on the
issue #948 scenario-file export helper.

## Decision

`write_export_records(records, output_dir)` writes:

- one validated `carla-replay-export.v1` JSON file per record,
- deterministic sanitized filenames derived from `scenario_id`,
- a small `manifest.json` with schema version `carla-replay-export-manifest.v1`.

The helper returns the manifest dictionary after writing it. Payloads are validated through the
existing `write_export_payload(...)` path.

## Boundary

This defines local file layout only. It does not provide a CLI, upload durable artifacts, import
CARLA, run replay, or claim simulator-transfer evidence.

## Validation

Proof command for the implementation branch:

```bash
uv run pytest tests/carla_bridge/test_t0_export.py -q
```

The RED run failed with a missing import for `write_export_records`. After implementation, the
focused CARLA-free test file passed with `15 passed`.
