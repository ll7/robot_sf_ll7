# Issue #948 CARLA T0 Scenario File Export

## Goal

Issue #948 adds a CARLA-free batch helper for exporting every entry in a scenario manifest to
`carla-replay-export.v1` neutral payload records. It is a child of issue #872 and is stacked on the
issue #946 scenario-entry export helper.

## Decision

`build_export_payloads_from_scenario_file(...)` loads scenario entries through the existing
scenario loader, preserves manifest order, and returns records containing:

- `scenario_id`: the scenario-loader identifier,
- `payload`: the validated neutral export payload from `build_export_payload_from_scenario_entry`.

The helper intentionally does not write files. A later CLI/export command can decide output layout,
file naming, and artifact persistence.

## Boundary

This is still T0 neutral export construction only. It does not import CARLA, run replay, perform
CARLA-world coordinate mapping, or claim transfer evidence.

## Validation

Proof command for the implementation branch:

```bash
uv run pytest tests/carla_bridge/test_t0_export.py -q
```

The RED run failed with a missing import for `build_export_payloads_from_scenario_file`. After
implementation, the focused CARLA-free test file passed with `14 passed`.
