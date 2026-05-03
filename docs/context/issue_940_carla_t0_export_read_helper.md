# Issue #940 CARLA T0 Export Read Helper

## Goal

Issue #940 adds a CARLA-free read boundary for `carla-replay-export.v1` JSON payloads. It is a
small child of issue #872 and is stacked on the issue #930 schema package plus the issue #934
builder API.

## Decision

`robot_sf_carla_bridge.read_export_payload(path)` loads UTF-8 JSON from disk, validates the parsed
payload through the existing `carla-replay-export.v1` schema, and returns the validated dictionary.
This keeps exported replay files from bypassing the same contract used by
`write_export_payload(...)` and `build_export_payload(...)`.

The helper intentionally remains T0-only:

- it does not import CARLA,
- it does not translate Robot-SF coordinates into CARLA world coordinates,
- it does not run replay or metric parity checks,
- and a successful read is only schema evidence, not simulator-transfer evidence.

## Validation

Proof commands for the implementation branch:

```bash
uv run pytest tests/carla_bridge/test_t0_export.py -q
```

The TDD RED run failed with two missing-import failures for `read_export_payload`. After the helper
and public export were added, the focused CARLA-free test file passed with `9 passed`.

## Follow-Up Boundary

The next useful #872 slice is still scenario conversion: concrete certified Robot-SF scenario output
needs to be mapped into the builder/export schema. CARLA runtime replay remains T1 and must fail
closed as `not-available` when the simulator or Python API is absent.
