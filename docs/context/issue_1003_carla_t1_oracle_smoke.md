# Issue #1003 CARLA T1 Oracle Replay Smoke

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1003>

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/872>

Contract predecessor: [Issue #928 CARLA T0/T1 Oracle Replay Contract](issue_928_carla_t0_t1_replay_contract.md)

## Scope

Issue #1003 adds a narrow T1 oracle replay smoke setup boundary for one certified T0 export
manifest payload. The implementation is intentionally setup-only:

- it consumes one existing T0 export manifest/payload,
- validates the CARLA bridge schema catalog and selected T0 payload before replay setup,
- then requires the optional CARLA Python API before reporting an `oracle-replay` setup summary,
- and fails closed with `not-available` when CARLA is not importable.

This is not a CARLA benchmark result. Full metrics parity, multi-map replay, perception integration,
and long-running CARLA campaigns remain out of scope for this issue.

## Implementation

- `robot_sf_carla_bridge.replay_smoke.build_t1_oracle_replay_smoke_setup(...)` selects one payload
  from a T0 export manifest, validates the catalog/payload contract, and returns a JSON-safe
  setup-only summary when CARLA is available.
- `robot_sf_carla_bridge.replay_smoke.validate_t1_replay_catalog_payload(...)` validates the current
  schema catalog against `carla-bridge-schema-catalog.v1`, then requires the selected payload
  `schema_version` to match the catalog's `t0_export_payload` entry.
- `robot-sf-carla-t1-oracle-smoke --manifest <path> [--scenario-id <id>] [--json]` exposes the
  setup boundary as a project script.

## Validation

Focused unit and packaging coverage:

```bash
rtk uv run pytest tests/carla_bridge/test_t1_replay_smoke.py \
  tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q
```

Result: `5 passed in 12.10s`.

The tests cover:

- no-CARLA fail-closed CLI behavior,
- setup-only positive path with a fake importable `carla` module,
- schema-catalog payload version mismatch rejection,
- selecting a requested scenario from a multi-payload manifest,
- project-script registration.

Local CARLA availability:

```bash
rtk uv run robot-sf-check-carla --json
```

Result:

```json
{"available": false, "dependency": "carla", "reason": "CARLA Python API package 'carla' is not importable", "schema_version": "carla-availability.v1", "status": "not-available"}
```

Real T0 export and no-CARLA T1 smoke:

```bash
rtk uv run robot-sf-export-carla-t0 \
  --scenario-file configs/scenarios/single/planner_sanity_simple.yaml \
  --output-dir output/carla/issue_1003/t0 \
  --robot-sf-commit "$(rtk git rev-parse HEAD)"

rtk uv run robot-sf-validate-carla-t0-batch \
  --manifest output/carla/issue_1003/t0/manifest.json --json

rtk uv run robot-sf-carla-t1-oracle-smoke \
  --manifest output/carla/issue_1003/t0/manifest.json --json
```

The batch validation reported one payload:

```json
{"manifest": "output/carla/issue_1003/t0/manifest.json", "payload_count": 1, "scenario_ids": ["planner_sanity_simple"], "schema_version": "carla-replay-export-batch-validation-summary.v1"}
```

The T1 smoke command exited nonzero and failed closed because CARLA is not installed:

```json
{"action": "Install CARLA and ensure its Python API is on PYTHONPATH before using CARLA replay entry points.", "dependency": "carla", "manifest": "output/carla/issue_1003/t0/manifest.json", "mode": "not-available", "reason": "CARLA Python API package 'carla' is not importable. Install CARLA and ensure its Python API is on PYTHONPATH before using CARLA replay entry points.", "schema_version": "carla-t1-oracle-replay-smoke.v1", "status": "not-available"}
```

No live CARLA replay setup was run on this machine because `carla` is not importable. The fake-CARLA
unit test covers the positive setup-summary branch without making benchmark or runtime parity
claims.
