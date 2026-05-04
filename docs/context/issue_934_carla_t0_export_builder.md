# Issue 934 CARLA T0 Export Builder API

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/934>

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/872>

Depends on issue #930 / PR #931.

## Goal

Add typed builder objects for `carla-replay-export.v1` so later CARLA bridge work can construct
neutral replay payloads without hand-assembling schema-shaped dictionaries.

## Public Surface

The builder API lives in `robot_sf_carla_bridge` and remains CARLA-free:

- `Pose2D`
- `CertificateRef`
- `ScenarioReplayRef`
- `RobotReplaySpec`
- `PedestrianReplaySpec`
- `SimulationSpec`
- `build_export_payload(...)`

`build_export_payload(...)` returns a dictionary and validates it through the same JSON schema used
by `validate_export_payload(...)` and `write_export_payload(...)`.

## Scope Boundary

This is still T0 neutral export work only. It does not parse scenario-loader YAML, map Robot-SF
coordinates into CARLA towns, import CARLA, or run replay. Optional pose heading is omitted from
the serialized payload when unset so schema validation can distinguish absent heading from an
invalid JSON null.

## Validation

Targeted TDD evidence:

```bash
uv run pytest tests/carla_bridge/test_t0_export.py -q
```

The RED run failed with `ImportError: cannot import name 'CertificateRef' from
'robot_sf_carla_bridge'`, proving the tests covered the missing public builder surface. The GREEN
run passed:

```text
8 passed in 13.13s
```

Before PR handoff, run the stacked readiness gate against the #930 branch:

```bash
git diff --check
BASE_REF=origin/930-carla-t0-export-schema scripts/dev/pr_ready_check.sh
```

## Follow-Up Boundary

The next #872 child can connect concrete certified Robot-SF scenario-loader output to these builder
objects. CARLA runtime replay remains a later T1 issue on a CARLA-capable machine.
