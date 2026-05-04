# Issue 930 CARLA T0 Neutral Export Schema

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/930>

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/872>

Predecessor contract issue: <https://github.com/ll7/robot_sf_ll7/issues/928>

## Goal

Add the first import-safe `robot_sf_carla_bridge` code surface without making CARLA a normal
Robot-SF dependency. This is the T0 neutral export step only: it defines a versioned JSON payload
shape and explicit CARLA availability status for later oracle replay work.

## Public Surfaces

- Package: `robot_sf_carla_bridge`
- Schema version: `carla-replay-export.v1`
- Schema file: `robot_sf_carla_bridge/schemas/carla_replay_export.v1.json`
- Export helpers:
  - `load_export_schema()`
  - `validate_export_payload(payload)`
  - `write_export_payload(payload, output_path)`
- Availability helper:
  - `check_carla_availability()`

## Contract Boundary

The package must import without CARLA installed. CARLA runtime availability is reported by
`check_carla_availability()` as an explicit status object, with missing CARLA represented as
`not-available`.

The T0 payload schema records:

- scenario identity, source config, map id, and `scenario_cert.v1` certificate metadata,
- robot start/goal/footprint/kinematics,
- pedestrian starts, routes, and timing metadata,
- static geometry references,
- simulation horizon and termination conditions,
- trajectory metric field names,
- provenance for the Robot-SF commit, exporter, and certificate generator.

Exported JSON is not CARLA parity evidence. A later T1 replay issue must run CARLA with oracle
state and compare trajectory metrics before making simulator-transfer claims.

## Validation

Targeted TDD evidence:

```bash
uv run pytest tests/carla_bridge/test_t0_export.py -q
```

The RED run failed with `ModuleNotFoundError: No module named 'robot_sf_carla_bridge'`, proving the
tests covered new package behavior. The GREEN run passed:

```text
5 passed in 21.07s
```

Before PR handoff, also run:

```bash
git diff --check
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Follow-Up Boundary

Next #872 implementation children should connect concrete certified Robot-SF scenario-loader output
to this schema, then add a CARLA runtime smoke command on a CARLA-capable machine. Coordinate
mapping, CARLA town selection, perception, ROS integration, and training remain out of scope here.
