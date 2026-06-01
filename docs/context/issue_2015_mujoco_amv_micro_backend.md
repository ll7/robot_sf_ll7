# Issue #2015 MuJoCo AMV Micro-Backend Diagnostic (2026-06-01)

Related issue: [#2015](https://github.com/ll7/robot_sf_ll7/issues/2015)

## Decision

Robot SF can support a narrow MuJoCo micro-backend diagnostic path for AMV actuation response
experiments, but only as an optional local probe. The implemented helper does not add MuJoCo to
routine project dependencies and does not replace the Robot SF simulator.

The supported first slice is:

```text
simulation_trace_export.v1 selected-action trace, duration-coded CSV, or built-in demo trace
  -> optional MuJoCo runtime/model construction proof
  -> deterministic AMV kinematic response with acceleration, deceleration, yaw-rate, angular-rate,
     and latency limits
  -> diagnostic JSON and Markdown summary
```

## Claim Boundary

This is diagnostic-only evidence. It is not:

- social-navigation benchmark evidence;
- calibrated AMV hardware evidence;
- a Robot SF simulator backend replacement;
- a MuJoCo contact or map-geometry parity proof;
- controller latency calibration.

Calibrated or paper-facing AMV claims still require the source-provenance gates in
[#1585](https://github.com/ll7/robot_sf_ll7/issues/1585) and
[#1559](https://github.com/ll7/robot_sf_ll7/issues/1559).

## Implemented Entry Point

```bash
uv run python scripts/tools/mujoco_amv_micro_backend.py \
  --trace tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json \
  --output-json output/diagnostics/issue_2015_mujoco_trace_probe.json \
  --output-md output/diagnostics/issue_2015_mujoco_trace_probe.md
```

The trace mode accepts only the existing `simulation_trace_export.v1` schema and reads
`frames[].planner.selected_action.linear_velocity` plus
`frames[].planner.selected_action.angular_velocity`. Invalid trace payloads fail closed through the
existing schema loader.

For a self-contained local smoke, use:

```bash
uv run python scripts/tools/mujoco_amv_micro_backend.py \
  --demo-fixture \
  --output-json output/diagnostics/issue_2015_mujoco_demo.json \
  --output-md output/diagnostics/issue_2015_mujoco_demo.md
```

The helper fails closed with exit code `2` if MuJoCo cannot be imported. That is intentional:
MuJoCo is an optional diagnostic runtime, not a normal install requirement.

For custom duration-coded inputs, pass a CSV with:

```text
duration_s,v_m_s,omega_rad_s
```

The output schema is `mujoco_amv_micro_backend.v1`. The JSON payload records:

- MuJoCo version and minimal model XML checksum;
- routine dependency status (`false`);
- optional source `simulation_trace_export.v1` metadata;
- limit and latency configuration;
- unicycle `v, omega` command contract;
- unsupported semantics;
- per-step commanded/applied `v` and `omega`;
- derived linear acceleration and pose;
- clip flags and compact summary fields.

## Unsupported Semantics

The first slice intentionally does not model:

- pedestrian dynamics;
- social-navigation benchmark outcomes;
- map geometry and obstacle contacts;
- hardware-calibrated AMV response;
- controller latency calibration.

## Validation

Targeted validation for this PR:

```bash
rtk scripts/dev/run_worktree_shared_venv.sh -- pytest tests/tools/test_mujoco_amv_micro_backend.py -q
rtk scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/mujoco_amv_micro_backend.py \
  --trace tests/fixtures/analysis_workbench/simulation_trace_export_v1/minimal_trace.json \
  --output-json output/diagnostics/issue_2015_mujoco_trace_probe.json \
  --output-md output/diagnostics/issue_2015_mujoco_trace_probe.md
rtk git diff --check
```

The demo output belongs under ignored `output/` and should not be treated as a durable dependency.
Promote only compact evidence if a future calibrated or benchmark-facing follow-up explicitly needs
it.
