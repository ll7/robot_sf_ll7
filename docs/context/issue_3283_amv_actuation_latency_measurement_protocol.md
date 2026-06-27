# Issue #3283 — AMV Actuation Latency & Rider-Coupling Measurement Protocol

**Status:** `blocked-external-input` (measurement plan only; no measured-value claim)
**Evidence tier:** `blocked` — proposal / intake contract, not benchmark evidence
**Issue:** [#3283](https://github.com/ll7/robot_sf_ll7/issues/3283)
**Related lanes:** #1559, #1585, #1586, #2000, #2415

Autonomous micromobility vehicle (AMV) = the small autonomous vehicle whose actuation
envelope Robot SF models. "Actuation latency" is the delay between a commanded action and
the vehicle's mechanical response; "rider-coupling response" is how a rider or carried load
changes that response.

## Purpose

The research report flags **actuation latency** and **rider-coupling response** as missing
pieces of AMV command-response evidence. Collecting that data is **externally blocked**. This
note defines the measurement/intake **protocol** and a machine-checkable **manifest contract**
so the moment real data is staged and reviewed it can be wired in without re-deriving the
schema. It makes **no measured-value claim** and does not collect or fabricate data.

The contract is enforced by
`robot_sf/benchmark/actuation_latency_measurement_manifest.py`
(`check_amv_actuation_latency_measurement_manifest`), with the schema at
`robot_sf/benchmark/schemas/amv_actuation_latency_measurement_manifest.v1.json` and a worked
example at
`configs/benchmarks/issue_3283_amv_actuation_latency_measurement_manifest_example.yaml`.

## Measurement protocol

### Quantities (command → response chain)

| Quantity | Meaning | Example sensor | Canonical units |
| --- | --- | --- | --- |
| `command_issuance` | Timestamp the actuation command is published | command-bus logger | `timestamp_s` |
| `mechanical_response` | Onset of wheel/drivetrain response | wheel encoder | `rad/s` |
| `yaw_response` | Yaw-rate response to a steering/differential command | IMU gyro (z) | `rad/s` |
| `braking_latency` | Delay to braking effort onset | brake-pressure transducer | `bar` |
| `acceleration_latency` | Delay to longitudinal acceleration onset | IMU accel (x) | `m/s^2` |
| `rider_load_condition` | Rider/load mass condition under test | seat load cell | `kg` |
| `rider_response` | Rider input coupling into the vehicle | handlebar force/torque | `N` |

### Sampling rate

Each channel declares its own `sampling_rate_hz`. Command and high-rate response channels
(command/mechanical/yaw/braking/acceleration) should sample fast enough to resolve the latency
of interest — the example uses 200 Hz (5 ms resolution). Rider channels may sample slower.

### Synchronization

All channels must share a common time base so cross-channel latency is meaningful. The manifest
requires `synchronization.{method, reference_clock, max_skew_ms}`; `max_skew_ms` bounds the
residual alignment error (the example: a shared hardware trigger with PTP, ≤ 2 ms skew).

### Provenance

A `measured` manifest must carry `source_id`, `source_uri`, `source_type`, `measurement_date`,
and per-field `units` before any value is trusted — mirroring the calibrated-actuation
provenance gate in `robot_sf/benchmark/synthetic_actuation.py`. Staging flows through
`.agents/skills/data-staging-provenance/SKILL.md`.

## Proposed AMV-profile manifest fields

These extend the synthetic actuation-envelope schema. A future **measured** profile would add:

**Latency**
- `command_to_motion_latency_s` (`s`)
- `command_to_yaw_latency_s` (`s`)
- `braking_onset_latency_s` (`s`)
- `acceleration_onset_latency_s` (`s`)

**Rider-coupling**
- `rider_load_kg` (`kg`)
- `rider_coupling_gain` (`dimensionless`)
- `rider_response_latency_s` (`s`)

## Synthetic-vs-measured separation

Each proposed field carries a `value_status`, gated by the manifest's `measurement_status`:

| `measurement_status` | required `value_status` | contract status | measured-value claim |
| --- | --- | --- | --- |
| `blocked-external-input` | `pending` | `blocked` | not allowed |
| `synthetic-only` | `synthetic-placeholder` | `synthetic-only` | not allowed |
| `measured` | `measured` (+ full provenance) | `ready` | allowed |

A `synthetic-only` manifest that declares a measured provenance source is rejected as
boundary conflation. Synthetic actuation placeholders
(`robot_sf.benchmark.synthetic_actuation`) stay diagnostic-only and are never promoted into
these fields without accepted provenance.

## Validation

```bash
uv run python -m pytest tests/benchmark -k actuation -q
uv run python scripts/tools/check_amv_actuation_latency_measurement_manifest.py
```

## Blocked-until / next step

Real AMV command-response data must be collected or accepted from a reliable source and staged
+ reviewed. Only then does a manifest move to `measurement_status: measured`, provenance is
populated, and the proposed fields can carry measured values. Until then this lane is a
contract, not evidence — and other AMV actuation calibration work (e.g. #1585, #1586) is not
blocked by it.
