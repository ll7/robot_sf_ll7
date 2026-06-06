# Issue #2456 AMV Local Navigation Evaluation Suite

Date: 2026-06-07

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2456>

Status: proposal. Structural proposal only; no executable benchmark evidence.

## Purpose

This note documents the proposed AMV Local Navigation Evaluation Suite (suite ID
`amv_local_nav_suite_v0`) that defines structured evaluation dimensions, required metrics, and
evidence-boundary classifications for autonomous mobile vehicle (AMV) local navigation. The suite is
designed to surface planner limitations masked by holonomic or pedestrian-only benchmarks.

## Scope

The proposal defines:

- Eight evaluation dimensions with required metrics and existing/proposed status.
- Evidence-boundary classification: synthetic, proxy, or hardware-calibrated (blocked).
- Two scenario classes: sidewalk and crossing.
- Related issue links to #1559 (evidence boundary gate), #1585 (calibration source gate), #2000
  (real command-response trace acquisition), and #2001 (accepted proxy source).

The suite does **not** provide:

- Hardware-calibrated validity for any dimension.
- Real AMV data collection results.
- Simulator port integration or evidence.
- Benchmark-success claims.
- Paper-facing actuation-realism conclusions.

## Evaluation Dimensions

| Dimension | Evidence boundary | Metrics existing | Metrics proposed |
| --- | --- | --- | --- |
| Actuation feasibility | synthetic | 3 | 0 |
| Braking margin | proxy | 1 | 2 |
| Yaw-rate saturation | synthetic | 1 | 2 |
| Command smoothness | synthetic | 0 | 3 |
| Latency sensitivity | synthetic | 0 | 3 |
| Low-speed stability | proxy | 0 | 3 |
| Narrow corridor progress | synthetic | 0 | 3 |
| Sidewalk/crossing | proxy | 0 | 3 |

Total: 5 existing metrics, 19 proposed metrics.

## Evidence Boundary Classification

### Synthetic

Dimensions with purely software-defined stress factors and no external calibration link. Useful for
sensitivity analysis; do not support AMV hardware truth claims.

- `actuation_feasibility`, `yaw_rate_saturation`, `command_smoothness`, `latency_sensitivity`,
  `narrow_corridor_progress`

### Proxy

Dimensions that may reference platform-class proxy sources (e.g., e-scooter longitudinal data from
#2001) but have not been calibrated against real AMV hardware.

- `braking_margin`, `low_speed_stability`, `sidewalk_crossing`

### Hardware-calibrated (blocked)

No dimension in this suite has hardware-calibrated evidence. All dimensions are synthetic or proxy.
Promotion requires:

- Acceptance of a calibration source via #1585.
- Real command-response trace availability via #2000.
- Per-dimension field backing from the accepted source.

## Related Issue Links

- **#1559** (AMV evidence/claim boundary gate): linked as the gate for calibrated AMV actuation
  claims. Not resolved or bypassed by this proposal.
- **#1585** (AMV calibration source acceptance): linked as the pending calibration-source gate. No
  claim of acceptance or readiness.
- **#2000** (Real command-response trace acquisition): linked as the pending trace acquisition
  issue. No claim that real traces exist or are available.
- **#2001** (AMV actuation proxy source analysis): linked as the accepted platform-class proxy
  source for longitudinal acceleration and deceleration only. Proxy coverage does not extend to
  yaw rate, angular acceleration, latency, update rate, or command-response dynamics.

## Validation

This note is documentation and structural guidance only. It creates no new benchmark evidence and
does not modify runtime metric computation or runnable benchmark campaigns. Validate updates with
the docs proof consistency checker and path/link checks.

The canonical suite manifest is at `configs/benchmarks/issue_2456_amv_local_nav_suite_v0.yaml`.
Test coverage for the suite manifest contract is at
`tests/benchmark/test_issue_2456_amv_local_nav_suite.py`.
