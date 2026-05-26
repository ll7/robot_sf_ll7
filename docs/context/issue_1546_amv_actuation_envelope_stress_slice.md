# Issue #1546 AMV Actuation-Envelope Stress Slice Design

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1546>

Related context:

- [`docs/context/issue_1344_paired_amv_protocol_report.md`](issue_1344_paired_amv_protocol_report.md)
- [`docs/context/issue_1353_broader_amv_preflight.md`](issue_1353_broader_amv_preflight.md)
- [`docs/context/evidence/issue_1344_paired_amv_primary_2026-05-20/README.md`](evidence/issue_1344_paired_amv_primary_2026-05-20/README.md)
- [`docs/context/evidence/issue_1353_broader_amv_2026-05-26/README.md`](evidence/issue_1353_broader_amv_2026-05-26/README.md)
- [`docs/context/issue_691_benchmark_fallback_policy.md`](issue_691_benchmark_fallback_policy.md)
- [`docs/benchmark_spec.md`](../benchmark_spec.md)
- [`docs/multi_amv_benchmark.md`](../multi_amv_benchmark.md)

## Goal

Define a conservative, benchmark-safe design for a **single-AMV constrained-execution stress slice**
that can later test whether planner behavior degrades when the robot is forced to obey a tighter
actuation envelope. This note is design-only: it does not implement constraint injection, does not
run campaigns, and does not claim that any proposed values match real hardware.

## Claim Boundary

- This note defines a **synthetic stress profile**, not a real AMV hardware specification.
- The proposed defaults below are **starting values that require hardware or controller-trace
  calibration** before any real-world, safety, or transfer interpretation.
- Existing `#1344` and `#1353` artifacts are **benchmark protocol evidence**, not actuation-envelope
  calibration evidence.
- This note does **not** change benchmark pass/fail semantics, planner eligibility, seed policy, or
  the fail-closed fallback policy.
- `fallback`, `degraded`, `failed`, `partial-failure`, and `not_available` rows remain
  **non-success evidence** under
  [`docs/context/issue_691_benchmark_fallback_policy.md`](issue_691_benchmark_fallback_policy.md).

## Evidence Basis

1. `#1344` proved the paired nominal/stress AMV protocol for core rows, but both campaigns reported
   `amv_coverage_status=warn` and the tracked AMV coverage summaries showed `Observed = -` for all
   required AMV dimensions.
2. `#1353` broadened planner coverage, but the 2026-05-26 evidence still treated AMV coverage as a
   warning surface, not a calibrated actuation contract. It also kept caveated rows such as
   `socnav_bench not_available` out of success evidence.
3. The repository already has relevant command-limit terminology in
   `robot_sf/planner/hybrid_rule_local_planner.py` and trajectory metrics in
   `robot_sf/benchmark/metrics.py`, including `velocity_max`, `acceleration_max`, `jerk_mean`,
   `curvature_mean`, `energy`, `min_clearance`, `near_misses`, and `time_to_collision_min`.
4. The current tracked AMV profile in `configs/benchmarks/issue_1344_paired_stress_primary.yaml` and
   `configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml` covers
   `use_case/context/speed_regime/maneuver_type`, but **not** controller latency, update rate, or
   actuation saturation.

## Proposed Envelope Definition

Treat the AMV actuation envelope as the tuple:

`{max_linear_accel, max_linear_decel, max_yaw_rate, max_angular_accel, jerk/smoothness profile, effective latency, effective update rate}`

For the first follow-up slice, keep the envelope **single-robot, differential-drive, paper_facing:
false**, and explicitly synthetic.

| Parameter | Synthetic starting default | Repository anchor | Caveat |
| --- | --- | --- | --- |
| Max linear acceleration | `2.0 m/s^2` | `HybridRuleLocalPlannerConfig.max_linear_accel` | Starting stress default only; not a hardware claim. |
| Max linear deceleration / braking | `2.5 m/s^2` | `HybridRuleLocalPlannerConfig.max_linear_decel` | Treat as absolute braking cap; requires calibration before interpretation. |
| Max yaw rate | `1.2 rad/s` | `HybridRuleLocalPlannerConfig.max_angular_speed` | Use as a synthetic steering-turn-rate ceiling only. |
| Max angular acceleration | `4.0 rad/s^2` | `HybridRuleLocalPlannerConfig.max_angular_accel` | Synthetic controller-response ceiling; not validated against hardware. |
| Jerk / smoothness | No hard hardware cap in v0; track `jerk_mean`, `jerk_max`, `curvature_mean`, and `energy` | `robot_sf/benchmark/metrics.py` | Start as an analysis surface, not an exclusion gate. |
| Effective latency | Step-delay categories relative to benchmark `dt=0.1`: `0-step`, `1-step`, `2-step` | Existing benchmark `dt` in `#1344/#1353` configs | Delay categories are software stress injections, not measured controller latency. |
| Effective update rate | `10 Hz` matched (`dt=0.1`), `5 Hz` reduced (`2*dt`) | Benchmark `dt=0.1` | Use only as a synthetic sensitivity ladder for a future diagnostic slice. |

### Recommended Interpretation

- The **first implemented slice** should expose these values as a versioned synthetic profile such as
  `amv-actuation-stress-v0`.
- The profile should be reported as **constraint provenance**, not as a benchmark-quality AMV
  certification.
- Any later hardware-aligned profile should use a different profile name and cite the calibration
  source explicitly.

## Metrics

### Metrics available now

These metrics already exist and are sufficient for a conservative first slice:

| Metric | Why it belongs in the slice | Current source |
| --- | --- | --- |
| `success`, `timeout`, `time_to_goal_norm` | Detect whether tighter constraints break task completion. | `robot_sf/benchmark/metrics.py` |
| `total_collision_count`, `near_misses`, `min_clearance`, `time_to_collision_min` | Safety degradation under constraint pressure. | `robot_sf/benchmark/metrics.py` |
| `velocity_max` | Detect speed clipping or inability to accelerate to nominal cruise behavior. | `robot_sf/benchmark/metrics.py` |
| `acceleration_max` | Peak linear acceleration demand under the constrained profile. | `robot_sf/benchmark/metrics.py` |
| `jerk_mean`, `jerk_max` | Smoothness proxy for stop-go or clipped command behavior. | `robot_sf/benchmark/metrics.py` |
| `curvature_mean` | Turning-demand proxy when yaw-rate limits bind. | `robot_sf/benchmark/metrics.py` |
| `energy`, `stalled_time`, `failure_to_progress` | Distinguish slow-but-safe behavior from genuine envelope-induced deadlock. | `robot_sf/benchmark/metrics.py` |

### Metrics still missing for a stronger actuation claim

These should be follow-up-derived metrics, not implied by current evidence:

- **signed braking peak**: current metrics expose acceleration magnitude, not signed longitudinal
  braking demand;
- **yaw-rate saturation fraction**: requires commanded or reconstructed yaw-rate traces;
- **command clip fraction**: percentage of steps where the requested command was clipped by the
  synthetic envelope;
- **delay provenance**: explicit record of applied step-delay/update-hold mode in episode outputs.

For `#1546`, those are design requirements for a future implementation issue, not blockers for this
note.

## Compact Scenario / Seed Candidate List

Use a small paired slice anchored to existing scenario names and the canonical named seed sets.
Prefer the benchmark `eval` seed set (`[111, 112, 113]`) for comparability with `#1344` and `#1353`;
use scenario-native seeds only for local debugging.

| Scenario candidate | Surface role | Why it stresses the envelope | Seed recommendation |
| --- | --- | --- | --- |
| `classic_overtaking_medium` | primary longitudinal-control row | Passing pressure can reveal acceleration and braking saturation without needing extreme density. | `eval` |
| `classic_bottleneck_high` | primary stop-go row | Narrow choke points amplify braking/restart demand and stall risk. | `eval` |
| `classic_cross_trap_high` | primary turning row | Local-minimum circulation can expose yaw-rate and angular-acceleration limits. | `eval` |
| `francis2023_blind_corner` | occlusion/turn-entry row | Useful for delay-sensitive turn initiation and clearance loss. | `eval` |
| `francis2023_intersection_wait` | yield/relaunch row | Good for latency and jerk sensitivity after a forced wait. | `eval` |

Optional nominal anchor rows for comparison only:

- `single_ped_crossing_orthogonal`
- `classic_doorway_low`

The slice should stay compact until the implementation can report explicit command-envelope
provenance. This is a stress diagnostic, not a new broad benchmark matrix.

## Validation / Preflight Manifest Sketch

The first implementation follow-up should keep the same fail-closed benchmark discipline used by the
camera-ready workflow, but add explicit synthetic-envelope provenance. The YAML below is a
**proposal-only manifest sketch**: it includes fields and artifact names that do not exist in the
current benchmark schema yet. Treat it as an implementation target, not as a runnable config or
current report contract.

```yaml
name: issue_1546_amv_actuation_stress_slice_v0
paper_facing: false
base_protocol: issue_1344_paired_amv
scenario_candidates:
  - classic_overtaking_medium
  - classic_bottleneck_high
  - classic_cross_trap_high
  - francis2023_blind_corner
  - francis2023_intersection_wait
seed_policy:
  mode: seed-set
  seed_set: eval
kinematics_matrix:
  - differential_drive
synthetic_actuation_profile:
  name: amv-actuation-stress-v0
  max_linear_accel_m_s2: 2.0
  max_linear_decel_m_s2: 2.5
  max_yaw_rate_rad_s: 1.2
  max_angular_accel_rad_s2: 4.0
  latency_mode: one-step-delay
  update_mode: 5hz-hold
required_reports:
  - reports/campaign_summary.json
  - reports/campaign_table.md
  - reports/amv_coverage_summary.md
  - reports/actuation_envelope_summary.json
  - reports/actuation_envelope_summary.md
preflight_checks:
  - config validation succeeds
  - scenario names resolve
  - synthetic profile is recorded in manifest and episode metadata
  - unsupported/fallback/degraded rows fail closed
```

The new `actuation_envelope_summary.*` artifact should be a **diagnostic supplement** that reports
constraint provenance plus the core available-now metrics above. It should not redefine benchmark
success.

### Minimum V0 Acceptance Bar

The first implementation issue can close without hardware-calibrated saturation metrics only if it:

1. records the synthetic profile name and parameter values in the campaign manifest and episode
   metadata;
2. proves the synthetic envelope is actually applied, or fails closed when it cannot be applied;
3. reports the existing completion, safety, smoothness, and progress metrics listed above;
4. emits explicit placeholders or `not_available` fields for unimplemented derived saturation
   metrics so downstream readers cannot mistake missing clip/yaw/braking metrics for zeros.

If the implementation can cheaply derive `command_clip_fraction`, `yaw_rate_saturation_fraction`,
or signed braking peaks from recorded commands, it should include them. Those derived metrics are
recommended for stronger follow-up evidence, but the minimum v0 contract is provenance plus
fail-closed application of the synthetic profile.

## Recommendation For Follow-Up

**Recommendation: yes, open a follow-up implementation issue for a constrained-execution benchmark
slice, but keep it narrowly diagnostic.**

Recommended scope of that follow-up:

1. Add a versioned synthetic actuation profile with explicit provenance in manifests and episode
   records.
2. Reuse a compact `paper_facing: false` scenario subset and the named `eval` seed set.
3. Report existing safety/smoothness metrics and add derived saturation metrics when command traces
   make them auditable; otherwise record them explicitly as `not_available`.
4. Fail closed when the profile cannot be applied or when execution falls back/degrades.

Not recommended in the first follow-up:

- broad all-planners campaigns,
- paper-facing claims,
- pass/fail gate changes,
- real-AMV or deployment-language interpretation,
- multi-AMV coordination claims from [`docs/multi_amv_benchmark.md`](../multi_amv_benchmark.md).

## Out Of Scope

- Implementing a vehicle or controller model;
- running new campaigns or publishing result tables;
- revising the existing AMV coverage contract dimensions;
- claiming that the synthetic envelope matches a delivery robot or shared-space micromobility
  platform;
- promoting fallback, degraded, unavailable, or skipped rows as success evidence.
