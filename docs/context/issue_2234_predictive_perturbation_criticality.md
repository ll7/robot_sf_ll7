# Issue #2234 Predictive Perturbation Criticality Protocol

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2234>

Status: proposal and analysis design from existing Issue #1610 perturbation evidence.

## Research Question

Can perturbation criticality computed from a small pilot predict which scenario features or seeds
will later produce planner failures, near misses, low-progress timeouts, or unsafe clearance on
held-out seeds or scenario variants?

This note defines the smallest conservative test design. It does not run a new perturbation
campaign and does not claim predictive validity.

## Source Evidence

The canonical synthesis anchor is
[issue_1965_perturbation_criticality_synthesis.md](issue_1965_perturbation_criticality_synthesis.md).
It classifies Issue #1610 children as diagnostic local evidence and defines the
`criticality_summary.v1` schema. The high-signal and low-signal families come from:

| Source | Perturbation family | Current observation | Boundary |
| --- | --- | --- | --- |
| [issue_1904_scenario_perturbation_criticality_pilot.md](issue_1904_scenario_perturbation_criticality_pilot.md) | `robot_route_offset` | First paired pilot: terminal deltas `0.0`; min-distance `+0.0037 m`. | One-seed diagnostic smoke. |
| [issue_1933_perturbation_seed_coverage.md](issue_1933_perturbation_seed_coverage.md) | `robot_route_offset` | Four-seed expansion: terminal deltas `0.0`; min-distance `+0.0007 m`. | Low-criticality diagnostic slice. |
| [issue_1937_ped_route_offset.md](issue_1937_ped_route_offset.md) | `pedestrian_route_offset` | Clearance changed locally, especially corridor rows; terminal metrics neutral. | Route-only scenario support. |
| [issue_1941_ped_timing_phase.md](issue_1941_ped_timing_phase.md) | `single_pedestrian_start_delay_offset` | `francis2023_intersection_wait` min-distance `+4.159358 m`; terminal metrics neutral. | Explicit single-pedestrian timing only. |
| [issue_1943_ped_speed_perturbation.md](issue_1943_ped_speed_perturbation.md) | `single_pedestrian_speed_offset` | Overall min-distance `-0.588347 m`; `intersection_wait` `-2.002917 m`; one ORCA seed-local outcome flip. | Outcome flip is not replicated. |
| [issue_1949_ped_wait_duration_perturbation.md](issue_1949_ped_wait_duration_perturbation.md) | `single_pedestrian_wait_duration_offset` | Eligible rows flat on tested magnitudes. | One wait-bearing scenario. |
| [issue_1951_intersection_wait_phase_grid.md](issue_1951_intersection_wait_phase_grid.md) | phase grid | Speed signs are monotonic by magnitude; positive start delay is large and flat; wait duration remains flat. | One scenario, one pedestrian, three seeds. |
| [issue_1953_intersection_wait_speed_grid_trace.md](issue_1953_intersection_wait_speed_grid_trace.md) | speed-grid trace | `+0.5 m/s` speed row gives mean closest-clearance delta `-3.862581 m` with same nearest pedestrian index. | Trace mechanism, not outcome prediction. |

## Predictive Hypothesis

Pre-register this hypothesis for the next executable issue:

> On eligible completed-pair rows, perturbation family, magnitude/sign, phase variables, target
> entity, baseline clearance state, planner, scenario family, and seed bucket predict the sign or
> threshold crossing of held-out failure, near-miss/clearance, or low-progress outcomes better than
> a family-agnostic baseline.

This is intentionally about prediction, not causality. A successful held-out result would say that
the chosen criticality features are useful warning signals for scenario or planner selection. It
would not prove that perturbing the feature causes future failures.

## Predictor Contract

Use only fields that existing notes show are observable or can be recorded without inventing new
semantics:

| Predictor | Examples | Required handling |
| --- | --- | --- |
| `perturbation_family` | route offset, start delay, speed offset, wait duration, waypoint, density | Must match the `criticality_summary.v1` family string. |
| `magnitude` and `sign` | `+0.5 s`, `+0.5 m/s`, `-0.25 m/s` | Keep units explicit and do not compare magnitudes across unit families directly. |
| `phase_context` | `wait_at`, phase-grid bucket, closest-approach timing | If absent, record `not_available`; do not infer from unsupported route-only rows. |
| `target_entity` | robot route, single pedestrian `h1`, route pedestrian | Route-only and single-pedestrian targets are separate domains. |
| `scenario_family` and `scenario_id` | `intersection_wait`, `join_group`, `head_on_corridor` | Use family as a grouping variable and scenario ID for leakage checks. |
| `planner_key` and mode | `goal`, `orca`, hybrid guard | Keep native/adapter/fallback/degraded status visible. |
| `baseline_state` | baseline status, starting clearance, no-op timeout/collision | Use as a covariate so the model does not mistake already-hard rows for perturbation signal. |
| `seed_bucket` | discovery seed, held-out seed | Freeze before analysis and do not tune thresholds on held-out seeds. |

Unavailable covariates should be listed explicitly in the artifact, not backfilled. Current likely
gaps include stable pedestrian IDs in trace slices, route-pedestrian speed fields, global density
semantics, calibrated AMV actuation state, and full failure-mechanism labels per episode.

## Target Variables

The first held-out test should define binary and directional targets from already reported deltas:

| Target | Primary threshold | Interpretation |
| --- | --- | --- |
| `failure_or_collision_delta` | collision delta `> 0` or success delta `< 0` | Hard outcome warning. |
| `low_progress_or_timeout_delta` | timeout delta `> 0` or progress delta below a predeclared threshold | Progress warning when terminal status is unchanged. |
| `clearance_risk_delta` | min-distance delta `<= -0.5 m` | Intrusive-clearance warning; threshold is intentionally larger than route-offset noise. |
| `near_miss_delta` | near-miss delta `> 0` where available | Secondary safety warning. |
| `trace_phase_shift` | closest-approach time or progress delta exceeds predeclared threshold | Mechanism explanation, not the main prediction target. |

The `-0.5 m` clearance threshold separates the large local speed/phase responses from the tiny
route-offset deltas observed in Issue #1904 and Issue #1933. A future issue may tighten this
threshold, but it should do so before inspecting held-out results.

## Held-Out Test Design

Minimum viable design:

1. Use the `francis2023_intersection_wait` speed/start-delay family as the discovery slice because
   it has the clearest signed local response and trace support.
2. Freeze discovery seeds from the existing evidence and choose at least two held-out seeds that
   were not used to choose thresholds.
3. Keep planners fixed to `goal`, `orca`, and `scenario_adaptive_hybrid_orca_v2_collision_guard`
   unless the issue explicitly changes one axis.
4. Use the same horizon, dt, paired-baseline rule, and fail-closed status accounting as the source
   Issue #1610 pilots.
5. Exclude rows with `invalid`, `fallback`, `degraded`, `missing`, or `failed` status from predictor
   utility claims, while reporting their counts.
6. Report at least one family-agnostic baseline, such as "all speed rows are critical" or "all
   perturbations are non-critical", so the predictor is compared against a simple rule.
7. Promote only a compact tracked summary and checksum/provenance files; leave raw episode JSONL
   under ignored `output/` unless a small fixture is needed.

Optional second axis after the first held-out result:

- add `pedestrian_route_offset` for a route-based contrast, or
- add `single_pedestrian_wait_duration_offset` as a low-sensitivity negative control.

Do not add both a new perturbation family and a new planner set in the same first predictive test.

## Decision Rule

Classify the next result as:

| Result class | Rule |
| --- | --- |
| `predictive_signal_supported` | At least two held-out seeds show the predeclared high-criticality feature crossing the target threshold with no fallback/degraded support rows, and the simple baseline is worse on the same target. |
| `negative_control_supported` | A predeclared low-sensitivity family stays below the target threshold on all completed held-out rows. |
| `inconclusive` | Mixed signs, underpowered rows, missing covariates, or threshold crossings only in invalid/fallback/degraded rows. |
| `not_predictive` | The discovery high-criticality feature fails to predict the held-out target while the simple baseline performs as well or better. |

Even `predictive_signal_supported` remains diagnostic unless it is replicated across another
scenario family or perturbation family under the same protocol.

## Recommendation

Proceed to a small held-out validation issue only after the issue contract names:

- the frozen discovery rows;
- the held-out seed IDs;
- the single target variable and threshold;
- the simple baseline rule;
- the exact status filter;
- the compact artifact schema.

Until that issue runs, the current perturbation-criticality lane should be described as a
plausible predictive hypothesis with strong local phase/speed diagnostics, not as predictive
scenario-quality evidence.

## Validation

This docs-only protocol was checked with:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
