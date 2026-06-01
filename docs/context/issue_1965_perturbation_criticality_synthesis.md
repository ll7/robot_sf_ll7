# Issue #1965 Perturbation Criticality Synthesis

Issue: [#1965](https://github.com/ll7/robot_sf_ll7/issues/1965)
Parent: [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610)
Status: synthesis checkpoint

## Goal

Synthesize the #1610 perturbation-criticality children into a reusable method before adding more
perturbation families. The evidence below is diagnostic local evidence only. It is not
benchmark-strength, robustness, or paper-facing evidence.

## Evidence Inventory

| Issue / PR | Perturbation Family | Boundary | Evidence Status | Main Observation | Limitation |
|---|---|---|---|---|---|
| [#1904](https://github.com/ll7/robot_sf_ll7/issues/1904) / [#1907](https://github.com/ll7/robot_sf_ll7/pull/1907) | `robot_route_offset` | `goal`, `orca`, seed-limit 1, horizon 80 | diagnostic valid | 6/6 paired rows completed; success/collision/timeout deltas `0.0`; min-distance `+0.003691 m` | one-seed smoke only |
| [#1933](https://github.com/ll7/robot_sf_ll7/issues/1933) / [#1934](https://github.com/ll7/robot_sf_ll7/pull/1934) | `robot_route_offset` | `goal`, `orca`, seed-limit 4 | diagnostic valid | 24/24 paired rows completed; success/collision/timeout deltas `0.0`; min-distance `+0.000667 m` | low-clearance effect, no terminal outcome signal |
| [#1935](https://github.com/ll7/robot_sf_ll7/issues/1935) / [#1936](https://github.com/ll7/robot_sf_ll7/pull/1936) | `robot_route_offset` plus stronger local planner | adds `scenario_adaptive_hybrid_orca_v2_collision_guard` | diagnostic valid | 36/36 paired rows completed; min-distance `+0.0116 m`, strongest local candidate `+0.0335 m` | still no success/collision/timeout delta |
| [#1937](https://github.com/ll7/robot_sf_ll7/issues/1937) / [#1938](https://github.com/ll7/robot_sf_ll7/pull/1938) | `pedestrian_route_offset` | three-planner, seed-limit 4 | diagnostic valid with one fail-closed exclusion | 60/60 pairs completed; `pedestrian_route_offset` min-distance `+0.0978 m`; terminal metrics neutral | only route-based classic scenarios support `ped_routes` |
| [#1939](https://github.com/ll7/robot_sf_ll7/issues/1939) / [#1940](https://github.com/ll7/robot_sf_ll7/pull/1940) | corridor trace response | `classic_head_on_corridor_low` | diagnostic trace valid | 12/12 trace pairs completed; mean clearance `+0.153489 m`; hybrid progress effect mostly seed 117 | trace-level mechanism, not broad criticality |
| [#1941](https://github.com/ll7/robot_sf_ll7/issues/1941) / [#1942](https://github.com/ll7/robot_sf_ll7/pull/1942) | `single_pedestrian_start_delay_offset` | three-planner, seed-limit 4 | diagnostic valid with route-only fail-closed probe | 21/21 pairs completed; `francis2023_intersection_wait` min-distance `+4.159358 m`; terminal metrics neutral | only explicit `single_pedestrians` timing |
| [#1943](https://github.com/ll7/robot_sf_ll7/issues/1943) / [#1944](https://github.com/ll7/robot_sf_ll7/pull/1944) | `single_pedestrian_speed_offset` | three-planner, seed-limit 4 | diagnostic valid with route-only fail-closed probe | 30/30 pairs completed; overall min-distance `-0.588347 m`; `intersection_wait` `-2.002917 m`; one ORCA seed changed collision to success | seed-local outcome flip, no route-ped speed contract |
| [#1945](https://github.com/ll7/robot_sf_ll7/issues/1945) / [#1946](https://github.com/ll7/robot_sf_ll7/pull/1946) | ORCA leave-group speed trace | `francis2023_leave_group`, ORCA | diagnostic trace valid | seed 258 flip inspected as a fragile phase/order mechanism | not replicated as robustness evidence |
| [#1947](https://github.com/ll7/robot_sf_ll7/issues/1947) / [#1948](https://github.com/ll7/robot_sf_ll7/pull/1948) | timing vs speed trace | `francis2023_intersection_wait` | diagnostic trace valid | timing trace `+4.159358 m`; speed trace `-2.002917 m` | trace comparison only, terminal metrics neutral |
| [#1949](https://github.com/ll7/robot_sf_ll7/issues/1949) / [#1950](https://github.com/ll7/robot_sf_ll7/pull/1950) | `single_pedestrian_wait_duration_offset` | three-planner, seed-limit 4 | diagnostic valid with two fail-closed probes | 9/9 eligible pairs completed; all mean deltas `0.0` | useful negative result for one wait-bearing scenario |
| [#1951](https://github.com/ll7/robot_sf_ll7/issues/1951) / [#1952](https://github.com/ll7/robot_sf_ll7/pull/1952) | intersection-wait phase grid | start-delay, speed, wait-duration magnitudes | diagnostic valid with one excluded negative-delay row | speed signs were monotonic by magnitude; start delay was strongly positive; wait duration stayed flat | one scenario, one pedestrian, three seeds |
| [#1953](https://github.com/ll7/robot_sf_ll7/issues/1953) / [#1954](https://github.com/ll7/robot_sf_ll7/pull/1954) | speed-grid trace | `speed_delta_m_s: +0.5` | diagnostic trace valid | same nearest-pedestrian index; phase effect explains most clearance loss | pedestrian IDs are inferred from indices |
| [#1955](https://github.com/ll7/robot_sf_ll7/issues/1955) / [#1956](https://github.com/ll7/robot_sf_ll7/pull/1956) | trace runner selector | tooling | valid tooling support | `--perturbed-variant-id` removes filtered-manifest workaround | not evidence by itself |
| [#1957](https://github.com/ll7/robot_sf_ll7/issues/1957) / [#1958](https://github.com/ll7/robot_sf_ll7/pull/1958) | `single_pedestrian_trajectory_waypoint_offset` | `goal`, seed-limit 2, horizon 40 | diagnostic smoke valid with one fail-closed probe | 2/2 pairs completed; min-distance `+0.034080 m`; terminal metrics neutral | one planner, two seeds, smoke strength only |
| [#1959](https://github.com/ll7/robot_sf_ll7/issues/1959) / [#1964](https://github.com/ll7/robot_sf_ll7/pull/1964) | `pedestrian_density_offset` | `goal`, seed-limit 2, horizon 40 | diagnostic smoke valid with one fail-closed probe | 2/2 pairs completed; all mean deltas `0.0` | density-to-count behavior is route/runtime dependent |

## Family Classification

| Family | Observed Sensitivity | Interpretation Strength | Redundancy / Novelty | Remaining Limitation | Verdict |
|---|---|---|---|---|---|
| `robot_route_offset` | very low: min-distance near zero, terminal metrics neutral | moderate for low-criticality route-offset claim on the tested slice | baseline family | no outcome signal; limited scenarios | consolidate, do not expand by default |
| `pedestrian_route_offset` | low to moderate clearance response, especially corridor rows | moderate diagnostic mechanism evidence | stronger than robot route offset | route-only support; no terminal signal | keep as useful trace diagnostic, not benchmark claim |
| `single_pedestrian_start_delay_offset` | high clearance response in `intersection_wait` | strong local phase-sensitivity evidence | distinct from route offsets | explicit single-ped only; terminal metrics neutral | promote into the reusable schema as a high-signal diagnostic family |
| `single_pedestrian_speed_offset` | high signed clearance response in `intersection_wait`; one seed-local ORCA outcome flip | strong local mechanism evidence after trace inspection | distinct from start delay and wait duration | seed-local outcome flip not replicated; route-ped speed unsupported | promote for diagnostic use with paired trace checks |
| `single_pedestrian_wait_duration_offset` | flat on tested magnitudes | moderate negative evidence | overlaps phase/timing but probes different field | one wait-bearing scenario only | keep as low-sensitivity evidence; do not repeat without new scenario justification |
| phase-grid combinations | high for speed/start-delay sign map, flat for wait duration | strong for local sign/magnitude map | synthesizes earlier single-point pilots | one scenario/pedestrian | use as the model for future family summaries |
| `single_pedestrian_trajectory_waypoint_offset` | small positive clearance in tiny smoke | low; smoke only | spatial single-ped path shape axis | one planner, two seeds | needs controlled follow-up before conclusions |
| `pedestrian_density_offset` | flat in tiny smoke | low; smoke only | crowd/density axis | route density is not exact add/remove-one contract | keep as smoke-proven but not yet informative |

## Criticality Summary Schema

Use `criticality_summary.v1` for future perturbation-family summaries:

```yaml
schema_version: criticality_summary.v1
parent_issue: "#1610"
source_issue: "#NNNN"
source_pr: "#NNNN"
perturbation_family: string
family_parameters:
  magnitude_grid: list
  target_selector: string
  caps: mapping
semantic_boundary: diagnostic_local | benchmark_candidate | benchmark_evidence | invalid | fallback | degraded | missing
validity_constraints:
  supported_scenario_surfaces: list
  fail_closed_conditions: list
  excluded_variants: list
regeneration:
  manifest: path
  command: string
  output_boundary: ignored_output | tracked_summary | durable_artifact
scenario_scope:
  source_scenarios: list
  eligible_scenario_types: list
planner_scope:
  planners: list
  planner_modes: list
seed_scope:
  seeds: list
  seed_limit: integer
paired_baseline:
  required: true
  noop_variant_rule: string
status_counts:
  completed: integer
  invalid: integer
  fallback: integer
  degraded: integer
  missing: integer
  failed: integer
effect_summary:
  success_delta: number | null
  collision_delta: number | null
  timeout_delta: number | null
  min_distance_delta_m: number | null
  trace_mechanism: string | null
interpretation:
  verdict: string
  confidence: low | medium | high
  caveats: list
next_action:
  routing: stop | consolidate | controlled_slice | promote_diagnostic | split_followup
  follow_up_issues: list
```

Rows with `invalid`, `fallback`, `degraded`, `missing`, or `failed` status must be counted
separately and excluded from completed-pair effect means. `diagnostic_local` rows must not be
reported as benchmark or paper-facing evidence.

## Stop And Continue Rules

Stop or deprioritize a new perturbation-family pilot when:

- it repeats an explained mechanism without changing planner, scenario, seed, or target surface;
- previous rows show no success, collision, timeout, or meaningful clearance sensitivity;
- interpretation rests on a single seed-local flip without trace confirmation;
- a row depends on route/loader semantics that are not explicit in the manifest contract;
- the result would not change a planner-design, scenario-design, benchmark, or robustness-training
  action.

Continue with a controlled slice when:

- a family shows clear metric sensitivity and the mechanism is still uncertain;
- trace evidence suggests a concrete planner/search-policy change to test;
- a family has only smoke proof but fills a real semantic gap in the perturbation vocabulary;
- the next run changes exactly one axis: family, magnitude grid, planner set, scenario set, or seed
  budget.

## Parent Routing

Recommendation for [#1610](https://github.com/ll7/robot_sf_ll7/issues/1610): consolidate now.

The #1610 lane should pause additional one-off family expansion until a perturbation-family
registry/schema exists and at least one controlled follow-up is selected from the synthesis above.
The strongest current diagnostic signal is the `francis2023_intersection_wait` phase/speed surface,
especially the speed sign/magnitude map and p050 trace mechanism. The weakest repeated signal is
small robot-route offsets. Wait-duration and density should be preserved as low-sensitivity evidence
instead of rerun casually.

Do not promote any current row into benchmark-strength or paper-facing evidence. A controlled S20/S30
slice is premature until `criticality_summary.v1` is implemented and the candidate diagnostic family
has an explicit paired-baseline, validity, and trace-review rule.

## Follow-Up Recommendations

Opened synthesis-backed follow-up issues:

1. [#1980](https://github.com/ll7/robot_sf_ll7/issues/1980):
   `validation: define perturbation-family registry and criticality_summary.v1 writer`
2. [#1981](https://github.com/ll7/robot_sf_ll7/issues/1981):
   `research: propagate #1610 perturbation mechanism findings into claim map/context notes`
3. Optional later controlled slice: `intersection_wait` speed/start-delay diagnostic with the
   schema-backed summary contract, only after the registry exists.

## Validation

This note is a synthesis over tracked issue/PR state and tracked context/evidence files. Validation
should use the docs proof consistency checker, link/path inspection, and `git diff --check`.
