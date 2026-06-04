# Issue #2222 Perturbation Criticality Metric

Date: 2026-06-04

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2222>

Status: diagnostic synthesis and reusable metric contract from existing issue #1610 perturbation
evidence. This note does not run a new perturbation campaign and does not make paper-facing causal
claims.

## Purpose

The issue #1610 perturbation lane produced useful local pilots, but the family-specific results
need one reusable statistic and validity contract before more perturbation families are added.
`criticality_metric.v1` standardizes how future reports should classify perturbation families,
paired-row effects, and stop/continue decisions.

## Metric Schema

Use one row per perturbation family, scenario-family slice, planner set, and seed set.

```yaml
schema_version: criticality_metric.v1
source_issue: "#NNNN"
source_note: docs/context/...
perturbation_family: string
scenario_family: string | all_tested
planner_set: list
seed_count: integer
paired_row_status:
  completed: integer
  invalid: integer
  fallback: integer
  degraded: integer
  missing: integer
  failed: integer
metric_inputs:
  outcome_flip_rate: number | null
  success_delta: number | null
  collision_delta: number | null
  timeout_delta: number | null
  min_distance_delta_m: number | null
  clearance_delta_m: number | null
  progress_delta: number | null
mechanism_label: string
validity_status: interpretable | redundant | invalid | incomplete | missing_evidence
criticality_class: high | medium | low | flat | smoke_only | inconclusive
claim_boundary: diagnostic_only | proposal | blocked | benchmark_candidate
decision:
  routing: stop | consolidate | controlled_followup | trace_review | heldout_protocol
  rationale: string
```

## Validity Rules

- Compute effects only on completed paired baseline/perturbed rows that match planner, seed,
  scenario family, horizon, and perturbation family.
- Count invalid, fallback, degraded, missing, and failed rows separately; do not include them in
  effect means or ranking-support claims.
- Treat terminal outcome flips as high signal only when they are replicated or paired with
  trace-backed mechanism evidence. Single seed-local flips remain diagnostic.
- Treat large clearance or progress shifts with neutral terminal metrics as mechanism evidence, not
  planner-performance evidence.
- Mark route-only, single-pedestrian-only, density, or waypoint perturbations with their supported
  scenario surface instead of generalizing across all scenario families.
- Use [issue_2234_predictive_perturbation_criticality.md](issue_2234_predictive_perturbation_criticality.md)
  for held-out predictive validation. This issue defines the metric; #2234 defines the next
  predictive test protocol.

## Existing Family Classification

The compact machine-readable classification is tracked at:

- `docs/context/evidence/issue_2222_criticality_metric_2026-06-04/criticality_metric_summary.csv`
- `docs/context/evidence/issue_2222_criticality_metric_2026-06-04/criticality_metric_manifest.json`

Summary by family:

| Family | Criticality class | Validity | Routing | Interpretation |
| --- | --- | --- | --- | --- |
| `robot_route_offset` | `low` | `interpretable` | `stop` | Terminal metrics stayed neutral and min-distance deltas were near zero across one-seed, four-seed, and stronger-planner slices. |
| `pedestrian_route_offset` | `medium` | `interpretable` | `trace_review` | Corridor clearance responds locally, but route-only support and neutral terminal metrics keep it diagnostic. |
| `single_pedestrian_start_delay_offset` | `high` | `interpretable` | `controlled_followup` | `francis2023_intersection_wait` shows the clearest positive clearance phase response. |
| `single_pedestrian_speed_offset` | `high` | `interpretable` | `heldout_protocol` | Strong signed clearance response and one ORCA seed-local outcome flip; needs #2234-style held-out validation before predictive claims. |
| `single_pedestrian_wait_duration_offset` | `flat` | `interpretable` | `consolidate` | Eligible rows stayed flat on current magnitudes; preserve as a low-sensitivity negative control. |
| `single_pedestrian_trajectory_waypoint_offset` | `smoke_only` | `incomplete` | `controlled_followup` | Tiny smoke completed, but one planner/two seeds are insufficient for mechanism conclusions. |
| `pedestrian_density_offset` | `smoke_only` | `incomplete` | `consolidate` | Tiny smoke stayed flat; density-to-count behavior remains route/runtime dependent. |

## Stop/Continue Decision

Stop adding new perturbation families until a future executable issue either:

1. runs the #2234 held-out validation protocol for the high-signal `intersection_wait`
   speed/start-delay family, or
2. implements a writer that emits `criticality_metric.v1` rows from paired perturbation outputs.

Continue only when the next issue changes exactly one axis: perturbation family, magnitude grid,
planner set, scenario set, seed budget, or trace-review target. Do not add both a new family and a
new planner/scenario surface in the same first follow-up.

## Claim Boundary

This synthesis supports a reusable diagnostic method and research-routing decision. It does not
support causal scenario-feature claims, robustness claims, or paper-facing planner claims. Any
future metric promotion requires executable evidence with paired-row status accounting, explicit
thresholds, and fallback/degraded/unavailable exclusions.

## Validation

This note and compact classification manifest should be validated with docs proof consistency,
JSON/CSV inspection, cited-path checks, and `git diff --check`.
