# Issue 1055 Exposure-Aware H500 Tables

Date: 2026-05-07

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1055>

## Goal

Define and populate exposure-aware h500 reporting tables so long-horizon completion gains are read
alongside duration, collision, near-miss, force-exposure, and comfort-exposure signals.

## Evidence

Generated representative report:

* `docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/report.md`
* `docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/fixed_vs_h500_outcome_table.csv`
* `docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/exposure_aware_trace_table.csv`

Source evidence:

* `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv`
* `docs/context/issue_1056_h500_failure_classification.md`

## Table Schema

### Fixed vs H500 Outcome Table

Required columns:

* `mechanism_target`, `scenario_id`, `seed`, and `classification`
* fixed and h500 success indicators
* fixed and h500 collision indicators
* fixed and h500 episode length in steps
* success, collision, and step deltas

Purpose: separate strict-time completion effects from long-horizon safety regressions.

### Exposure-Aware Trace Table

Required columns:

* planner/scenario identity and h500 classification
* variant (`fixed_h100` or `h500`)
* success and collision indicators
* `steps_recorded` and `sim_seconds`
* raw near-miss count, near misses per episode, near misses per successful episode
* near misses per step and per simulated second when retained traces support those rates
* force-exposure step count and per-step/per-second rates
* comfort-exposure sum and per-step rate
* minimum robot-pedestrian distance when available
* retained trace pointer

Purpose: make runtime exposure visible so h500 is not reduced to a single winner table.

## Representative Findings

| Scenario | Class | H500 success delta | H500 collision delta | Exposure interpretation |
|---|---|---:|---:|---|
| `classic_bottleneck_low` | `time_budget_clean_relief` | +1 | 0 | Clean h500 completion; no near-miss, force-exposure, comfort-exposure, or pedestrian-distance signal in the retained trace. |
| `classic_t_intersection_medium` | `exposure_enabled_completion` | +1 | 0 | H500 succeeds, but force-exposure steps rise from 9 to 50 and comfort exposure from 3.0 to 16.667. |
| `classic_merging_low` | `safety_regressed_long_horizon` | 0 | +1 | H500 runs 173 more steps than fixed h100 and reaches collision after force exposure starts. |

## Interpretation Rules

* Report h500 success together with duration and safety/exposure terms.
* Use per-step or per-second rates only when retained traces or raw episodes support them.
* Mark missing raw-trace rates unavailable; do not infer them from aggregate summaries.
* Do not treat zero near-miss counts as proof of safety when force exposure, comfort exposure, or
  collision increased.
* Keep fallback/degraded rows out of aggregate h500 tables unless they are explicitly labeled and
  excluded from benchmark-success claims.

## Validation

Manual report path:

```bash
uv run python - <<'PY'
# Generated from
# docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv
# into docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/
PY
```

Validation commands:

* `rtk column -s, -t docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/fixed_vs_h500_outcome_table.csv`
* `rtk column -s, -t docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/exposure_aware_trace_table.csv`
* `rtk git diff --check`

No reusable report-generation code was introduced, so no new unit tests are required.
