# Issue #2543 Trace Failure Predicates

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2543>

## Scope

Issue #2543 adds a compact trace-level predicate contract for dissertation-relevant
local-navigation failure mechanisms. The implementation lives in
`robot_sf/analysis_workbench/trace_failure_predicates.py` and consumes typed
`simulation_trace_export.v1` payloads from
`robot_sf/analysis_workbench/simulation_trace_export.py`.

This is analysis-workbench infrastructure only. It does not report benchmark rates, planner
superiority, real-world safety, or paper-facing mechanism frequencies.

## Contract

The extractor emits `trace_failure_predicates.v1` payloads with:

- `predicate_id`;
- `time_interval_s`;
- `steps`;
- `involved_actors`;
- `scenario_family`;
- `planner_id`;
- `evidence_fields`;
- `severity`;
- `validity_status`;
- summary counts by scenario family, planner, validity status, and predicate ID.

Initial predicate IDs:

- `late_evasive_reaction`;
- `oscillatory_local_control`;
- `occlusion_triggered_near_miss`;
- `bottleneck_deadlock`;
- `zero_motion_timeout_behavior`;
- `clearance_critical_interaction`.

Issue #2543 implements the first tested smoke semantics for all initial predicate IDs:

- `late_evasive_reaction`;
- `clearance_critical_interaction`;
- `oscillatory_local_control`;
- `zero_motion_timeout_behavior`;
- `bottleneck_deadlock`;
- `occlusion_triggered_near_miss`.

The occlusion predicate is fail-closed: near-miss geometry without explicit occlusion or visibility
fields emits `validity_status: not_available` instead of inferring an occlusion-triggered failure.

## Evidence

Tracked compact evidence:

- [evidence/issue_2543_trace_failure_predicates/summary.json](evidence/issue_2543_trace_failure_predicates/summary.json)
- Follow-up denominator-aware table application:
  [issue_2667_trace_failure_predicate_tables.md](issue_2667_trace_failure_predicate_tables.md)

The smoke fixture in `tests/validation/test_trace_failure_predicates.py` emits three valid
predicate types from a synthetic `simulation_trace_export.v1` trace and a separate
`not_available` occlusion row when visibility evidence is absent.

## Validation

```bash
rtk uv run pytest tests/validation/test_trace_failure_predicates.py -q
rtk uv run ruff check robot_sf/analysis_workbench/trace_failure_predicates.py robot_sf/analysis_workbench/__init__.py tests/validation/test_trace_failure_predicates.py
```

Expected result: targeted predicate tests pass and Ruff reports no issues.

## Claim Boundary

These predicates are explicit trace diagnostics and table-preparation fields. They can support
future failure-mode tables by making evidence fields machine-readable, but any denominator-aware
rates, planner comparisons, or dissertation Results claims require a separate benchmark issue with
scenario-family denominators and durable execution evidence.
