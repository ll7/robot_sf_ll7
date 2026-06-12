# Issue #2667 Trace Failure Predicate Tables

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2667>

## Scope

Issue #2667 applies the denominator-aware table CLI from #2654 to versioned
`simulation_trace_export.v1` inputs and records the diagnostic implication for the trace-failure
predicate lane.

The durable evidence bundle is:

- [evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/README.md](evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/README.md)

## Input Provenance

The run combines:

- a tracked synthetic `crossing_proxy` / `orca` / seed `111` trace fixture added under the #2667
  evidence directory, mirroring the predicate contracts covered by
  `tests/validation/test_trace_failure_predicates.py`;
- the two tracked #2428 AMMV/default Social Force `simulation_trace_export.v1` traces from
  [issue_2428_mechanism_trace_panels.md](issue_2428_mechanism_trace_panels.md).

The #2428 traces preserve real durable diagnostic provenance but produced no predicate rows. The
synthetic trace exists to make the table artifact exercise valid rows and the fail-closed
`not_available` occlusion behavior on a committed input.

## Result

The generated `trace_failure_predicate_tables.v1` artifact contains:

- 3 input traces;
- 7 aggregate rows;
- 6 valid rows;
- 1 `not_available` row for `occlusion_triggered_near_miss` because
  `planner.occlusion_or_visibility` is absent.

Observed valid predicate groups cover bottleneck deadlock, clearance-critical interaction, late
evasive reaction, oscillatory local control, and zero-motion timeout behavior.

## Denominators

Each aggregate row reports `trace_denominator` for its
`scenario_family` / `planner_id` / `seed` / trace-source group. The predicate rows in this run are
all from `crossing_proxy` / `orca` / seed `111`, so their row-level denominator is `1`.

Issue #2687 tightened future table generation so JSON and Markdown outputs also preserve
`zero_predicate_groups` when a trace group has zero valid predicates. This separates:

- a true zero valid-predicate group;
- a `not_available` predicate row caused by missing evidence fields;
- a trace absent from the input set;
- excluded, fallback, or degraded runs that should not enter the denominator.

The original #2667 evidence bundle remains diagnostic-only and was not regenerated as a new
benchmark result.

## Interpretation

This is diagnostic-only evidence. It proves the table-generation path can consume versioned
`simulation_trace_export.v1` inputs, preserve denominator fields, and carry a fail-closed
`not_available` predicate row into JSON/Markdown outputs. After #2687, future table outputs also
make zero valid-predicate groups explicit, but rate claims still require a predeclared benchmark
matrix.

It does not establish predicate rates, planner rankings, AMMV benefit, benchmark outcomes,
paper-facing mechanism frequencies, or safety claims. A future predeclared benchmark matrix should
include these predicates only after choosing scenario-family and seed denominators up front and
recording how zero-row trace groups should be represented.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/build_trace_failure_predicate_tables.py \
  --trace docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/inputs/synthetic_crossing_proxy_orca_111_trace_export.json \
  --trace docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/default_social_force_trace_export.json \
  --trace docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json \
  --json-output docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/trace_failure_predicate_tables.json \
  --markdown-output docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/trace_failure_predicate_tables.md
python -m json.tool docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/inputs/synthetic_crossing_proxy_orca_111_trace_export.json
python -m json.tool docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/summary.json
python -m json.tool docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/trace_failure_predicate_tables.json
```
