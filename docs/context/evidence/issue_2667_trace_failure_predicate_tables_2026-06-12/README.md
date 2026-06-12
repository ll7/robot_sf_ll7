# Issue #2667 Trace Failure Predicate Tables

Related issues: [#2667](https://github.com/ll7/robot_sf_ll7/issues/2667),
[#2654](https://github.com/ll7/robot_sf_ll7/issues/2654),
[#2543](https://github.com/ll7/robot_sf_ll7/issues/2543),
[#2428](https://github.com/ll7/robot_sf_ll7/issues/2428)

This bundle applies the denominator-aware trace failure predicate table CLI to a tracked diagnostic
input bundle.

Claim boundary: diagnostic-only. These rows exercise the table contract and preserve one
`not_available` row, but they are not benchmark rates, planner comparisons, paper-facing mechanism
frequencies, or safety evidence. A future predeclared benchmark matrix would need broader
scenario-family and seed denominators before using these predicates for benchmark conclusions.

## Inputs

- `inputs/synthetic_crossing_proxy_orca_111_trace_export.json`: tracked synthetic
  `simulation_trace_export.v1` fixture that mirrors the predicate contracts already covered by
  `tests/validation/test_trace_failure_predicates.py`. It includes close-interaction, late
  evasive, oscillatory-control, bottleneck-deadlock, timeout, and missing-occlusion-evidence
  signals.
- `../issue_2428_mechanism_trace_panels_2026-06-06/traces/default_social_force_trace_export.json`:
  tracked loader-valid default Social Force trace from the #2428 AMMV mechanism panel bundle.
- `../issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json`:
  tracked loader-valid AMMV-aware Social Force trace from the same #2428 bundle.

The #2428 traces provide durable real diagnostic trace provenance and produced no predicate rows in
this run. The synthetic trace provides a compact, versioned contract fixture so the preserved table
contains valid rows plus the fail-closed `not_available` occlusion row.

## Outputs

- `trace_failure_predicate_tables.json`: machine-readable aggregate table.
- `trace_failure_predicate_tables.md`: Markdown rendering of the same aggregate rows.
- `summary.json`: compact provenance, command, and interpretation summary.
- `checksums.sha256`: checksums for the versioned input and generated outputs.

## Result

The generated table includes 3 input traces and 7 aggregate rows. All observed predicate rows come
from the synthetic `crossing_proxy` / `orca` / seed `111` trace with denominator `1`; the two #2428
durable traces are included as negative diagnostic inputs and do not add predicate rows.

Observed predicate groups:

- `bottleneck_deadlock`: valid, high severity.
- `clearance_critical_interaction`: valid, high and medium severity rows.
- `late_evasive_reaction`: valid, high severity.
- `occlusion_triggered_near_miss`: `not_available`, `not_available` severity because
  `planner.occlusion_or_visibility` is absent.
- `oscillatory_local_control`: valid, medium severity.
- `zero_motion_timeout_behavior`: valid, high severity.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/build_trace_failure_predicate_tables.py \
  --trace docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/inputs/synthetic_crossing_proxy_orca_111_trace_export.json \
  --trace docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/default_social_force_trace_export.json \
  --trace docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/ammv_social_force_trace_export.json \
  --json-output docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/trace_failure_predicate_tables.json \
  --markdown-output docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/trace_failure_predicate_tables.md
python -m json.tool docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/inputs/synthetic_crossing_proxy_orca_111_trace_export.json
python -m json.tool docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/trace_failure_predicate_tables.json
```
