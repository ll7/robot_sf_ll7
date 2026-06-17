# Issue #2981 ORCA Residual Emission Evidence

This directory contains diagnostic-only evidence for the scripted ORCA residual mechanism-trace
emission path.

Files:

- `orca_residuals_planner_decision_trace.v1.json`: tracked planner-decision fixture input.
- `orca_residual_mechanism_trace.jsonl`: emitted `mechanism_trace.v1` rows.
- `orca_residual_emission_report.json`: compact provenance, row count, and classification-count
  report.

Boundary: these artifacts prove only that durable fixture input can be transformed into
schema-valid `orca_residuals` mechanism-trace rows through the script path. They do not prove a
benchmark outcome, planner comparison, safety improvement, or paper-facing claim.
