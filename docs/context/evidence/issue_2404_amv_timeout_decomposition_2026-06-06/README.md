# Issue #2404 AMV Timeout Decomposition Evidence

This directory preserves compact analysis-only evidence for Issue #2404. It does not copy raw
diagnostic traces from `output/`; instead it maps the already promoted Issue #2308 AMV timeout
trace rerun to the explicit decomposition fields and decision outputs requested by Issue #2404.

Files:

- `summary.json`: machine-readable decision summary, source diagnostic provenance, requested-field
  values, and claim boundary.
- `decomposition_fields.csv`: one row per requested Issue #2404 decomposition field.
- `decision_outputs.csv`: the requested decision-output vocabulary with the active classification
  and rejected alternatives.

Source evidence:

- `docs/context/issue_2308_amv_timeout_trace_analysis.md`
- `docs/context/evidence/issue_2308_amv_timeout_trace_2026-06-05/summary.json`

Claim boundary: diagnostic-only synthesis; not benchmark-strength AMV mitigation evidence.
