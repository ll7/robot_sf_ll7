# Issue #2402 Static-Recenter Activation Evidence

This directory preserves compact analysis-only evidence for Issue #2402. It does not copy raw
diagnostic traces from `output/`; instead it maps the already promoted Issue #2306 instrumented
static-recenter rerun to the explicit activation fields and decision outputs requested by Issue
Issue #2402.

Files:

- `summary.json`: machine-readable decision summary, source diagnostic provenance, requested-field
  values, and claim boundary.
- `activation_fields.csv`: one row per held-out scenario with the Issue #2402 activation fields.
- `decision_outputs.csv`: the requested decision-output vocabulary with active and rejected
  classifications.

Source evidence:

- `docs/context/issue_2306_static_recenter_activation_trace.md`
- `docs/context/evidence/issue_2306_static_recenter_activation_trace_2026-06-05/summary.json`

Claim boundary: diagnostic-only synthesis; not benchmark-strength mitigation or transfer evidence.
