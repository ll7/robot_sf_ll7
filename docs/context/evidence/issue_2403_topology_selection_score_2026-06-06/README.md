# Issue #2403 Topology Selection-Score Evidence

This directory preserves compact analysis-only evidence for Issue #2403. It does not copy raw
diagnostic traces from `output/`; instead it maps the already promoted Issue #2307 instrumented
topology rerun to the explicit fields and decision outputs requested by Issue #2403.

Files:

- `summary.json`: machine-readable decision summary, source diagnostic provenance, requested-field
  values, and claim boundary.
- `field_coverage.csv`: one row per requested Issue #2403 field showing whether the field was
  produced or only documented as absent.
- `decision_outputs.csv`: the requested decision-output vocabulary with the active classification
  and rejected alternatives.

Source evidence:

- `docs/context/issue_2307_topology_score_diagnostic.md`
- `docs/context/evidence/issue_2307_topology_score_diagnostic_2026-06-05/summary.json`

Claim boundary: diagnostic-only synthesis; not benchmark-strength mitigation evidence.
