# Issue 4195 h600 Aggregation Artifact

This directory contains diagnostic-only h600 interpretation artifacts for jobs 13268, 13273.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- Scope: per-planner aggregation plus issue #4195 checklist items 3-5.
- This artifact does not assert benchmark success, dissertation-ready evidence, paper-grade evidence, or a planner ranking claim.
- No full benchmark campaign, Slurm submission, graphics processing unit job, retention decision, or dissertation claim edit was run for this slice.
- SNQI recalibration, horizon-sensitivity, and exposure diagnostics are diagnostic-only interpretation artifacts.
- Comparability is limited to shared planner arms whose `scenario_matrix_hash` and `comparability_mapping_hash` match across the two campaign summaries.

## Contents

- `planner_metric_summary.csv`: one row per job, planner, and metric with per-seed values where available plus bootstrap confidence intervals.
- `planner_metric_summary.md`: Markdown rendering of the same rows.
- `comparability_check.json` and `comparability_check.md`: shared-arm scenario matrix comparability check.
- `snqi_recalibration_bundle.json` and `snqi_recalibration_report.md`: analysis-only h600 recalibration and h500 reversal checks.
- `horizon_sensitivity_report.json` and `horizon_sensitivity_report.md`: h600-vs-h500 rank-stability and rank-flip diagnostic.
- `interaction_exposure_diagnostics.json` and `interaction_exposure_diagnostics.md`: episode-level exposure coverage readiness; fail-closed when required fields are absent.
- `h600_mechanism_labels_sidecar.csv` `h600_interaction_exposure_sidecar.csv`: retained h600 episode-row sidecars from issue #4242. All retained rows are explicit `not_derivable_missing_trace`; no geometry-only mechanism labels or exposure zero-imputation are used.
- `h600_mechanism_exposure_backfill_manifest.json` `h600_mechanism_exposure_backfill_report.md`: compact sidecar provenance, status counts, and diagnostic-only claim boundary.
- `hybrid_roster_h600_transfer_packet.md`: pre-registered h600 hybrid-roster campaign (job 13282, issue #4230) — the F-C4(ii) gate input leg; AI-generated, needs review, promotes nothing.
- `f_c4_ii_interpretation_gate.md`: integrates the hybrid-roster packet (13282) with the confirm/extended bundle (13268/13273) into the diagnostic F-C4(ii) boundary (supported / diagnostic-only / not-supported) so issue #4195 checklist reading comes from committed artifacts. Records the maintainer sign-off (2026-07-03, F-C4 draft→supported, diss `0d853df`) as durable provenance; the evidence stays diagnostic-tier and this note promotes nothing on its own.
- `source_manifest.json`: input paths, campaign metadata, and source file SHA-256 digests.
- `SHA256SUMS`: checksums for generated files in this directory.

## F-C4(ii) integration gate

- The h600 hybrid-roster leg (job 13282) shares the confirm/extended scenario-matrix hash `c10df617a87c`, the comparability precondition for reading F-C4(ii) across all three legs.
- Gate reading is `diagnostic-only`: hybrids separate above every prediction-equipped arm at h600 (Δ ≥ 0.20, disjoint CIs); the hybrid-vs-ORCA lead stays CI-overlapping (diagnostic) at the 3-seed budget. See `f_c4_ii_interpretation_gate.md` for the full boundary.
- Sign-off: the maintainer promoted F-C4 draft→supported (pillars (i)+(ii)) at exactly this guarded wording on 2026-07-03 (diss `0d853df`), closing the #4195 interpretation chain; pillar (iii) stays draft (outside #4195).
- Validate the integration with `uv run python scripts/validation/check_issue_4195_f_c4_ii_gate.py` (fail-closed: SHA256SUMS coverage/digest match, required boundary sections, shared-matrix-hash agreement).

- Terminality review: `issue_4195_terminality_review.md` records that the #4195 checklist and sign-off boundary are satisfied, verifies PR #4321 and PR #4374 are merged, confirms no open pull request covers the closure scope, and keeps pillar (iii) documentation plus any S30 hybrid-vs-ORCA escalation outside #4195.
- Issue #4230 terminality: `issue_4230_terminality_review.md` and `issue_4230_terminality_summary.json` record that PR #4265, completed job 13282 packet, and #4195 sign-off close the h600 hybrid-roster implementation lane; S30 or pillar-(iii) follow-ups remain outside issue #4230.

## Notes

- Metric rows: 80.
- Shared-arm comparability status: `pass`.
- SNQI recalibration status: `ok`.
- Horizon-sensitivity status: `ok`.
- Interaction-exposure status: `blocked_missing_required_fields`.
- Comfort rows: 16.
- Comfort per-seed values are not present in `seed_episode_rows.csv`; those rows preserve the campaign-summary aggregate mean confidence interval and are marked `no_seed_episode_column`.

## Issue #4239 SNQI Weight-Set Ranking Packet

The issue #4239 packet is diagnostic-only h600 Social Navigation Quality Index (SNQI)
weight-set ranking support for jobs `13268` and `13273`. It de-duplicates shared planner arms,
keeps the three-seed caveat, and does not choose canonical weights, edit paper or dissertation
claims, run campaigns, submit Slurm or graphics processing unit jobs, or copy raw output trees.

New compact artifacts, when the fail-closed preflight is ready, use the
`snqi_weight_set_h600_*` prefix and are checksummed in `SHA256SUMS`.
