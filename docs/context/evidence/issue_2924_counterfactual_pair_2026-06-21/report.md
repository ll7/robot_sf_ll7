# Issue #2924 Counterfactual Pair Report

## Boundary

- Evidence tier: `analysis_only`
- Claim boundary: `analysis_only_not_benchmark_or_paper_grade_evidence`
- Fallback/degraded/not_available rows: fail closed before verdict calculation.

## Pair

- Pair id: `issue_2924_prediction_risk_occluded_emergence`
- Scenario: `issue_2924_prediction_risk_occluded_emergence`
- Seed: `111`
- Planner: `hybrid_rule_v0_minimal`
- Artifact invariant: `issue_2924_prediction_risk_pair_fixture.v1`

## Result

- Expected mechanism: `prediction_risk_gating`
- Activation delta: `1` active rows
- Outcome delta: `min_clearance_m` `0.18` -> `0.42` (delta `0.24`; expected `increase`)
- Hypothesis verdict: `survived`
- Reason: mechanism activated and min_clearance_m moved increase by +0.24

## Trace Panels

- Panel manifest: `docs/context/evidence/issue_2924_counterfactual_pair_2026-06-21/panels/trajectory_panel_manifest.json`
- Captions: `docs/context/evidence/issue_2924_counterfactual_pair_2026-06-21/panels/captions.md`
