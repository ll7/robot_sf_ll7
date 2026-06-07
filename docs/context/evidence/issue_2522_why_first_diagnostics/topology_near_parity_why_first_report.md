# Why-First Report: Topology Near-Parity Corrective Diagnostics

## Outcome Summary

- Planner/scenario: `topology_guided_hybrid_rule_v0` on `classic_realworld_double_bottleneck_high`.
- Outcome: near-parity gating activated and reached command arbitration, but the corrective-behavior smoke exhausted the horizon without success.
- Execution status: `failed`.
- Limitation: this row is fallback/degraded/failed/not-available evidence and must not be counted as benchmark success.
- Metrics: `route_selector_non_primary_selection_count`=42, `topology_command_non_primary_influence_count`=7, `topology_command_steps`=33, `max_route_progress_delta_m`=0.16812408921843236, `success`=false.

## Mechanism Activation

- Mechanism: `topology_hypothesis_near_parity_diversity_gate_v0`.
- Activation status: `activated_but_not_corrective`.
- Evidence: Issue #2518 accepted the diagnostic selector signal with 42 non-primary selections; Issue #2530 preserved that signal but ended horizon_exhausted with no success.

## Failure Mechanism Classification

- Classification: `topology_signal_without_route_progress`.
- Rationale: The topology signal left primary-route dominance and influenced commands, but route progress and terminal outcome did not satisfy the corrective-behavior bar.

## Paired Comparator

- Comparator: `issue_2518_selector_acceptance_vs_issue_2530_corrective_smoke`.
- Comparator outcome: selection diversity remained present, but the follow-up corrective smoke classified the lane as revise.
- Delta: `non_primary_selection_count`=0, `classification_changed_from_accept_to_revise`=true, `non_primary_topology_command_steps`=7.

## Trace Evidence

- docs/context/evidence/issue_2518_topology_near_parity_gate_2026-06-07/summary.json.
- docs/context/evidence/issue_2530_topology_near_parity_corrective_smoke_2026-06-07/summary.json.
- docs/context/issue_2518_topology_near_parity_gate.md.
- docs/context/issue_2530_topology_near_parity_corrective_smoke.md.

## Alternative Explanations

- Insufficient topology hypotheses remained common: 70 of 160 frames lacked enough hypotheses.
- The diversity bonus is a diagnostic tie-break parameter rather than a production tuning claim.
- The single scenario and seed may expose route-progress geometry rather than a general topology-guidance effect.

## Continue / Revise / Stop Decision

- Decision: `revise`.
- Rationale: The reports clarify that topology near-parity is a real diagnostic signal but not a completed corrective mechanism.
- Next step: Test the #2563 primary-route reuse penalty in a paired diagnostic that preserves the #2530 fields and compares route progress or terminal behavior before any benchmark claim.

## Claim Boundary

Report strength is limited to the compact input evidence.
