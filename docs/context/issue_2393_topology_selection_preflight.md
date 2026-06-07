# Issue #2393 Topology Selection Preflight

Issue: [#2393](https://github.com/ll7/robot_sf_ll7/issues/2393)
Status: current, diagnostic-only preflight.

## Claim Boundary

This note names the next bounded topology hypothesis-selection revision after
[#2258](issue_2258_topology_primary_route_audit.md) and
[#2307](issue_2307_topology_score_diagnostic.md). It is a research-routing and
diagnostic-gate note, not benchmark evidence and not planner promotion evidence.

The prior evidence showed:

- topology alternatives were present often enough to reject an absence-only explanation;
- the route selector still chose `primary_route` on effectively every scored opportunity;
- every scored alternative in #2307 was rejected as `lower_topology_selection_score`;
- downstream topology-command wins inherited the selected hypothesis, so command wins did not
  become corrective reroutes.

## Named Revision

Proposed revision:
`topology_hypothesis_near_parity_diversity_gate_v0`.

The revision should keep `topology_guided_hybrid_rule_v0` diagnostic-only and change only the
upstream hypothesis selection surface. Instead of letting pure remaining-route length dominate every
masked alternative, score non-primary hypotheses through a near-parity gate:

1. keep `primary_route` as the default when no valid, distinct alternative exists;
2. consider only non-primary alternatives whose remaining route is close to the primary route,
   initially `primary_vs_best_alternative_route_distance <= 0.75 m` or no more than a 5 percent
   route-length increase;
3. require the alternative's static-clearance minimum to be no worse than the primary route by more
   than `0.05 m`;
4. within that gated set, apply a small diversity tie-break or bonus to the best non-primary
   hypothesis before selecting the winner;
5. report the selected hypothesis, margin, gate reason, and route-distance delta in the diagnostic
   trace.

Expected mechanism effect: if the route selector is overselecting `primary_route` mainly because
masked alternatives carry a small length penalty, the gate should convert near-parity opportunities
into at least one real non-primary selection while refusing long or less-clear detours. This targets
the observed score-surface failure without retuning downstream command scoring or making a broad
topology framework change.

This is intentionally narrower than a stateful stall bonus. The evidence currently proves
route-length and selection-score dominance more directly than a reusable stalled-primary state
model, and the existing diagnostic fields can falsify the near-parity hypothesis first.

## Diagnostic Gate

Use the same core slice as #2258/#2307 unless the implementation issue explains why it changed:

```yaml
candidate: topology_guided_hybrid_rule_v0
scenario_name: classic_realworld_double_bottleneck_high
seed: 111
horizon: 160
stage: full_matrix
min_hypotheses: 2
max_hypotheses: 3
required_fields:
  - alternative_hypothesis_count
  - selected_hypothesis
  - score_margin_to_primary_route
  - rejection_reason
  - switch_opportunity_count
  - primary_vs_best_alternative_route_distance
  - near_parity_gate_reason
  - selected_static_clearance_min_m
  - best_alternative_static_clearance_min_m
```

Canonical command shape:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python \
  scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2393_topology_near_parity_gate
```

## Decision Criteria

Accept the revision for a follow-up diagnostic rerun only if all of these hold:

- `diagnostic_status` remains `diagnostic_complete`;
- at least one non-primary route-selector selection occurs with margin greater than `1e-3`, not just
  a numerical tie;
- at least one topology-command influence entry uses a non-primary selected hypothesis, or the trace
  clearly shows why non-primary selection did not reach command influence;
- every selected non-primary frame satisfies the route-length slack and static-clearance floor;
- topology remains labeled diagnostic-only in the candidate registry and config.

Revise the revision if:

- non-primary selections happen only as numerical ties;
- the diversity gate activates on too many frames without producing command influence;
- `primary_vs_best_alternative_route_distance` shows the proposed `0.75 m` or 5 percent slack is
  too tight or too loose for this slice;
- the diagnostic lacks enough fields to distinguish a real gate activation from ordinary score
  noise.

Reject this topology mitigation lane for the slice if:

- `primary_route` still wins all route-selector and topology-command influence frames;
- non-primary selections require long or less-clear detours that violate the gate;
- the diagnostic fails closed or cannot preserve the required fields;
- the rerun can only be made to look favorable by changing downstream command scoring before the
  upstream selection gate is clear.

## Follow-Up Implementation Scope

A future implementation issue can execute this note without rereading the full #2258/#2307 history:

- add explicit `primary_vs_best_alternative_route_distance` and near-parity gate reason fields if
  the current trace output does not preserve them;
- add the minimal configurable slack/static-clearance/diversity parameters under the existing
  diagnostic-only topology candidate;
- add targeted tests for the gate using synthetic primary-vs-alternative hypotheses;
- rerun the diagnostic command above and classify the result as accept, revise, or reject.

Do not treat a successful single-slice gate as benchmark-strength mitigation evidence. It would only
justify a later benchmark or broader diagnostic contract.

## Follow-Up Result

Issue [#2518](https://github.com/ll7/robot_sf_ll7/issues/2518) implemented and reran this
near-parity gate on 2026-06-07. The result is recorded in
[issue_2518_topology_near_parity_gate.md](issue_2518_topology_near_parity_gate.md) with compact
evidence at
[evidence/issue_2518_topology_near_parity_gate_2026-06-07/summary.json](evidence/issue_2518_topology_near_parity_gate_2026-06-07/summary.json).

Classification: `accept` as diagnostic-only follow-up direction. The rerun produced 42 non-primary
route-selector selections and 7 non-primary topology-command influence selections, with no missing
required selected-row gate fields. It still left 70 of 160 frames with insufficient hypotheses, so
the accepted direction is selector scoring, not hypothesis availability or benchmark-strength
planner improvement.
