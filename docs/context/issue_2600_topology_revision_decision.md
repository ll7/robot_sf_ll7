# Issue #2600 Topology Revision Decision

Issue: [#2600](https://github.com/ll7/robot_sf_ll7/issues/2600)
Status: current, diagnostic-only decision.

## Claim Boundary

This note resolves the Issue #2600 decision gate by linking the current topology
evidence chain to the already-selected revision in
[#2563](issue_2563_topology_corrective_revision.md). It adds no runtime evidence,
benchmark evidence, planner promotion, or leaderboard claim. Topology guidance remains
diagnostic-only until the selected revision passes a controlled follow-up diagnostic.

## Decision

```yaml
topology_revision_decision:
  selected_revision: primary_route_reuse_penalty_under_near_parity_alternatives
  expected_effect: >
    Penalize repeated primary_route reuse when eligible near-parity alternatives remain
    available and recent primary-route selections have not produced enough route-progress
    evidence to justify another reselection.
  target_failure_mode: >
    The near-parity gate can move selection and local-command influence off primary_route,
    but the corrective smoke still exhausted the horizon with weak route-progress change.
    Selection diversity alone is therefore insufficient; the next revision should test
    whether primary-route reuse is suppressing progress after alternatives become eligible.
  input_evidence:
    - docs/context/issue_2258_topology_primary_route_audit.md
    - docs/context/issue_2403_topology_selection_score_decision.md
    - docs/context/issue_2518_topology_near_parity_gate.md
    - docs/context/issue_2530_topology_near_parity_corrective_smoke.md
    - docs/context/issue_2563_topology_corrective_revision.md
    - docs/context/issue_2570_topology_revise_status_propagation.md
  diagnostic_gate: >
    Run the same canonical full_matrix double-bottleneck slice, with an explicit comparator
    and the fields named in #2563, before expanding to broader topology benchmark work.
  reject_if: >
    Non-primary topology-command influence collapses to zero, required diagnostic fields
    fail closed, or route labels change without plausible route-progress or terminal-behavior
    improvement.
  related_issue_2540_action: narrow_scope
  claim_boundary: diagnostic_only
```

## Evidence Rationale

The selected revision is not new in this note; #2563 already chose
`primary_route_reuse_penalty_under_near_parity_alternatives` after the #2530 `revise`
classification. This note makes that choice the explicit #2600 resolution and prevents
future topology work from re-opening all four candidate families in parallel.

The evidence chain supports the selected revision more directly than the deferred alternatives:

- [#2258](issue_2258_topology_primary_route_audit.md) and
  [#2403](issue_2403_topology_selection_score_decision.md) showed primary-route
  overselection while alternatives were present and scored.
- [#2518](issue_2518_topology_near_parity_gate.md) showed the near-parity gate can produce
  non-primary route-selector selections and non-primary topology-command influence.
- [#2530](issue_2530_topology_near_parity_corrective_smoke.md) showed the same canonical
  slice still ended `horizon_exhausted`, with weak route-progress change, so the lane stayed
  `revise` rather than `continue`.
- [#2563](issue_2563_topology_corrective_revision.md) narrowed the next mechanism to a
  primary-route reuse penalty and deferred broader sequence-diversity, strict stall-trigger,
  and hysteresis/switch-cost families.
- [#2570](issue_2570_topology_revise_status_propagation.md) propagated the diagnostic-only
  status and routed follow-up work through #2540 or a narrower child derived from #2563.

## Issue #2540 Action

Issue [#2540](https://github.com/ll7/robot_sf_ll7/issues/2540) should be narrowed to the
selected #2563 revision instead of executing as an open-ended paired or broadened topology
diagnostic. The useful next child is either:

- an implementation/launch packet for the primary-route reuse penalty under the #2563 gate; or
- a smaller diagnostic-design issue if the current trace fields are insufficient to evaluate
  reuse, route progress, switch count, and terminal behavior against the comparator.

It should not run another broad smoke that treats selection diversity alone as planner
improvement.

## Validation

For this decision note, use cheap docs validation:

```bash
rtk uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
rtk git diff --check
```

No runtime planner validation is expected here because #2600 selects and records the next
hypothesis boundary; it does not implement the mechanism or add benchmark evidence.
