# Issue #2570 Topology Revise Status Propagation

Issue: [#2570](https://github.com/ll7/robot_sf_ll7/issues/2570)
Status: current, analysis-only propagation.

## Claim Boundary

This note propagates the current topology-selection status after
[#2530](issue_2530_topology_near_parity_corrective_smoke.md) classified the near-parity lane as
`revise` and [#2563](issue_2563_topology_corrective_revision.md) selected
`primary_route_reuse_penalty_under_near_parity_alternatives` as the next proposal. It does not add
new runtime evidence, benchmark evidence, or planner-promotion evidence.

## Current Topology Status

Topology guidance remains diagnostic-only. The current evidence supports a `revise` lane, not a
benchmark-improvement claim:

- [#2258](issue_2258_topology_primary_route_audit.md) and
  [#2403](issue_2403_topology_selection_score_decision.md) showed primary-route overselection with
  alternatives present.
- [#2518](issue_2518_topology_near_parity_gate.md) showed that near-parity gating can move route
  selection and local command influence off `primary_route`.
- [#2530](issue_2530_topology_near_parity_corrective_smoke.md) showed that the corrective-behavior
  bar was still not met: the canonical slice ended `horizon_exhausted`, so topology stayed
  `revise`.
- [#2563](issue_2563_topology_corrective_revision.md) selected one next hypothesis, not a completed
  mechanism implementation.

## Surfaces Updated

This propagation pass updates current guidance surfaces that future agents use for queue selection
and claim boundaries:

- [mechanism_closure_status.md](mechanism_closure_status.md): topology row now points to the
  Issue #2530 / Issue #2563 revise state and selected next hypothesis.
- [issue_2228_research_dashboard.md](issue_2228_research_dashboard.md): topology rows and queue
  guidance now route to Issue #2563 / Issue #2540 instead of the older Issue #2258 audit as the
  next step.
- [policy_search/candidate_registry_summary.md](policy_search/candidate_registry_summary.md):
  topology candidate summary now names the diagnostic-only `revise` boundary and removes the
  duplicate row.

Historical evidence notes and old policy-search reports were not rewritten when they already state
their own local claim boundary. They remain provenance for the chain above, not current queue
guidance.

## Open Issue Audit

| Issue | Action | Reason |
| --- | --- | --- |
| [#2540](https://github.com/ll7/robot_sf_ll7/issues/2540) | no body change | Already requires a paired/broadened diagnostic, an explicit comparator, and diagnostic-only claim boundary. It is the correct follow-up for Issue #2563. |
| [#2522](https://github.com/ll7/robot_sf_ll7/issues/2522) | comment | Body still referenced the older `#2519/#2518` lineage for topology near-parity diagnostics. Added a comment pointing report generation to Issue #2530, PR #2539, and Issue #2563. |
| [#2521](https://github.com/ll7/robot_sf_ll7/issues/2521) | comment | Body still described topology near-parity as the strongest mechanism-improvement lane via `#2519`. Added a comment freezing that lane at `revise` until the Issue #2563 hypothesis passes a diagnostic gate. |

No open issue was found that should be moved to blocked/hold solely because of topology revise
status. The key guard is to keep topology benchmark-claim work behind #2540 or a narrower
implementation child derived from #2563.

## Validation

For this analysis-only propagation:

```bash
rtk uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
rtk git diff --check
```

Full runtime validation is intentionally skipped because no planner behavior or benchmark metric is
changed.
