# Issue #2518 Topology Near-Parity Gate

Issue: [#2518](https://github.com/ll7/robot_sf_ll7/issues/2518)
Status: current, diagnostic-only result.

## Claim Boundary

This note records the bounded diagnostic rerun for
`topology_hypothesis_near_parity_diversity_gate_v0`, the follow-up revision named by
[#2393](issue_2393_topology_selection_preflight.md) after the #2307/#2403 primary-route
overselection diagnosis. It is not benchmark success evidence and does not promote
`topology_guided_hybrid_rule_v0` out of diagnostic-only status.

## Result

Classification: `accept` for the next topology-selection research direction.

The canonical slice completed:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2518_topology_near_parity_gate
```

Compact evidence:
[evidence/issue_2518_topology_near_parity_gate_2026-06-07/summary.json](evidence/issue_2518_topology_near_parity_gate_2026-06-07/summary.json)

Observed signals:

- `diagnostic_status`: `diagnostic_complete`
- topology status counts: `ok=90`, `insufficient_hypotheses=70`
- route-selector selections: `primary_route=56`, non-primary hypotheses total `42`
- selected near-parity gate reasons: `selected_non_primary_near_parity=42`,
  `route_distance_exceeds_slack=25`, `static_clearance_floor_failed=17`,
  `eligible_near_parity_alternative=14`
- selected ok rows missing required #2518 fields: `0` for
  `primary_vs_best_alternative_route_distance`, `near_parity_gate_reason`,
  `selected_static_clearance_min_m`, and `best_alternative_static_clearance_min_m`
- topology-command influence: `primary_route=26`, `masked_cell_84_103=6`,
  `masked_cell_85_103=1`

The issue acceptance bar required at least one non-primary route-selector selection for `accept`;
the run produced 42. It also produced 7 non-primary topology-command influence entries, so the gate
did not stop at an internal selector-only trace artifact.

## Interpretation

The near-parity diversity gate falsifies the strongest negative reading of #2403 for this slice:
primary-route dominance is not inevitable once the selector can distinguish near-parity alternatives
from long or less-clear detours. The selected non-primary examples have finite route-distance delta,
selected static clearance, and best-alternative static clearance fields, so the diagnostic can explain
why the gate activated.

The revision should remain diagnostic-only. The result is useful because it moves the topology
selection research lane from `primary_route_overselected` to a concrete accepted diagnostic
direction, not because it proves benchmark improvement.

## Remaining Limits

This does not solve topology hypothesis availability: 70 of 160 frames still had insufficient
hypotheses. It also does not tune a production planner; `near_parity_diversity_bonus=0.5` is a
diagnostic tie-break value chosen to overcome the observed small route-length dominance inside the
preflight slack/floor gate.

Recommended next step: run a broader diagnostic sweep or benchmark-facing experiment that keeps the
near-parity gate diagnostic-only but tests whether the selector change improves route progress,
success, or mechanism evidence beyond this single scenario/seed slice.
