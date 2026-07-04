# Issue #2660 Topology Successor Gate After Reuse-Penalty Regression

Issue: [#2660](https://github.com/ll7/robot_sf_ll7/issues/2660)
Status: historical successor-selection decision. The selected child implementation and paired
diagnostic result are now recorded in
[issue_2704_progress_gated_topology_successor.md](../issue_2704_progress_gated_topology_successor.md).

## Claim Boundary

This note records the successor decision after the
[Issue #2624 reuse-penalty paired diagnostic](../issue_2624_topology_reuse_penalty_gate.md). It does
not promote `topology_guided_hybrid_rule_v0_reuse_penalty`, add benchmark evidence, or claim a
planner improvement.

## Decision

```yaml
topology_successor_gate:
  rejected_for_promotion: topology_guided_hybrid_rule_v0_reuse_penalty
  decision: revise
  selected_successor_gate: primary_route_reselection_requires_route_progress
  successor_status: implementation_needed_before_smoke
  claim_boundary: diagnostic_only
```

The reuse-penalty candidate should not be rerun unchanged on the canonical double-bottleneck slice
and should not be reopened as a promotion candidate. It activated, but the measured behavior moved
the lane away from promotion:

- reuse penalty applied on 9 steps;
- selected-hypothesis counts shifted from `primary_route=56, non_primary_total=40` in the
  comparator to `primary_route=49, non_primary_total=47` in the candidate;
- non-primary topology-command influence stayed flat at 7 steps;
- topology-command steps fell from 33 to 31;
- maximum route-progress delta regressed by `-0.9514718767541765 m`;
- both runs ended `horizon_exhausted`.

The successor gate is therefore not another broad selector-design pass. The next hypothesis should
make primary-route reselection conditional on route-progress evidence: when eligible near-parity
alternatives remain available, repeated `primary_route` selections should stay penalized until the
current primary route shows enough recent route progress to justify rearming it.

## Why This Successor

The evidence chain rules out two tempting but weaker continuations:

- Selection-diversity-only work is already proven insufficient. Issue #2518 and #2530 showed the
  near-parity gate can move selection away from `primary_route`, but the canonical h160 slice still
  ended `horizon_exhausted`.
- Reuse-count-only penalization is too weak. Issue #2624 shows that reducing recent
  `primary_route` reuse can change selected labels, but it did not improve command influence,
  route progress, or terminal outcome.

That leaves progress-gated reselection as the smallest next discriminating hypothesis. It tests
whether the selector can distinguish "primary route is still justified because it is making
progress" from "primary route is being reused despite available alternatives and no progress
signal."

Uncertainty: medium. The conclusion is strong that the unchanged reuse-penalty candidate should
stop on this slice, but the successor mechanism remains a proposal until implemented and smoked.

## Successor Contract

Minimum implementation behavior for a future child issue:

- record recent per-hypothesis or selected-route progress over a short window;
- apply the existing reuse penalty only when eligible near-parity alternatives exist and recent
  primary-route progress is below a predeclared threshold;
- expose compact diagnostic fields such as `primary_route_recent_progress_m`,
  `primary_route_progress_gate_satisfied`, and `reuse_penalty_suppressed_by_progress`;
- preserve the existing route-selector counts, near-parity reasons, topology-command influence
  counts, route-progress deltas, hypothesis switch count, and terminal outcome fields;
- fail closed as diagnostic-only unless a separate benchmark issue predeclares denominator,
  comparator, scenario set, and promotion rule.

Follow-up implementation issue:
[#2704](https://github.com/ll7/robot_sf_ll7/issues/2704).

Outcome: Issue #2704 implemented the selected progress-gated successor and classified the paired
smoke as `revise`: the mechanism ran and suppressed six reuse penalties when progress was
sufficient, but it did not improve route progress, non-primary topology-command influence, or the
`horizon_exhausted` terminal outcome on the canonical h160 slice. Do not rerun the progress-gated
candidate unchanged on that slice as promotion evidence.

Suggested diagnostic gate after implementation:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate <progress-gated-topology-candidate> \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2660_progress_gated_successor
```

Comparator should remain `topology_guided_hybrid_rule_v0` on the same slice unless the child issue
predeclares a narrower comparator.

## Smoke Applicability

No short successor smoke was run in this PR because the selected successor gate does not currently
exist as a candidate under `configs/policy_search/candidates/`. Running the existing
`topology_guided_hybrid_rule_v0_reuse_penalty` candidate again would repeat the rejected gate rather
than exercise the selected successor.

## Validation

Use cheap docs validation for this synthesis:

```bash
rtk uv run python scripts/validation/check_research_lane_states.py
rtk uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
rtk git diff --check
```
