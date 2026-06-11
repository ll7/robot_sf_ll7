# Issue #2621 Topology Revision Hypothesis

Issue: [#2621](https://github.com/ll7/robot_sf_ll7/issues/2621)
Status: current, diagnostic-only decision.

## Claim Boundary

This note resolves the post-near-parity topology revision choice. It does not add runtime evidence,
benchmark evidence, planner promotion, or leaderboard evidence. It translates the latest topology
diagnostic chain into one next executable gate so the lane does not reopen multiple revision
families in parallel.

## Decision Output

```yaml
topology_revision_hypothesis:
  revision_name: primary_route_reuse_penalty_under_near_parity_alternatives
  selected_from_family: primary-route reuse penalty
  targeted_failure_mode: >
    The near-parity selector can move route selection and topology-command influence away from
    primary_route, but the corrective smoke still exhausted the horizon with weak route-progress
    change. The remaining failure is repeated primary-route reuse under eligible near-parity
    alternatives, not absence of alternatives or downstream command arbitration alone.
  expected_behavior_change: >
    When eligible near-parity alternatives remain available after recent primary_route selections,
    reduce primary_route selection score enough to preserve or increase non-primary
    topology-command influence while making route-progress and terminal-behavior effects explicit.
  required_trace_fields:
    - diagnostic_status
    - topology_status_counts
    - route_selector_selected_hypothesis_counts
    - selected_row_near_parity_gate_reasons
    - topology_command_influence_counts
    - topology_reuse_penalty
    - reuse_penalty_applied
    - reuse_penalty_reason
    - recent_primary_selection_count
    - eligible_near_parity_alternative_exists
    - route_progress_delta
    - hypothesis_switch_count
    - terminal_outcome
  diagnostic_gate: >
    Run the paired canonical full_matrix double-bottleneck diagnostic for
    topology_guided_hybrid_rule_v0_reuse_penalty against topology_guided_hybrid_rule_v0 on
    scenario classic_realworld_double_bottleneck_high, seed 111, horizon 160, max_hypotheses 3,
    and min_hypotheses 2. Preserve compact summaries only; raw output remains local.
  accept_if: >
    The diagnostic completes, required reuse-penalty and near-parity fields are present,
    non-primary topology-command influence is preserved or increased relative to the comparator,
    route-progress evidence improves without worse terminal behavior, and the comparator does not
    explain the change away.
  reject_if: >
    Non-primary topology-command influence collapses to zero, required fields fail closed, the
    mechanism changes only route labels without plausible route-progress or terminal-behavior
    improvement, or switching volatility increases without progress benefit.
  implementation_issue_needed: false
```

Decision outcome: `revision_ready_for_implementation_spike`.

The implementation spike already exists as
`topology_guided_hybrid_rule_v0_reuse_penalty` from
[#2540](issue_2540_topology_reuse_penalty_diagnostic.md). The remaining work is the paired
diagnostic gate, not another selector-design issue.

## Evidence Rationale

The selected revision is the only family with a direct evidence chain from the latest topology
work:

- [#2258](issue_2258_topology_primary_route_audit.md) showed topology alternatives were present,
  but topology-command wins inherited `primary_route`.
- [#2403](issue_2403_topology_selection_score_decision.md) classified the failure as
  `primary_route_overselected`, with scored alternatives rejected by lower selection score.
- [#2518](issue_2518_topology_near_parity_gate.md) showed the near-parity gate can produce 42
  non-primary route-selector selections and 7 non-primary topology-command influence selections.
- [#2530](issue_2530_topology_near_parity_corrective_smoke.md) kept the lane in `revise` because
  the same slice still ended `horizon_exhausted` with weak route-progress signal.
- [#2563](issue_2563_topology_corrective_revision.md) selected the primary-route reuse penalty and
  deferred strict stall triggers, minimum exposure gates, and hysteresis/switch-cost adjustments.
- [#2600](issue_2600_topology_revision_decision.md) confirmed that #2540 should be narrowed to the
  selected #2563 revision.
- [#2540](issue_2540_topology_reuse_penalty_diagnostic.md) implemented the diagnostic candidate and
  summary plumbing, but explicitly did not run the paired diagnostic.

The other candidate families remain deferred. A progress-stall trigger needs stronger
route-progress accounting before it should be the first corrective revision. Hysteresis or
switch-cost adjustment targets volatility, while the current failure is weak progress after the
selector becomes capable of non-primary influence. A minimum exposure gate could be useful later,
but it is broader than the reuse-penalty test now that a concrete diagnostic candidate exists.

## Next Diagnostic Command Shape

Reuse the #2540 launch packet, with the reuse-penalty candidate and comparator kept explicit:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0_reuse_penalty \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2621_reuse_penalty
```

Comparator:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir output/diagnostics/issue2621_baseline_comparator
```

## Validation

This is a decision note only. Validate with cheap docs checks:

```bash
uv run python scripts/validation/check_research_lane_states.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```
