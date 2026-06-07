# Issue #2453 Planner Mechanism Cards

Issue: [#2453](https://github.com/ll7/robot_sf_ll7/issues/2453)
Parent issue: [#2451](https://github.com/ll7/robot_sf_ll7/issues/2451)
Status: docs/evidence organization only.

## Purpose

These cards summarize active planner research candidates by mechanism, activation signal, known
evidence, and claim boundary. They do not add benchmark evidence, promote any planner, or make
paper-facing claims. Positive evidence below means "supports keeping the mechanism in the research
queue"; negative evidence means "bounds or falsifies the current mechanism shape."

## Template

```yaml
planner_id:
mechanism_claim:
expected_failure_mode_addressed:
activation_signal:
required_diagnostics:
known_positive_evidence:
known_negative_evidence:
transfer_status:
claim_boundary:
canonical_smoke_command:
```

## Mechanism Cards

### `issue_2170_static_recenter_only`

```yaml
planner_id: issue_2170_static_recenter_only
mechanism_claim: >
  Static-recenter scoring can recover heading or local command choice when a static obstacle or
  local-minimum state stalls the fast-progress hybrid-rule baseline away from nearby pedestrians.
expected_failure_mode_addressed: static-obstacle local deadlock or low-progress local minimum.
activation_signal: >
  Activation count, first activation step, selected command source, selected
  terms["static_recenter"], and progress delta after activation.
required_diagnostics:
  - per-step static-recenter activation fields
  - command-source attribution
  - route-progress delta before and after activation
  - collision and near-miss parity versus the matched baseline
known_positive_evidence:
  - docs/context/issue_2180_one_factor_h500.md reports static_recenter_only_minus_base success
    +0.056, collision delta 0.000, near-miss delta 0.000, and average-speed delta +0.075.
  - docs/context/issue_2182_component_effect_synthesis.md reports the one-factor diagnostic row
    improved success by +0.056 and average speed by +0.075 without collision or near-miss penalty.
  - docs/context/issue_2261_static_recenter_slice_local.md preserves that result as a local
    diagnostic signal worth keeping for predeclared static-obstacle recovery slices.
known_negative_evidence:
  - docs/context/issue_2221_static_recenter_transfer.md found no held-out pilot lift versus
    hybrid_rule_v3_fast_progress.
  - docs/context/issue_2306_static_recenter_activation_trace.md and
    docs/context/issue_2402_static_recenter_activation_decision.md record zero activations on the
    inspected held-out rows, so the unsolved held-out row is an inactive-mechanism negative.
  - docs/context/issue_2438_static_recenter_activation_closure.md closes the current held-out route
    as `mechanism_inactive` and recommends stopping unless a future activation-targeted slice is
    predeclared.
transfer_status: >
  slice_local for the h500 diagnostic component; stop the current held-out transfer route because
  static recentering was inactive on the unsolved held-out row.
claim_boundary: diagnostic-only mechanism card; not planner-ranking, transfer, or benchmark evidence.
canonical_smoke_command: >
  uv run python scripts/validation/run_policy_search_candidate.py --candidate
  issue_2170_static_recenter_only --stage smoke
```

### `topology_guided_hybrid_rule_v0`

```yaml
planner_id: topology_guided_hybrid_rule_v0
mechanism_claim: >
  A topology-hypothesis route selector can expose alternate local homotopies and inject a bounded
  selected-hypothesis command when the primary route is a poor local guide.
expected_failure_mode_addressed: primary-route local topology mismatch or route-guide deadlock.
activation_signal: >
  Alternative hypothesis count, selected hypothesis, score margin to primary route, rejection
  reason, switch opportunity count, and topology-command influence source.
required_diagnostics:
  - per-hypothesis length and static-clearance score components
  - selected-hypothesis and rejection-reason fields
  - non-primary route-selector selections with non-tie margins
  - topology-command influence using a non-primary selected hypothesis
known_positive_evidence:
  - docs/context/issue_2258_topology_primary_route_audit.md shows alternatives were generated often
    enough to reject a pure "no alternatives" explanation.
  - docs/context/issue_2282_topology_selection_instrumentation.md added score components, rank,
    margin, and rejection reasons to make the mechanism diagnosable.
  - docs/context/issue_2393_topology_selection_preflight.md defines a near-parity diversity-gate
    revision and explicit accept/revise/reject fields.
known_negative_evidence:
  - docs/context/issue_2307_topology_score_diagnostic.md classifies the current row as
    scoring_overselects_primary: 98 scored alternatives, 97 primary-route selections, one numerical
    tie, and no corrective non-primary topology-command influence.
  - docs/context/issue_2403_topology_selection_score_decision.md records
    primary_route_overselected and keeps the result analysis-only.
transfer_status: diagnostic-only; revise upstream selection before any transfer or benchmark claim.
claim_boundary: not mitigation evidence until a targeted diagnostic shows real non-primary selection.
canonical_smoke_command: >
  uv run python scripts/validation/run_policy_search_candidate.py --candidate
  topology_guided_hybrid_rule_v0 --stage smoke
```

### `actuation_aware_hybrid_rule_v0`

```yaml
planner_id: actuation_aware_hybrid_rule_v0
mechanism_claim: >
  Penalizing commands that would be clipped by the synthetic AMV actuation envelope can reduce
  command-feasibility stress while preserving the existing hybrid-rule safety filters.
expected_failure_mode_addressed: synthetic AMV command clipping or speed-envelope saturation.
activation_signal: >
  Command clip steps, command clip fraction, windowed clipping, requested/applied command gap,
  yaw-rate saturation, final route progress, and timeout driver.
required_diagnostics:
  - matched baseline/intervention route-progress trace
  - command clipping and command gap over time
  - final distance-to-goal and timeout classification
  - explicit calibrated-hardware evidence boundary
known_positive_evidence:
  - docs/context/issue_2259_amv_clipping_success_boundary.md and
    docs/context/issue_2268_amv_timeout_decomposition.md show mean command clip fraction fell from
    0.2750 to 0.1875 on the matched synthetic smoke.
  - docs/context/issue_2308_amv_timeout_trace_analysis.md shows total clip steps fell from 22 to 15
    and first-window clip fraction fell from 0.75 to 0.40.
  - docs/context/issue_2443_amv_trace_review.md preserves the compact progress-versus-clipping
    review and classifies the result as feasibility_improved_but_route_blocked.
known_negative_evidence:
  - The same AMV notes show both rows remained zero-success timeout_low_progress cases.
  - docs/context/issue_2404_amv_timeout_decomposition_decision.md records final route progress as
    effectively unchanged and keeps route/task progress as the best-supported blocker.
transfer_status: synthetic diagnostic only; no calibrated AMV, hardware, benchmark, or paper claim.
claim_boundary: useful feasibility signal, not navigation-success or planner-superiority evidence.
canonical_smoke_command: >
  uv run python scripts/validation/run_policy_search_candidate.py --candidate
  actuation_aware_hybrid_rule_v0 --stage amv_actuation_smoke
```

### `orca_residual_guarded_ppo_v0`

```yaml
planner_id: orca_residual_guarded_ppo_v0
mechanism_claim: >
  A learned residual can modify the nominal ORCA command inside a hard guarded runtime contract, so
  the learned component may improve progress without bypassing reciprocal-avoidance safety checks.
expected_failure_mode_addressed: low-progress learned or guarded policy behavior under a safe prior.
activation_signal: >
  Residual active rate, raw and bounded residual magnitude, residual clipping, guard veto or shield
  override, fallback/degraded status, route progress, and checkpoint lineage.
required_diagnostics:
  - durable training and checkpoint lineage
  - raw model action, bounded residual, and post-guard command
  - guard override, hard-constraint violation, fallback, and degraded rates
  - smoke-row success/collision/near-miss/failure-mode summary
known_positive_evidence:
  - docs/context/policy_search/reports/2026-06-05_orca_residual_guarded_ppo_v0_smoke.md reports one
    valid smoke row, so the runtime surface can execute.
  - docs/context/issue_2311_orca_residual_lane_decision.md records no observed guard saturation:
    shield_decision_count 80, shield_override_rate 0.0, and hard-constraint violation rate 0.0.
known_negative_evidence:
  - The same smoke evidence records success_rate 0.0 and timeout_low_progress 1.
  - docs/context/issue_2408_orca_residual_low_progress_analysis.md classifies the primary observed
    failure as learned_component_no_route_progress with confidence 0.85 and recommends revising the
    residual objective before another smoke.
  - docs/context/policy_search/learned_policy_registry.md marks checkpoint availability pending and
    the benchmark status as smoke_only.
transfer_status: smoke-only runtime evidence; no trained-residual success or benchmark support.
claim_boundary: launch/runtime surface only until the residual objective and lineage are revised.
canonical_smoke_command: >
  uv run python scripts/validation/run_policy_search_candidate.py --candidate
  orca_residual_guarded_ppo_v0 --stage smoke
```

### `adaptive_proxemic_selector_v1`

```yaml
planner_id: adaptive_proxemic_selector_v1
mechanism_claim: >
  A neutral-default proxemic selector can choose conservative, neutral, or open fixed profiles from
  local context, reserving the open profile for sparse low-progress recovery while using
  conservative behavior near humans or in constrained passages.
expected_failure_mode_addressed: proxemic over-conservatism versus human-clearance risk tradeoff.
activation_signal: >
  Selected profile, source candidate, local pedestrian distance/count, constrained-passage width,
  low-progress context, comfort exposure, near-miss rate, and route progress.
required_diagnostics:
  - selected_profile_counts and last_selection diagnostics
  - comfort and near-miss tradeoff metrics
  - matched progress and clearance comparison against fixed proxemic profiles
  - diagnostic-only claim-boundary check
known_positive_evidence:
  - tests/planner/test_adaptive_proxemic_selector.py covers v1 routing to neutral in clear scenes,
    conservative in constrained passages, and open only for sparse low-progress context.
  - tests/validation/test_run_policy_search_candidate.py checks that the v1 candidate remains
    diagnostic-only and resolves configs/policy_search/candidates/adaptive_proxemic_selector_v1.yaml.
  - docs/context/policy_search/reports/2026-05-31_adaptive_proxemic_selector_v1_smoke.md reports
    a passing single-row smoke on planner_sanity_simple with success 1.0000, collision 0.0000, and
    near miss 0.0000.
known_negative_evidence:
  - docs/context/policy_search/reports/2026-05-31_adaptive_proxemic_selector_v1_nominal_sanity.md
    classifies the nominal_sanity stage as revise with success 0.2222, near miss 0.2222, and 11
    timeout_low_progress failures.
  - No durable trace note currently shows comfort, near-miss, or success improvement beyond
    diagnostic arithmetic context.
  - The candidate registry marks the row experimental_spike with claim_scope diagnostic_only.
transfer_status: diagnostic configuration and unit-routing evidence only.
claim_boundary: not evidence of proxemic improvement until comfort and near-miss tradeoffs are run.
canonical_smoke_command: >
  uv run python scripts/validation/run_policy_search_candidate.py --candidate
  adaptive_proxemic_selector_v1 --stage smoke
```

## Research Direction Summary

| Candidate | Current strongest signal | Next smallest useful proof |
| --- | --- | --- |
| `issue_2170_static_recenter_only` | Local h500 diagnostic positive with #2438 held-out `mechanism_inactive` closure. | Stop the current held-out transfer route; only reopen via a predeclared static-obstacle slice where activation should occur, preserving activation and command-source fields. |
| `topology_guided_hybrid_rule_v0` | Alternatives exist, but scoring overselects `primary_route`. | Implement or test the near-parity diversity gate and require real non-primary selection, not numerical ties. |
| `actuation_aware_hybrid_rule_v0` | Synthetic command feasibility improves while route progress remains blocked. | Analyze route-progress geometry or task-completion blockers before more actuation scoring variants. |
| `orca_residual_guarded_ppo_v0` | Runtime surface executes without guard saturation, but low progress remains. | Revise residual objective/diagnostics and require durable raw/bounded residual magnitude fields. |
| `adaptive_proxemic_selector_v1` | Smoke wiring passed, but nominal-sanity stayed revise and comfort tradeoffs are unproven. | Run a tiny matched comfort/near-miss/progress trace before claiming proxemic improvement. |

## Validation

Commands in these cards are canonical smoke surfaces, not runs performed by this documentation
change. This issue is a docs/evidence organization pass, so validation should prove links, command
entry points, and referenced files exist rather than spend benchmark time.
