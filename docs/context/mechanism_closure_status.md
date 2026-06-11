# Mechanism Closure Status

Issue: [#2387](https://github.com/ll7/robot_sf_ll7/issues/2387)
Related synthesis thread: [#2389](https://github.com/ll7/robot_sf_ll7/issues/2389)
Thread scaffold: [issue_2389_mechanism_aware_evaluation_thread.md](issue_2389_mechanism_aware_evaluation_thread.md)
Status: current research-status surface; not benchmark evidence.

## Purpose

This note keeps one compact view of current mechanism-closure state for AMV-relevant local
navigation work. It summarizes existing tracked notes and issue contracts so future research work
can choose the next empirical action without reading every issue thread.

This is a status table only. It does not promote diagnostic, blocked, smoke, adapter, or local-only
evidence into a benchmark result or paper-facing claim.

## Status Table

| Mechanism lane | Current state | Next evidence | Decision state | Source links |
| --- | --- | --- | --- | --- |
| Static recentering on held-out transfer and static-deadlock rows | Held-out transfer remains `inactive` on the unsolved row. The #2588 static-recenter-only and #2590 escape-recenter-pair h120 traces each found one active trace-change row (`classic_bottleneck_low`, seed `113`) without terminal rescue. Issue #2592 reran that active row at h500 and both recenter pairings converted the baseline 500-step local-minimum failure into success at step `122`. Issue #2594 repeated the same two pairings across the predeclared 3-scenario x 3-seed h500 slice; all 18 pair-rows completed with required fields, the same unsolved active row was rescued in both pairings, and the other 16 pair-rows were already solved by both candidates. Issue #2596 classifies this as useful controlled-trace evidence whose promotion is blocked by scope. | Do not promote static recentering as planner improvement from one unsolved active row. If the lane continues, the next empirical issue should predeclare a harder unsolved-row expansion with the same trace-field contract and a stop rule that treats no new unsolved active rescue rows as `synthesize_stop`. | `stop` for held-out transfer; `controlled_trace_negative_mixed` for #2588/#2590; `delayed_rescue_candidate` for #2592 active-row h500; `broader_delayed_rescue_supported` for #2594 broader h500 controlled trace; `promotion_blocked_by_scope` for #2596 synthesis. | [#2306](https://github.com/ll7/robot_sf_ll7/issues/2306), [issue_2306_static_recenter_activation_trace.md](issue_2306_static_recenter_activation_trace.md), [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md), [issue_2438_static_recenter_activation_closure.md](issue_2438_static_recenter_activation_closure.md), [issue_2566_static_recenter_inactive_propagation.md](issue_2566_static_recenter_inactive_propagation.md), [#2588](https://github.com/ll7/robot_sf_ll7/issues/2588), [issue_2588_static_deadlock_controlled_trace.md](issue_2588_static_deadlock_controlled_trace.md), [#2590](https://github.com/ll7/robot_sf_ll7/issues/2590), [issue_2590_escape_recenter_static_deadlock_controlled_trace.md](issue_2590_escape_recenter_static_deadlock_controlled_trace.md), [#2592](https://github.com/ll7/robot_sf_ll7/issues/2592), [issue_2592_static_deadlock_active_row_h500.md](issue_2592_static_deadlock_active_row_h500.md), [#2594](https://github.com/ll7/robot_sf_ll7/issues/2594), [issue_2594_static_deadlock_broader_h500.md](issue_2594_static_deadlock_broader_h500.md), [#2596](https://github.com/ll7/robot_sf_ll7/issues/2596), [issue_2596_static_deadlock_recenter_claim_boundary.md](issue_2596_static_deadlock_recenter_claim_boundary.md) |
| Topology guidance / primary-route scoring | `revise` after near-parity gating reached non-primary selection and local command influence, but the #2530 corrective smoke still ended `horizon_exhausted` without benchmark-strength improvement. The #2522 why-first report preserves this as `topology_signal_without_route_progress`. | Test the #2563 `primary_route_reuse_penalty_under_near_parity_alternatives` hypothesis in a paired diagnostic before any benchmark or leaderboard claim. | `revise`; do not promote topology mitigation from the current diagnostic. | [#2530](https://github.com/ll7/robot_sf_ll7/issues/2530), [issue_2530_topology_near_parity_corrective_smoke.md](issue_2530_topology_near_parity_corrective_smoke.md), [#2563](https://github.com/ll7/robot_sf_ll7/issues/2563), [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md), [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [topology report](evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md) |
| AMV actuation-aware scoring | `active-but-irrelevant` to completion on the tiny AMV timeout trace. Command clipping improved, yaw saturation did not explain the timeout, and both candidates timed out with similar route progress. The #2522 why-first report preserves this as `route_task_progress_blocked_after_feasibility_improvement`. The #2531 decision keeps this row summary-timeline-only because #2443/#2522 have no raw `simulation_trace_export.v1` frame/event IDs for the matched actuation-aware slice. | Investigate route-progress geometry or horizon/task completion blockers separately before adding another AMV actuation scorer. If trace-level review is needed, open a dedicated exporter issue for the same `classic_cross_trap_high` seed `101` row rather than reusing AMMV Social Force traces. | `revise` away from actuation scoring as the immediate blocker; summary-timeline-only for trace explanation. | [#2308](https://github.com/ll7/robot_sf_ll7/issues/2308), [issue_2308_amv_timeout_trace_analysis.md](issue_2308_amv_timeout_trace_analysis.md), [issue_2268_amv_timeout_decomposition.md](issue_2268_amv_timeout_decomposition.md), [issue_2259_amv_clipping_success_boundary.md](issue_2259_amv_clipping_success_boundary.md), [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [AMV report](evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md), [#2531](https://github.com/ll7/robot_sf_ll7/issues/2531), [issue_2531_amv_trace_boundary.md](issue_2531_amv_trace_boundary.md) |
| AMMV Social Force renderable trace review | `diagnostic-trace-available`: #2405 proved one selected default/AMMV Social Force row can export as loader-valid `simulation_trace_export.v1`, and #2428 promoted the matching trace-panel bundle with 20-frame traces preserving `ammv` planner metadata. | Treat as diagnostic AMMV/default Social Force trace-panel evidence only. Do not use it to upgrade AMV actuation-aware hybrid-rule #2443/#2531 claims; future work still needs richer activation instrumentation or a deliberately more sensitive family. | `diagnostic_only`; not benchmark or paper-facing evidence. | [#2309](https://github.com/ll7/robot_sf_ll7/issues/2309), [issue_2309_amv_trace_export_blocker.md](issue_2309_amv_trace_export_blocker.md), [issue_2405_amv_step_export_decision.md](issue_2405_amv_step_export_decision.md), [issue_2428_mechanism_trace_panels.md](issue_2428_mechanism_trace_panels.md), [#2531](https://github.com/ll7/robot_sf_ll7/issues/2531), [issue_2531_amv_trace_boundary.md](issue_2531_amv_trace_boundary.md) |
| ORCA-residual behavior cloning | `slice-local` negative smoke signal after adapter and JSONL blockers were repaired: the smoke row ran, avoided collisions/near misses, but timed out with low progress and should not escalate to nominal unchanged. | Revise the residual objective or candidate contract so route progress is explicit under the guarded ORCA runtime contract, then rerun the bounded smoke gate. | `revise`; no nominal escalation until revised smoke passes. | [#2311](https://github.com/ll7/robot_sf_ll7/issues/2311), [issue_2311_orca_residual_lane_decision.md](issue_2311_orca_residual_lane_decision.md), [issue_2272_orca_residual_launch_packet_status.md](issue_2272_orca_residual_launch_packet_status.md) |
| Learned-risk model v1 | `blocked-by-missing-evidence`: the launch packet validates shape and fixture fields, but the durable training trace manifest and concrete baseline artifact URI are missing. | Materialize or fail-close durable trace inputs, baseline artifact URI, label availability, checksums, and training readiness before learned-risk training. | `blocked`; not training or planner evidence yet. | [#2312](https://github.com/ll7/robot_sf_ll7/issues/2312), [issue_2273_learned_risk_trace_preflight.md](issue_2273_learned_risk_trace_preflight.md), [issue_1395_learned_risk_launch_packet.md](issue_1395_learned_risk_launch_packet.md) |
| Local learned-policy baseline artifacts | `blocked-by-missing-evidence`: seven historical local-only model configs are explicitly unavailable; scanners now fail closed instead of treating them as promotion-ready. | Recover a durable checkpoint source with checksum and registry/artifact pointer, or retire/rewrite the configs for any future benchmark-facing use. | `stop` for the same local-only rows until recovered or retired. | [#2313](https://github.com/ll7/robot_sf_ll7/issues/2313), [issue_2313_local_baseline_quarantine.md](issue_2313_local_baseline_quarantine.md), [issue_2277_local_artifact_classification.md](issue_2277_local_artifact_classification.md) |

## Cross-Cutting Interpretation

Observed evidence supports a conservative research direction:

- Mechanisms that do not activate or do not influence command correction should stop or be revised
  before more benchmark runs.
- Mechanisms that improve a local sub-signal, such as command feasibility, still need route/task
  progress proof before they become planner-improvement evidence.
- Why-first reports are useful as interpretation aids, but they inherit the compact evidence limits
  of their inputs and do not change a lane's decision state by themselves.
- Learned components remain useful research directions, but current durable evidence mostly
  supports launch-packet, smoke, or blocked-input status rather than comparative synthesis.
- Mechanism-aware ranking remains diagnostic: it can reveal why aggregate success/collision ranks
  are incomplete, but it is not a replacement leaderboard.

## Next Empirical Actions

1. Use [#2389](https://github.com/ll7/robot_sf_ll7/issues/2389) to connect this closure surface to
   a paper/dissertation candidate thread, with the claim boundary still marked not paper-grade.
2. Prioritize one revision or proof issue at a time:
   topology primary-route reuse-penalty diagnostic, ORCA-residual objective revision, learned-risk trace
   materialization, or AMV trace-recorder implementation.
3. Do not add a new planner family merely to avoid a blocked row; close or revise the named
   mechanism first.

## Claim Boundary

This note is research-routing synthesis. It is not benchmark-strength evidence, not a manuscript
claim, and not a claim that any mechanism improves AMV navigation. Rows marked blocked, inactive,
slice-local, or revise need their own executable proof before downstream promotion.

## Validation

Planned validation for this docs-only status surface:

```bash
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
