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
| Static recentering on held-out transfer rows | `inactive` on the unsolved held-out row. The activation trace recorded zero recenter activations and unchanged terminal outcomes. | Only reopen if a future slice predeclares states where static recentering should activate; otherwise prefer route-progress mechanisms. | `stop` for this held-out slice. | [#2306](https://github.com/ll7/robot_sf_ll7/issues/2306), [issue_2306_static_recenter_activation_trace.md](issue_2306_static_recenter_activation_trace.md), [issue_2221_static_recenter_transfer.md](issue_2221_static_recenter_transfer.md) |
| Topology guidance / primary-route scoring | `revise` after near-parity gating reached non-primary selection and local command influence, but the #2530 corrective smoke still ended `horizon_exhausted` without benchmark-strength improvement. The #2522 why-first report preserves this as `topology_signal_without_route_progress`. | Test the #2563 `primary_route_reuse_penalty_under_near_parity_alternatives` hypothesis in a paired diagnostic before any benchmark or leaderboard claim. | `revise`; do not promote topology mitigation from the current diagnostic. | [#2530](https://github.com/ll7/robot_sf_ll7/issues/2530), [issue_2530_topology_near_parity_corrective_smoke.md](issue_2530_topology_near_parity_corrective_smoke.md), [#2563](https://github.com/ll7/robot_sf_ll7/issues/2563), [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md), [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [topology report](evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md) |
| AMV actuation-aware scoring | `active-but-irrelevant` to completion on the tiny AMV timeout trace. Command clipping improved, yaw saturation did not explain the timeout, and both candidates timed out with similar route progress. The #2522 why-first report preserves this as `route_task_progress_blocked_after_feasibility_improvement`. | Investigate route-progress geometry or horizon/task completion blockers separately before adding another AMV actuation scorer. | `revise` away from actuation scoring as the immediate blocker. | [#2308](https://github.com/ll7/robot_sf_ll7/issues/2308), [issue_2308_amv_timeout_trace_analysis.md](issue_2308_amv_timeout_trace_analysis.md), [issue_2268_amv_timeout_decomposition.md](issue_2268_amv_timeout_decomposition.md), [issue_2259_amv_clipping_success_boundary.md](issue_2259_amv_clipping_success_boundary.md), [issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md), [AMV report](evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md) |
| AMMV Social Force renderable trace review | `blocked-by-missing-evidence`: the benchmark path regenerates aggregate episode rows, not step-event frames with AMMV force/intrusion metadata, so no durable `simulation_trace_export.v1` evidence exists for the target AMV case. | Choose and implement either a benchmark recorder path that emits compatible step frames with AMMV metadata or a narrow direct-probe trace exporter with explicit limitations. | `blocked`; do not cite local probe output as durable trace evidence. | [#2309](https://github.com/ll7/robot_sf_ll7/issues/2309), [issue_2309_amv_trace_export_blocker.md](issue_2309_amv_trace_export_blocker.md), [issue_2236_trace_mechanism_evidence_rubric.md](issue_2236_trace_mechanism_evidence_rubric.md) |
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
