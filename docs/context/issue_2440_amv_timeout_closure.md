# Issue #2440 AMV Timeout Closure

Issue: [#2440](https://github.com/ll7/robot_sf_ll7/issues/2440)
Date: 2026-06-07
Status: current, diagnostic-only closure synthesis.

## Goal

Close the timeout-driver question for the AMV actuation-aware diagnostic without rerunning the
already-instrumented slice. The issue asked whether lower synthetic command clipping failed to
improve success because command feasibility improved without addressing the true timeout driver, or
because the actuation objective made progress too conservative.

## Decision

Decision outcome: `feasibility_improved_but_route_blocked`.

The durable #2404 and #2443 evidence is sufficient for Issue #2440's acceptance criteria:

- `actuation_aware_hybrid_rule_v0` reduced command clipping from `22` to `15` steps.
- The first 20-step window clip fraction fell from `0.75` to `0.40`.
- Both rows still failed as `timeout_low_progress`.
- Final route progress was effectively unchanged: the actuation-aware row finished `-0.0321 m`
  behind the baseline on route progress.
- Final distance to goal was effectively unchanged in the wrong direction: `+0.0321 m` farther
  from goal for the actuation-aware row.
- Mean average speed changed by only `+0.0047 m/s`.
- Yaw-rate saturation stayed `0.0` for both rows.

The best-supported timeout driver is therefore route/task progress remaining blocked after command
feasibility improved. Speed-cap binding and scoring conservativeness remain only partially
instrumented and are not primary explanations for this matched pair. Unrelated navigation deadlock
is not selected because the compact tracked artifacts do not include a deadlock or oscillation
detector.

## Decomposition Table

| Field | Value |
| --- | --- |
| `scenario_id` | `classic_cross_trap_high` |
| `seed` | `101` |
| `candidate` | `hybrid_rule_v3_fast_progress` baseline vs `actuation_aware_hybrid_rule_v0` intervention |
| `success` | `false` for both rows |
| `termination_reason` | `max_steps`; failure mode `timeout_low_progress` for both rows |
| `progress_over_time_available` | yes; #2443 preserves 20-step progress windows |
| `clipping_over_time_available` | yes; #2443 preserves 20-step clipping windows |
| `saturation_over_time_available` | yes; yaw-rate saturation windows are present and all zero |
| `command_speed_profile_available` | yes; #2404 records requested/applied linear-speed profile availability |
| `route_progress_blocked` | true |
| `speed_cap_binding` | partial, not primary |
| `scoring_too_conservative` | partial, not primary |
| `unrelated_navigation_deadlock` | not instrumented |
| `likely_timeout_driver` | `route_task_progress_blocked_after_feasibility_improvement` |

Machine-readable closure summary:
[evidence/issue_2440_amv_timeout_closure_2026-06-07/summary.json](evidence/issue_2440_amv_timeout_closure_2026-06-07/summary.json).

## Evidence Coverage

| Acceptance criterion | Evidence |
| --- | --- |
| Decomposition table generated or missing step-event blockers recorded | The table above re-expresses #2404/#2443 fields. Raw frame/event IDs remain blocked as `blocked_not_in_compact_artifact` in the #2443 summary. |
| Explain why lower clipping did or did not affect route progress and success | #2443 shows clipping improved but final route progress and goal distance stayed effectively unchanged, so success did not improve. |
| Recommend continue/revise/stop for actuation-aware scoring as a planner-improvement mechanism | Stop broad actuation-aware planner variants from this evidence. Continue only as a diagnostic/ranking-dimension question under #2446 or as route-progress geometry analysis. |
| Distinguish synthetic diagnostic evidence from proxy or calibrated AMV claims | This note is diagnostic-only and does not support calibrated AMV, hardware, safety, benchmark-strength, or paper-facing claims. |

## Follow-Up Boundary

Do not add another actuation-aware planner variant from this result. The useful follow-ups are:

- Issue #2446: decide whether actuation-feasibility metrics are useful as an evaluation/ranking
  dimension even when success is unchanged.
- A separate route-progress geometry or task-completion analysis if the research lane needs a new
  planner-improvement mechanism.
- A raw `simulation_trace_export.v1` trace only if publication-style frame/event panels become
  necessary.

## Claim Boundary

This is one matched synthetic AMV smoke pair. It is analysis-only evidence, not benchmark-strength
planner ranking and not calibrated AMV actuation evidence.

## Source Evidence

- [Issue #2259 AMV clipping versus success boundary](issue_2259_amv_clipping_success_boundary.md)
- [Issue #2404 AMV timeout decomposition decision](issue_2404_amv_timeout_decomposition_decision.md)
- [Issue #2443 AMV actuation trace review](issue_2443_amv_trace_review.md)
- [Issue #2443 compact summary](evidence/issue_2443_amv_trace_review_2026-06-07/summary.json)

## Validation

```bash
uv run python -m json.tool docs/context/evidence/issue_2440_amv_timeout_closure_2026-06-07/summary.json
uv run python scripts/validation/check_research_lane_states.py
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2440_amv_timeout_closure.md \
  --path docs/context/evidence/issue_2440_amv_timeout_closure_2026-06-07/summary.json \
  --path docs/context/catalog.yaml \
  --path docs/context/research_lane_states.md
git diff --check
```
