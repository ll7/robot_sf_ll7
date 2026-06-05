# Issue #2308 AMV Timeout Trace Analysis

Issue: [#2308](https://github.com/ll7/robot_sf_ll7/issues/2308)
Parent issue: [#2259](https://github.com/ll7/robot_sf_ll7/issues/2259)
Predecessor: [issue_2268_amv_timeout_decomposition.md](issue_2268_amv_timeout_decomposition.md)
Status: diagnostic-only trace rerun; actuation feasibility improved, but route/task progress
remained the timeout driver.

## Goal

Decompose the matched AMV actuation-smoke timeout from Issue #2259/#2268 with per-step route
progress and command-feasibility fields. The question is whether
`actuation_aware_hybrid_rule_v0` failed to improve success because route progress stalled, commands
became too conservative, or another navigation failure dominated.

This note uses a tiny rerun of the same `amv_actuation_smoke` row:

- Scenario: `classic_cross_trap_high`
- Seed: `101`
- Horizon: `80`
- Baseline: `hybrid_rule_v3_fast_progress`
- Intervention: `actuation_aware_hybrid_rule_v0`
- Synthetic actuation profile: `amv-actuation-stress-v0`

The result is not planner-ranking, benchmark-strength, calibrated AMV, hardware, safety, or
paper-facing evidence.

## Required Fields

| Field | Observation |
| --- | --- |
| `progress_over_time` | Both candidates made positive progress but timed out still about `3.6 m` from the goal. |
| `clipping_over_time` | Actuation-aware scoring reduced early clipping and total clip steps. |
| `saturation_over_time` | Yaw-rate saturation stayed `0.0` for both candidates in every 20-step window. |
| `command_speed_profile` | Actuation-aware scoring requested and applied lower linear speeds on average, but final route progress was effectively unchanged. |
| `timeout_driver` | Route/task progress remained insufficient at horizon. |
| `whether_actuation_aware_scoring_slowed_progress` | No success-relevant slowdown was observed; final route progress was `0.032 m` lower for actuation-aware, while last-10-step progress was `0.121 m` higher. |

## Matched Row Summary

| Candidate | Success | Failure | Clip Steps | Yaw Sat Steps | Avg Speed | Final Progress | Final Distance |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid_rule_v3_fast_progress` | `0` | `timeout_low_progress` | `22` | `0` | `1.6623` | `12.9040 m` | `3.6079 m` |
| `actuation_aware_hybrid_rule_v0` | `0` | `timeout_low_progress` | `15` | `0` | `1.6670` | `12.8719 m` | `3.6400 m` |

Delta, actuation-aware minus baseline:

- Command clip fraction: `-0.0875`
- Command clip steps: `-7`
- Mean average speed: `+0.0047 m/s`
- Final route progress: `-0.0321 m`
- Final distance to goal: `+0.0321 m`

## Windowed Trace

| Candidate | Steps | Progress Delta | Clip Fraction | Yaw Sat Fraction | Mean Requested v | Mean Applied v | Mean `abs(v_req-v_app)` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hybrid_rule_v3_fast_progress` | `0-19` | `2.8884` | `0.75` | `0.00` | `2.6858` | `1.8350` | `0.8508` |
| `hybrid_rule_v3_fast_progress` | `20-39` | `3.6421` | `0.10` | `0.00` | `2.9792` | `2.9675` | `0.0117` |
| `hybrid_rule_v3_fast_progress` | `40-59` | `3.7380` | `0.15` | `0.00` | `2.6488` | `2.6687` | `0.0199` |
| `hybrid_rule_v3_fast_progress` | `60-79` | `2.0724` | `0.10` | `0.00` | `1.1029` | `1.1157` | `0.0127` |
| `actuation_aware_hybrid_rule_v0` | `0-19` | `2.9276` | `0.40` | `0.00` | `1.6071` | `1.5500` | `0.0571` |
| `actuation_aware_hybrid_rule_v0` | `20-39` | `3.2348` | `0.20` | `0.00` | `1.7800` | `1.7725` | `0.0158` |
| `actuation_aware_hybrid_rule_v0` | `40-59` | `3.6002` | `0.00` | `0.00` | `2.0000` | `2.0000` | `0.0000` |
| `actuation_aware_hybrid_rule_v0` | `60-79` | `2.5209` | `0.15` | `0.00` | `1.3368` | `1.3435` | `0.0067` |

## Classification

Classification: `feasibility_improved_but_route_blocked`.

The trace supports the earlier #2268 aggregate result, but with stronger mechanism evidence:

- command feasibility improved, especially in the first 20 steps;
- yaw saturation did not explain either timeout;
- after the initial window, requested/applied command gaps were small;
- both candidates retained positive but insufficient route progress and timed out short of the goal;
- the actuation-aware scorer did not convert feasibility gains into task completion.

This does not support `scoring_too_conservative` as the primary explanation. The actuation-aware
candidate used lower average applied linear velocity than the baseline in this trace, but its final
progress and late progress were close enough that the success-relevant blocker remains route/task
completion, not persistent command infeasibility.

## Recommendation

Keep `actuation_aware_hybrid_rule_v0` diagnostic-only. Do not broaden AMV actuation claims from
this slice. The next useful work should investigate the route-progress geometry or horizon/task
completion blocker separately, rather than adding another AMV actuation scorer.

## Evidence

- Compact summary:
  [evidence/issue_2308_amv_timeout_trace_2026-06-05/summary.json](evidence/issue_2308_amv_timeout_trace_2026-06-05/summary.json)
- Evidence README:
  [evidence/issue_2308_amv_timeout_trace_2026-06-05/README.md](evidence/issue_2308_amv_timeout_trace_2026-06-05/README.md)

## Validation

Validation commands:

```bash
scripts/dev/run_worktree_shared_venv.sh -- pytest tests/benchmark/test_map_runner_utils.py::test_run_map_episode_records_synthetic_actuation_metrics -q
python -m json.tool docs/context/evidence/issue_2308_amv_timeout_trace_2026-06-05/summary.json
```
