# Issue #2767 Benchmark Table Candidates

Generated at: 2026-06-14T11:17:20.758447+00:00
Status: draft_only_not_for_manuscript_use_without_verification

This bundle contains conservative draft benchmark-results table candidates synthesized from
tracked claim/evidence inputs. It is not a manuscript draft and does not promote diagnostic,
stale, unavailable, fallback, degraded, proxy-only, or missing-denominator evidence.

## 1. Metric Summary (AMV Primary Protocol)
| Planner | Nominal success | Stress success | Nominal collisions | Stress collisions | Nominal SNQI | Stress SNQI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `goal` | 0.2500 | 0.0000 | 0.3333 | 0.2500 | -0.0967 | -0.1752 |
| `orca` | 0.2500 | 0.1667 | 0.0833 | 0.0764 | -0.2999 | -0.2466 |
| `social_force` | 0.0000 | 0.0000 | 0.0000 | 0.2500 | -1.0435 | -0.8591 |

Allowed wording: "Baseline AMV performance remains limited; stress success remains low and caveated."

Caveat: AMV coverage remains incomplete and SNQI contract status is warn/fail in the source map.

## 2. Topology Diagnostic Summary
| Field | Value |
|---|---|
| Claim | Topology-guided hybrid rule selection can diversify route selection on the canonical h160 double-bottleneck slice, but repeated selector variants have not improved topology-command influence, route progress, or terminal outcome. |
| Artifact status | current |
| Evidence tier | diagnostic |
| Allowed wording | Near-parity topology selection can diversify route labels, but current evidence still routes the mechanism to revise without improving non-primary command influence, route progress, or terminal outcome. |
| Caveat | Do not claim topology guidance improves success, transfer, or leaderboard performance. The lane is stop for same-family selector reruns on the canonical slice. |

Allowed wording: "Near-parity topology selection can diversify route labels, but current evidence still routes the mechanism to revise without improving non-primary command influence, route progress, or terminal outcome."

Caveat: Do not claim topology guidance improves success, transfer, or leaderboard performance. The lane is stop for same-family selector reruns on the canonical slice.

## 3. Signalized-Crossing Metric Summary
| Episode ID | Row type | Observable | Denominator |
|---|---|---|---|
| issue_2799_red_required_stop_observable--2799--8b0ca6d7b131d8d2 | red_required_stop | True | 1 |
| issue_2799_green_proceed_observable--2799--619198efe8ebb9e3 | green_proceed | True | 1 |
| issue_2799_unavailable_no_claim--2799--d65f4fe883ae734f | unavailable_no_claim | False | 0 (excluded: signal_state_metadata_absent) |
| issue_2799_proxy_only_denominator_excluded--2799--e4eb30e7d07f0c90 | proxy_only_denominator_excluded | False | 0 (excluded: signal_state_not_benchmark_evidence) |

Allowed wording: "Simulator-backed signalized-crossing denominator plumbing exists for explicit
planner-observable rows."

Caveat: Diagnostic only; does not prove traffic-signal realism, crossing-legality compliance, or
planner-ranking performance.

## 4. Prediction Baseline Summary
| Variant | Success rate | Mean min distance | Hard success | Best planner-grid row | Global mean min distance |
| --- | ---: | ---: | ---: | --- | ---: |
| Baseline predictive | 0.1304 | 2.1931 | 0.0000 | `risk_aware_adaptive` | 2.1931 |
| Obstacle-feature predictive | 0.1014 | 2.2105 | 0.0000 | `baseline_like` | 2.2081 |

Allowed wording: "The prediction interface and comparison surfaces exist, but current closed-loop
performance deltas are unproven or negative."

Caveat: Do not claim prediction-quality or planner-improvement evidence from contract or negative
same-seed rows.

## 5. Observation-Noise Diagnostic Summary
| Condition | Classification | Rationale |
|---|---|---|
| noop | diagnostic_only | No-perturbation baseline. Provides reference robot-pedestrian trajectory and action selection without observation noise. |
| low_noise | diagnostic_only | Perturbation produced mixed effects. Classified as diagnostic-only pending broader seed/scenario evidence. |
| medium_noise | diagnostic_only | Perturbation produced mixed effects. Classified as diagnostic-only pending broader seed/scenario evidence. |
| missed_detection_only | scenario_too_weak | Pedestrian fully missed after fixture visibility begins; no observation signal reaches the policy. Cannot test policy robustness. |
| occlusion_only | scenario_too_weak | Pedestrian position/velocity zeroed by occlusion after fixture visibility begins; no observation signal reaches the policy. |
| delay_only | robustness_evidence | Pedestrian observed at step 7 (2 steps after first-visible). Perturbation delayed/masked the observation and may have affected policy response timing. |
| combined | scenario_too_weak | Pedestrian position/velocity zeroed by occlusion after fixture visibility begins; no observation signal reaches the policy. |

Allowed wording: "Observation-noise diagnostics can identify fixture and forecast ambiguity for
planning follow-up."

Caveat: Diagnostic only; no sim-to-real, perception, or paper-facing robustness claim.

## 6. Negative-Result Summary
| Area | Verdict | Reason |
|---|---|---|
| CARLA Replay Parity | blocked | Robot actor spawn failure prevents oracle replay. |
| Predictive Planner v2 | negative | Obstacle-feature success worsened despite forecast improvement. |
| tab_issue_1023_campaign_table | non-claimable | Missing payload file |
| tab_issue_1023_scenario_family_breakdown | non-claimable | Missing payload file |
| unnamed | stale-needs-refresh | Output artifact missing output path |

## Conservative Rules Applied

- Draft-only unless dependencies are current and claimable.
- Diagnostic, stale, non-claimable, unavailable, fallback, degraded, proxy-only, or
  missing-denominator rows weaken or block wording.
- Fallback behavior is not acceptable as a successful benchmark outcome.
- No invented values; missing tracked sources produce unavailable rows.
