# Issue #2446 AMV Actuation Feasibility Ranking 2026-06-19

Status: diagnostic-only analysis (`analysis_only`), not benchmark or paper-facing evidence.

Related:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2446
- Timeout-driver closure: [issue_2440_amv_timeout_closure.md](issue_2440_amv_timeout_closure.md)
- Trace review predecessor: [issue_2443_amv_trace_review.md](issue_2443_amv_trace_review.md)
- AMV trace-boundary decision: [issue_2531_amv_trace_boundary.md](issue_2531_amv_trace_boundary.md)

```yaml
amv_feasibility_dimension:
  source_issues:
    - 2224
    - 2259
    - 2268
    - 2308
    - 2404
    - 2440
    - 2443
    - 2522
    - 2531
  compared_candidates:
    scenario: classic_cross_trap_high
    seed: 101
    stage: amv_actuation_smoke
    horizon: 80
    synthetic_actuation_profile: amv-actuation-stress-v0
    baseline:
      candidate: hybrid_rule_v3_fast_progress
      success_rate: 0.0
      collision_rate: 0.0
      timeout_mode: timeout_low_progress
      command_clip_fraction: 0.2750
      command_clip_steps: 22
      yaw_saturation_fraction: 0.0
      mean_avg_speed_m_s: 1.662309936770821
      final_route_progress_m: 12.904024665994093
      final_distance_to_goal_m: 3.6078725186171776
    intervention:
      candidate: actuation_aware_hybrid_rule_v0
      success_rate: 0.0
      collision_rate: 0.0
      timeout_mode: timeout_low_progress
      command_clip_fraction: 0.1875
      command_clip_steps: 15
      yaw_saturation_fraction: 0.0
      mean_avg_speed_m_s: 1.667037499328765
      final_route_progress_m: 12.87191147075262
      final_distance_to_goal_m: 3.639985713858651
    deltas_intervention_minus_baseline:
      command_clip_fraction: -0.0875
      command_clip_steps: -7
      yaw_saturation_fraction: 0.0
      mean_avg_speed_m_s: +0.004727562557943961
      final_route_progress_m: -0.032113195241472634
      final_distance_to_goal_m: +0.03211319524147352
      last_10_step_progress_delta_m: +0.12100258841798175
  success_ordering:
    outcome: no_ranking_difference
    reason: both candidates failed on identical mechanism class and success metrics
    ordered_candidates:
      - hybrid_rule_v3_fast_progress
      - actuation_aware_hybrid_rule_v0
    note: equal on success, collision, and timeout mode for this matched row
  feasibility_ordering:
    ordering:
      - actuation_aware_hybrid_rule_v0
      - hybrid_rule_v3_fast_progress
    reason: feasibility metrics improved (fewer clips, lower clip fraction) while terminal metrics stayed static
  disagreement_cases:
    - type: feasibility_success_divergence
      summary: command clipping improved (-0.0875 fraction, -7 steps) but timeout mode and success remained identical
    - type: feasibility_speed_vs_progress
      summary: mean speed was nearly unchanged (+0.0047 m/s), yet final progress/distance moved slightly against the intervention (+0.0321 m farther from goal)
    - type: feasibility_trace_limit
      summary: route/task progress blocking appears primary; raw simulation_trace_export.v1 frame/event IDs are blocked in compact artifacts
  diagnostic_value: diagnostic ranking signal
  planner_objective_recommendation:
    objective_fit: diagnostic-only_feasibility_ranking
    recommendation:
      - keep actuation-aware candidate as a feasibility-aware secondary ranking axis under a fixed success/collision/timeout objective
      - do not treat feasibility gains as planner improvement until route-progress geometry or task-completion blocker is addressed
      - no planner-ranking or paper-facing claim from this slice
```

This issue-level synthesis is diagnostic-only and uses existing tracked artifacts from #2440, #2443, and related AMV actuation closure/decomposition threads. Feasibility remains informative for secondary ordering among tied terminal outcomes, but it does **not** invert success interpretation on this matched slice.

## Interpretation Boundary

- The source row is a matched `classic_cross_trap_high` seed `101` AMV actuation-smoke slice, not a
  broad benchmark rerun.
- The feasibility ordering is useful only as a secondary diagnostic dimension when primary
  terminal outcomes are tied.
- The compact artifacts do not include raw `simulation_trace_export.v1` frame/event IDs, so the
  route-blocker interpretation remains summary-timeline-limited.
- This note makes no calibrated AMV hardware, planner-ranking, or paper-facing claim.
