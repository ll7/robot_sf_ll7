# Policy Search Comparison

| Candidate | Stage | Success | Collision | Near Miss | Classic Coll. | Francis Coll. |
|---|---|---:|---:|---:|---:|---:|
| scenario_adaptive_hybrid_orca_v1 | full_matrix | 0.2778 | 0.0139 | 0.3402777777777778 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v3_progress_2p4 | full_matrix | 0.2708 | 0.0139 | 0.3402777777777778 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v3_fast_progress_static_escape | full_matrix | 0.2639 | 0.0139 | 0.3402777777777778 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v3_waypoint2_route_lookahead8_static02 | full_matrix | 0.2639 | 0.0139 | 0.3333333333333333 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v3_waypoint2_route_lookahead8_static05 | full_matrix | 0.2639 | 0.0139 | 0.3333333333333333 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v0_minimal | full_matrix | 0.2569 | 0.0139 | 0.3263888888888889 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v3_fast_progress | full_matrix | 0.2569 | 0.0139 | 0.3333333333333333 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v4_recovery_aware | full_matrix | 0.2500 | 0.0139 | 0.3125 | 0.014492753623188406 | 0.013333333333333334 |
| ppo | baseline_reference | 0.2482 | 0.0993 | n/a | n/a | n/a |
| hybrid_rule_v3_dynamic_relaxed | full_matrix | 0.2431 | 0.0139 | 0.3125 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v3_teb_like_rollout | full_matrix | 0.2431 | 0.0139 | 0.3194444444444444 | 0.014492753623188406 | 0.013333333333333334 |
| hybrid_rule_v3_waypoint2_route_commit | full_matrix | 0.2431 | 0.0903 | 0.3472222222222222 | 0.10144927536231885 | 0.08 |
| hybrid_rule_v3_waypoint2_route_lookahead6 | full_matrix | 0.2431 | 0.0972 | 0.3333333333333333 | 0.10144927536231885 | 0.09333333333333334 |
| hybrid_rule_v3_waypoint2_speed2p2 | full_matrix | 0.2361 | 0.1042 | 0.3888888888888889 | 0.10144927536231885 | 0.10666666666666667 |
| mpc_clearance_sampler_v1 | full_matrix | 0.2361 | 0.2847 | 0.13194444444444445 | 0.42028985507246375 | 0.16 |
| hybrid_rule_v3_waypoint2_progress | full_matrix | 0.2292 | 0.1250 | 0.3333333333333333 | 0.13043478260869565 | 0.12 |
| hybrid_rule_v3_waypoint2_route_lookahead8_clearance1 | full_matrix | 0.2292 | 0.1250 | 0.3263888888888889 | 0.13043478260869565 | 0.12 |
| hybrid_rule_v3_waypoint2_route_lookahead8 | full_matrix | 0.2222 | 0.1250 | 0.3402777777777778 | 0.13043478260869565 | 0.12 |
| hybrid_rule_v3_waypoint2_route_lookahead8_inflation4 | full_matrix | 0.2222 | 0.1250 | 0.3402777777777778 | 0.13043478260869565 | 0.12 |
| hybrid_rule_v3_waypoint2_dynamic_clearance | full_matrix | 0.2153 | 0.1042 | 0.3263888888888889 | 0.08695652173913043 | 0.12 |
| hybrid_rule_v3_static_margin0 | full_matrix | 0.2153 | 0.1111 | 0.3263888888888889 | 0.10144927536231885 | 0.12 |
| hybrid_rule_v3_static_margin0_waypoint2 | full_matrix | 0.2153 | 0.1111 | 0.3263888888888889 | 0.10144927536231885 | 0.12 |
| hybrid_rule_v3_waypoint2_static_escape | full_matrix | 0.2153 | 0.1319 | 0.3263888888888889 | 0.14492753623188406 | 0.12 |
| hybrid_rule_v3_waypoint2_mild_comfort | full_matrix | 0.2014 | 0.0903 | 0.2847222222222222 | 0.057971014492753624 | 0.12 |
| orca | baseline_reference | 0.1844 | 0.0355 | n/a | 0.03 | 0.04 |
| hybrid_rule_v3_static_margin0_comfort | full_matrix | 0.1806 | 0.0972 | 0.2361111111111111 | 0.057971014492753624 | 0.13333333333333333 |
| hybrid_rule_v3_static_margin0_waypoint3 | full_matrix | 0.1458 | 0.1111 | 0.3472222222222222 | 0.15942028985507245 | 0.06666666666666667 |
| risk_guarded_ppo_v1 | full_matrix | 0.1181 | 0.1736 | 0.22916666666666666 | 0.2028985507246377 | 0.14666666666666667 |
| scenario_adaptive_orca_v1 | full_matrix | 0.0486 | 0.0347 | 0.2708333333333333 | 0.014492753623188406 | 0.05333333333333334 |
| planner_selector_v1 | full_matrix | 0.0486 | 0.3056 | 0.2569444444444444 | 0.4057971014492754 | 0.21333333333333335 |
| goal | baseline_reference | 0.0142 | 0.2411 | n/a | n/a | n/a |
| hybrid_orca_sampler_v1 | full_matrix | 0.0139 | 0.1111 | 0.16666666666666666 | 0.17391304347826086 | 0.05333333333333334 |
