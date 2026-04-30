# Policy Search Experiment Ledger

| Date | Candidate | Stage | Success | Collision | Near Miss | Classic Coll. | Francis Coll. | Decision | Note |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| 2026-04-29 | orca | frozen baseline | 0.1844 | 0.0355 | n/a | 0.0300 | 0.0400 | reference | Paper-facing safety anchor. |
| 2026-04-29 | ppo | frozen baseline | 0.2482 | 0.0993 | n/a | n/a | n/a | reference | Paper-facing success anchor, but unsafe. |
| 2026-04-29 | goal | frozen baseline | 0.0142 | 0.2411 | n/a | n/a | n/a | reference | Diagnostic lower bound. |
| 2026-04-29 | hybrid_orca_sampler_v1 | smoke | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | pass | Runs end to end, but all three `planner_sanity_simple` seeds timed out with low progress. |
| 2026-04-29 | risk_guarded_ppo_v1 | smoke | 1.0000 | 0.0000 | 0.0000 | n/a | n/a | pass | Completes `planner_sanity_simple` across all three smoke seeds. |
| 2026-04-29 | scenario_adaptive_orca_v1 | smoke | 1.0000 | 0.0000 | 0.0000 | n/a | n/a | pass | Family-override execution path now works locally and clears all three smoke seeds. |
| 2026-04-29 | planner_selector_v1 | smoke | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | pass | Runs end to end, but all three smoke seeds timed out with low progress. |
| 2026-04-29 | mpc_clearance_sampler_v1 | smoke | 1.0000 | 0.0000 | 0.0000 | n/a | n/a | pass | Increased the candidate open-space speed cap to 1.8 m/s; all three `planner_sanity_simple` smoke seeds now complete without collision or near misses. |
| 2026-04-29 | risk_guarded_ppo_v1 | nominal_sanity | 0.1667 | 0.2778 | 0.0556 | 0.4167 | 0.0000 | revise | Fails the nominal gate; classic scenarios dominate both collisions and low-progress timeouts. |
| 2026-04-29 | scenario_adaptive_orca_v1 | nominal_sanity | 0.1667 | 0.0000 | 0.1111 | 0.0000 | 0.0000 | revise | Collision-free but still stalled across classic and Francis subsets; the aggressive retune only increased intrusive near misses and was reverted. |
| 2026-04-30 | hybrid_rule_v0_minimal | smoke | 1.0000 | 0.0000 | 0.0000 | n/a | n/a | pass | Real map-runner smoke passed with planner-runtime diagnostics; three-seed open-sanity comparison also reached 1.0000 success vs `goal` 0.0000. |
| 2026-04-30 | hybrid_rule_v0_minimal | nominal_sanity | 0.1667 | 0.4444 | 0.2222 | 0.4167 | 1.0000 | revise | Minimal DWA is not promotion-ready; failures are mostly static-collision termination, low-progress timeouts, and intrusive near misses. |
| 2026-04-30 | hybrid_rule_v3_teb_like_rollout | smoke | 1.0000 | 0.0000 | 0.0000 | n/a | n/a | pass | Route-guide candidate preserves the open-sanity smoke behavior. |
| 2026-04-30 | hybrid_rule_v3_teb_like_rollout | nominal_sanity | 0.2778 | 0.0000 | 0.1667 | 0.0000 | 0.0000 | revise | Best corrected hybrid-rule result so far: collision-free and better than v0, but still dominated by low-progress timeouts. |
| 2026-04-30 | hybrid_rule_v4_recovery_aware | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 0.0000 | 0.0000 | reject | Static reorientation and stall-rotation recovery reduced fallback counts but lost doorway successes; do not promote over v3. |
| 2026-04-30 | hybrid_rule_v3_fast_progress | nominal_sanity | 0.1667 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | reject | Raising the speed envelope to 3.0 m/s increased intrusive near misses and collapsed classic-scenario success. |
| 2026-04-30 | hybrid_rule_v3_dynamic_relaxed | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 0.0000 | 0.0000 | reject | Shortening the hard dynamic horizon recovered one doorway seed but lost others; no aggregate gain over corrected v3. |
| 2026-04-30 | hybrid_rule_v3_progress_2p4 | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 0.0000 | 0.0000 | reject | Moderate 2.4 m/s speed and stronger progress pressure did not recover long-route timeouts and lost one success relative to corrected v3. |
| 2026-04-30 | hybrid_rule_v3_teb_like_rollout | nominal_sanity | 0.2222 | 0.0000 | 0.1667 | 0.0000 | 0.0000 | reject | Full-rollout static hard-clearance with the shared 5 cm hard margin removed the doorway static-collision risk but lost one doorway success to freezing. |
| 2026-04-30 | hybrid_rule_v3_static_margin0 | nominal_sanity | 0.2778 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | keep | Best current-code candidate: full-rollout static hard-clearance with 0 cm static margin restores the 5/18 success count and keeps zero collisions, at the cost of more intrusive near misses than the stricter margin. |
| 2026-04-30 | hybrid_rule_v3_static_margin0 | stress_slice | 0.2917 | 0.0000 | 0.2500 | 0.0000 | 0.0000 | tracked | Selected policy remains collision-free on stress slice, but still has low-progress timeouts and intrusive near misses. |
| 2026-04-30 | hybrid_rule_v3_static_margin0_waypoint2 | nominal_sanity | 0.2778 | 0.0000 | 0.2222 | 0.0000 | 0.0000 | keep | Ties selected margin-0 primary nominal metrics while improving mean minimum distance and average speed. |
| 2026-04-30 | hybrid_rule_v3_static_margin0_waypoint2 | stress_slice | 0.3333 | 0.0000 | 0.2083 | 0.0000 | 0.0000 | keep | New best: improves stress success and near-miss rate while preserving zero collisions. |
| 2026-04-30 | hybrid_rule_v3_static_margin0_comfort | nominal_sanity | 0.2222 | 0.0556 | 0.1111 | 0.0833 | 0.0000 | reject | Stronger comfort pressure reduced near misses but introduced a static collision. |
| 2026-04-30 | hybrid_rule_v3_static_margin0_waypoint3 | nominal_sanity | 0.2778 | 0.0556 | 0.1111 | 0.0833 | 0.0000 | reject | Larger route handoff improved speed but introduced a static collision. |
| 2026-04-30 | hybrid_rule_v3_waypoint2_mild_comfort | nominal_sanity | 0.2778 | 0.0556 | 0.0556 | 0.0833 | 0.0000 | reject | Mild comfort shaping reduced near misses but introduced a static collision. |
| 2026-04-30 | hybrid_rule_v3_waypoint2_progress | nominal_sanity | 0.2778 | 0.0556 | 0.1111 | 0.0833 | 0.0000 | reject | Stronger progress/speed pressure increased speed but introduced a static collision. |
| 2026-04-30 | mpc_clearance_sampler_v1 | nominal_sanity | 0.1667 | 0.2778 | 0.2222 | 0.4167 | 0.0000 | reject | Existing deterministic MPC-clearance candidate is unsafe on the nominal slice due static collisions. |
