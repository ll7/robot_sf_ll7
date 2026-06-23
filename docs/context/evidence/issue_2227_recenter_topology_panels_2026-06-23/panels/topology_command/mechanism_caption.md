## topology_command contrastive mechanism panel (Issue #2227)

- Mechanism flag toggled (isolation): `topology_command_enabled` (on vs off, only this flag differs).
- Scenario: `classic_bottleneck_medium` (seed 111, horizon 160, dt 0.1).
- Expected to act here? YES - bottleneck route-ambiguity slice where >=2 distinct masked-route hypotheses are expected, allowing a topology-hypothesis command to be selected (topology reselection hard slice).
- Activated? YES (diagnostic: `topology_status_counts_and_topology_hypothesis_source`; raw={"active": true, "selected_hypothesis_counts": {"masked_cell_76_87": 7, "masked_cell_77_87": 13, "masked_cell_78_87": 2, "primary_route": 121}, "status_counts": {"insufficient_hypotheses": 12, "not_available": 5, "ok": 143}, "topology_command_enabled": true, "topology_hypothesis_command_count": 37}).
- Command/source changed between arms? YES (on={"corridor_subgoal": 6, "dynamic_window": 103, "path_follow_0.5m": 6, "route_guide": 3, "topology_fail_closed": 5, "topology_hypothesis": 37}; off={"corridor_subgoal": 2, "dynamic_window": 119, "path_follow_0.5m": 10, "route_guide": 18, "topology_fail_closed": 8}).
- Outcome changed? YES (on={"collisions": 0, "near_misses": 0, "status": "failure", "steps": 160, "success": false, "termination_reason": "max_steps"}; off={"collisions": 0, "near_misses": 9, "status": "success", "steps": 157, "success": true, "termination_reason": "success"}).
- Final-pose trajectory delta: 1.8416 m.
- Classification: `activated_outcome_changed`.

Claim boundary: `diagnostic_only` / evidence tier `stress`. Planner-level activation accounting only; NOT a navigation-success, benchmark, ranking, or perception claim. Traces come from actual planner runs; only the mechanism flag differs between arms.
