## static_recenter contrastive mechanism panel (Issue #2227)

- Mechanism flag toggled (isolation): `static_recenter_enabled` (on vs off, only this flag differs).
- Scenario: `classic_bottleneck_low` (seed 113, horizon 160, dt 0.1).
- Expected to act here? YES - static deadlock / local-minimum bottleneck where the recenter probe is expected to perturb the robot off the wall (Issue #2592 active row).
- Activated? YES (diagnostic: `static_recenter_term_positive_in_decision_trace`; raw={"active": true, "first_activation_step": 7, "recenter_term_activation_count": 4, "static_recenter_command_selected_count": 0}).
- Command/source changed between arms? YES (on={"dynamic_window": 105, "path_follow_0.5m": 16, "route_guide": 1}; off={"dynamic_window": 158, "path_follow_0.5m": 1, "route_guide": 1}).
- Outcome changed? YES (on={"collisions": 0, "near_misses": 0, "status": "success", "steps": 122, "success": true, "termination_reason": "success"}; off={"collisions": 0, "near_misses": 0, "status": "failure", "steps": 160, "success": false, "termination_reason": "max_steps"}).
- Final-pose trajectory delta: 19.6183 m.
- Classification: `activated_outcome_changed`.

Claim boundary: `diagnostic_only` / evidence tier `stress`. Planner-level activation accounting only; NOT a navigation-success, benchmark, ranking, or perception claim. Traces come from actual planner runs; only the mechanism flag differs between arms.
