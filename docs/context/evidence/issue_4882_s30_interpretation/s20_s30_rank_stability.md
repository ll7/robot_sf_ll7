# S20-prefix versus S30 rank stability

**Claim boundary:** Diagnostic-only S30 interpretation. This packet does not promote a paper, dissertation, record-breaking, universal-planner, or real-world claim.

S20 is the preregistered 111-130 prefix, not an independent rerun; this isolates seed-budget sensitivity on the identical h600 matrix.

## `success`

- Kendall tau: `0.8667`
- Top planner changed: `false`
- S20 order: `hybrid_rule_v3_fast_progress_static_escape_continuous > hybrid_rule_v3_fast_progress_static_escape > scenario_adaptive_hybrid_orca_v2_collision_guard > scenario_adaptive_hybrid_orca_v1 > ppo > orca`
- S30 order: `hybrid_rule_v3_fast_progress_static_escape_continuous > hybrid_rule_v3_fast_progress_static_escape > scenario_adaptive_hybrid_orca_v1 > scenario_adaptive_hybrid_orca_v2_collision_guard > ppo > orca`

## `collision_event`

- Kendall tau: `1.0000`
- Top planner changed: `false`
- S20 order: `hybrid_rule_v3_fast_progress_static_escape_continuous > hybrid_rule_v3_fast_progress_static_escape > scenario_adaptive_hybrid_orca_v2_collision_guard > scenario_adaptive_hybrid_orca_v1 > ppo > orca`
- S30 order: `hybrid_rule_v3_fast_progress_static_escape_continuous > hybrid_rule_v3_fast_progress_static_escape > scenario_adaptive_hybrid_orca_v2_collision_guard > scenario_adaptive_hybrid_orca_v1 > ppo > orca`

## `time_to_goal_norm`

- Kendall tau: `0.3333`
- Top planner changed: `false`
- S20 order: `hybrid_rule_v3_fast_progress_static_escape > hybrid_rule_v3_fast_progress_static_escape_continuous > scenario_adaptive_hybrid_orca_v1 > scenario_adaptive_hybrid_orca_v2_collision_guard > ppo > orca`
- S30 order: `hybrid_rule_v3_fast_progress_static_escape > ppo > scenario_adaptive_hybrid_orca_v1 > scenario_adaptive_hybrid_orca_v2_collision_guard > hybrid_rule_v3_fast_progress_static_escape_continuous > orca`
