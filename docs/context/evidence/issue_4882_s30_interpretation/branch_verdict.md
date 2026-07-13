# S30 hybrid-versus-ORCA branch verdict

**Claim boundary:** Diagnostic-only S30 interpretation. This packet does not promote a paper, dissertation, record-breaking, universal-planner, or real-world claim.

**Verdict:** `branch_a_separation`

The primary decision uses the paired seed-block 95% interval for success. Collision-event separation is a secondary safety check. Adapter/native and simulator-only limitations remain.

| Hybrid arm | Success delta (95% CI) | Collision-event delta (95% CI) |
|---|---:|---:|
| `hybrid_rule_v3_fast_progress_static_escape` | +0.0792 [+0.0521, +0.1069] | -0.0778 [-0.1014, -0.0548] |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | +0.0910 [+0.0646, +0.1167] | -0.1347 [-0.1590, -0.1097] |
| `scenario_adaptive_hybrid_orca_v1` | +0.0729 [+0.0458, +0.1000] | -0.0681 [-0.0931, -0.0444] |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | +0.0729 [+0.0458, +0.1000] | -0.0722 [-0.0972, -0.0479] |
