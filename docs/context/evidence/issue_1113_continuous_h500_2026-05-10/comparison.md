# Policy Search Comparison

| Candidate | Stage | Success | Collision | Near Miss | Classic Coll. | Francis Coll. |
|---|---|---:|---:|---:|---:|---:|
| hybrid_rule_v3_fast_progress_static_escape_continuous | full_matrix_h500 | 0.9167 | 0.0139 | 0.3958 | 0.0145 | 0.0133 |
| scenario_adaptive_hybrid_orca_v1 | full_matrix_h500 | 0.9097 | 0.0208 | 0.4236 | 0.0290 | 0.0133 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | full_matrix_h500 | 0.9028 | 0.0139 | 0.4236 | 0.0145 | 0.0133 |
| ppo | baseline_reference | 0.2482 | 0.0993 | n/a | n/a | n/a |
| orca | baseline_reference | 0.1844 | 0.0355 | n/a | 0.0300 | 0.0400 |
| goal | baseline_reference | 0.0142 | 0.2411 | n/a | n/a | n/a |
