# Policy Search Comparison

| Candidate | Stage | Success | Collision | Near Miss | Classic Coll. | Francis Coll. |
|---|---|---:|---:|---:|---:|---:|
| hybrid_rule_v3_fast_progress_static_escape_continuous | full_matrix_h500 | 0.9167 | 0.0139 | 0.3958333333333333 | 0.014492753623188406 | 0.013333333333333334 |
| scenario_adaptive_hybrid_orca_v1 | full_matrix_h500 | 0.9097 | 0.0208 | 0.4236111111111111 | 0.028985507246376812 | 0.013333333333333334 |
| scenario_adaptive_hybrid_orca_v2_collision_guard | full_matrix_h500 | 0.9028 | 0.0139 | 0.4236111111111111 | 0.014492753623188406 | 0.013333333333333334 |
| ppo | baseline_reference | 0.2482 | 0.0993 | n/a | n/a | n/a |
| orca | baseline_reference | 0.1844 | 0.0355 | n/a | 0.03 | 0.04 |
| goal | baseline_reference | 0.0142 | 0.2411 | n/a | n/a | n/a |
