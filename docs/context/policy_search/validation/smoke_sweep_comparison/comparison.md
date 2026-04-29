# Policy Search Comparison

| Candidate | Stage | Success | Collision | Near Miss | Classic Coll. | Francis Coll. |
|---|---|---:|---:|---:|---:|---:|
| risk_guarded_ppo_v1 | smoke | 1.0000 | 0.0000 | 0.0 | 0.0 | 0.0 |
| scenario_adaptive_orca_v1 | smoke | 1.0000 | 0.0000 | 0.0 | 0.0 | 0.0 |
| ppo | baseline_reference | 0.2482 | 0.0993 | n/a | n/a | n/a |
| orca | baseline_reference | 0.1844 | 0.0355 | n/a | 0.03 | 0.04 |
| goal | baseline_reference | 0.0142 | 0.2411 | n/a | n/a | n/a |
| hybrid_orca_sampler_v1 | smoke | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 |
| mpc_clearance_sampler_v1 | smoke | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 |
| planner_selector_v1 | smoke | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 |
