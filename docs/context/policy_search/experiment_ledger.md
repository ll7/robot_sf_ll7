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
| 2026-04-29 | mpc_clearance_sampler_v1 | smoke | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | pass | Runs end to end, but all three smoke seeds timed out with low progress. |