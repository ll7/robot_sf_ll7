# SNQI Scalarization Sensitivity Diagnostic

This is a diagnostic export for Social Navigation Quality Index (SNQI) scalarization sensitivity; it is not benchmark evidence and does not establish SNQI as a primary index.

## Summary

- Decision disagreement rate: `0.138889`
- Max weight-sweep disagreement rate vs base: `0.250000`
- Max weight-zero pairwise reversals vs base: `9`
- Top term by mean absolute contribution: `w_time`

## Planner Rows

| Planner | SNQI rank | Constraints-first rank | Rank delta | SNQI mean | Constraints-first score | Pareto front |
|---|---:|---:|---:|---:|---:|:---:|
| ppo | 1 | 3 | +2 | -0.048550 | 0.206791 | yes |
| hybrid_rule_v3_fast_progress_static_escape | 2 | 1 | -1 | -0.079170 | 0.287886 | yes |
| scenario_adaptive_hybrid_orca_v1 | 3 | 2 | -1 | -0.080031 | 0.274838 | no |
| orca | 4 | 4 | +0 | -0.102191 | 0.093697 | no |
| socnav_sampling | 5 | 7 | +2 | -0.143172 | -0.600579 | no |
| prediction_planner | 6 | 5 | -1 | -0.180360 | -0.348098 | no |
| social_force | 7 | 6 | -1 | -0.207414 | -0.582915 | no |
| sacadrl | 8 | 9 | +1 | -0.238074 | -0.820221 | no |
| goal | 9 | 8 | -1 | -0.241916 | -0.797935 | no |

## Term Dominance

| Component | Mean abs contribution | Share |
|---|---:|---:|
| w_time | 0.081920 | 0.271656 |
| w_near | 0.081840 | 0.271394 |
| w_success | 0.077396 | 0.256655 |
| w_collisions | 0.044361 | 0.147107 |
| w_force_exceed | 0.010496 | 0.034806 |
| w_comfort | 0.004292 | 0.014232 |
| w_jerk | 0.001252 | 0.004150 |

## Out Of Scope

- No full benchmark campaign run.
- No Slurm or GPU submission.
- No paper or dissertation claim edits.
