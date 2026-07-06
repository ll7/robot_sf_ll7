# Issue #4013 Learned-Prediction MPC Diagnostic Comparison

- Claim boundary: diagnostic matched-scenario comparison; not paper-grade benchmark evidence
- Evidence tier: diagnostic-only
- Status: diagnostic_ready
- Paired seed count: 1
- World-model exclusions: dreamerv3, planet, td_mpc2, large_generative_world_model, paper_grade_claim

| Role | Episodes | Evidence episodes | Success rate | Collision rate | Near-miss rate | Excluded fallback/degraded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned_prediction_mpc | 1 | 1 | 0.000 | 0.000 | 0.000 | 0 |
| cv_prediction_mpc | 1 | 1 | 0.000 | 0.000 | 0.000 | 0 |
| model_free_baseline | 1 | 1 | 0.000 | 0.000 | 0.000 | 0 |

## Blockers
- none

## Closure Criteria
- met: short-horizon predictor checkpoint loads without fallback
- met: model-based action selection runs on smoke scenario
- met: comparison includes cv_prediction_mpc and one model-free baseline
- met: fallback/degraded rows are excluded from evidence
- met: claim boundary excludes large world-model and paper-grade claims
