# Forecast Calibration Report

- Report id: issue_2904_forecast_risk_eligibility_2026-06-20
- Decision: continue
- Claim status: analysis-only
- Reliability rows: 2
- Limitation rows: 0

| scenario_family | horizon_s | observation_tier | predictor_family | actor_class | semantic_metadata | coverage | miss_rate | status | sharpness | failure_taxonomy | eligibility |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |
| classic_crossing | 1 | deployable_tracked | probabilistic_cv | pedestrian | present | 0.91 | NA | calibrated_within_tolerance | 0.24 | none_observed | eligible_analysis_only |
| motion_rich_bicycle | 1 | deployable_observation | probabilistic_cv | bicycle | present | 0.88 | NA | calibrated_within_tolerance | 0.31 | none_observed | eligible_analysis_only |

Calibration summaries are analysis-only evidence. They do not prove planner safety or navigation benefit, and unavailable uncertainty denominators remain limitations.
