# Issue #2869 Forecast Risk Calibration Filter Diagnostic

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2869>

## Boundary

- **schema_version**: `forecast_risk_calibration_filter.diagnostic_comparison.v1`
- **claim_boundary**: `diagnostic_only_not_benchmark_evidence`
- **calibration_report**: `docs/context/evidence/issue_2865_forecast_calibration_report_2026-06-15/calibration_report.json`
- **diagnostic_weight**: 5.0
- **recommendation**: `wait`

## Risk Mode Comparison

| mode | status | forecast_risk_weight | high_risk selected | high_risk speed | high_risk penalty | false_positive selected | false_positive speed | false_positive penalty |
|---|---:|---:|---|---:|---:|---|---:|---:|
| no_risk | available | 0.0 | goal | 1.000000 | 0.000000 | goal | 1.000000 | 0.000000 |
| raw_risk | available | 5.0 | risk_dwa | 0.200000 | 5.000000 | goal | 1.000000 | 0.000000 |
| calibration_filtered | blocked | 5.0 | NA | NA | NA | NA | NA | NA |
| actor_class_aware | blocked | 5.0 | NA | NA | NA | NA | NA | NA |
| observation_tier_aware | blocked | 5.0 | NA | NA | NA | NA | NA | NA |

### Blocked mode reasons

- **calibration_filtered**: no_rows_risk_scoring_eligible
- **actor_class_aware**: actor_class_unavailable_in_all_rows
- **observation_tier_aware**: single_observation_tier: deployable_tracked

## Tradeoffs

| tradeoff | value |
|---|---|
| route_progress_tradeoff (no_risk - raw_risk) | 0.800000 |
| false_positive_stopping_avoided | True |
| false_positive_unnecessary_slowdown_count | 0 |

## Recommendation

**WAIT**

Calibration filtering cannot gate forecast-risk scoring: the Issue #2865 report contains no risk-scoring-eligible rows. Keep forecast-risk scoring opt-in and diagnostic-only until eligible calibration evidence exists.

> This report is diagnostic-only. It does not establish safety, navigation benefit, human realism, benchmark-strength predictor quality, or paper/dissertation claims.
