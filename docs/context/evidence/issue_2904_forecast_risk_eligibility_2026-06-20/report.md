# Issue #2869 Forecast Risk Calibration Filter Diagnostic

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2869>

## Boundary

- **schema_version**: `forecast_risk_calibration_filter.diagnostic_comparison.v1`
- **claim_boundary**: `diagnostic_only_not_benchmark_evidence`
- **calibration_report**: `docs/context/evidence/issue_2904_forecast_risk_eligibility_2026-06-20/calibration_report.json`
- **diagnostic_weight**: 5.0
- **recommendation**: `diagnostic_only`

## Risk Mode Comparison

| mode | status | forecast_risk_weight | high_risk selected | high_risk speed | high_risk penalty | false_positive selected | false_positive speed | false_positive penalty |
|---|---:|---:|---|---:|---:|---|---:|---:|
| no_risk | available | 0.0 | goal | 1.000000 | 0.000000 | goal | 1.000000 | 0.000000 |
| raw_risk | available | 5.0 | risk_dwa | 0.200000 | 5.000000 | goal | 1.000000 | 0.000000 |
| calibration_filtered | available | 5.0 | goal | 1.000000 | 0.000000 | goal | 1.000000 | 0.000000 |
| actor_class_aware | available | 5.0 | goal | 1.000000 | 0.000000 | goal | 1.000000 | 0.000000 |
| observation_tier_aware | available | 5.0 | goal | 1.000000 | 0.000000 | goal | 1.000000 | 0.000000 |

### Blocked mode reasons


## Tradeoffs

| tradeoff | value |
|---|---|
| route_progress_tradeoff (no_risk - raw_risk) | 0.800000 |
| false_positive_stopping_avoided | True |
| false_positive_unnecessary_slowdown_count | 0 |

## Recommendation

**DIAGNOSTIC_ONLY**

Calibration-filtered rows are available in the input report, but this tool remains a deterministic diagnostic comparison rather than benchmark evidence. Use it only to decide whether a same-seed benchmark comparison is now warranted.

> This report is diagnostic-only. It does not establish safety, navigation benefit, human realism, benchmark-strength predictor quality, or paper/dissertation claims.
