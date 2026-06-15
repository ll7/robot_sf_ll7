# Forecast Risk Policy Stack Diagnostic Comparison

- **claim_boundary**: `diagnostic_only_not_benchmark_evidence`
- **diagnostic_weight**: 5.0
- **recommendation**: `forecast_risk_scoring_diagnostic_consistent`

## Cases

### high_risk_diagnostic_slows_goal

| Field | Baseline | Diagnostic | Delta |
|---|---|---|---|
| selected_proposal | goal | risk_dwa |  |
| speed | 1.0 | 0.2 | -0.8 |
| progress_proxy | 1.0 | 0.2 | -0.8 |
| forecast_penalty | 0.0 | 5.0 | 5.0 |
| high_risk_speed_reduction | 0.0 | 1.0 | 1.0 |
| shield_stop_count | 0 | 0 | 0 |
| false_positive_unnecessary_slowdown_count |  | 0 |  |

### false_positive_suppresses_penalty

| Field | Baseline | Diagnostic | Delta |
|---|---|---|---|
| selected_proposal | goal | goal |  |
| speed | 1.0 | 1.0 | 0.0 |
| progress_proxy | 1.0 | 1.0 | 0.0 |
| forecast_penalty | 0.0 | 0.0 | 0.0 |
| high_risk_speed_reduction | 0.0 | 0.0 | 0.0 |
| shield_stop_count | 0 | 0 | 0 |
| false_positive_unnecessary_slowdown_count |  | 0 |  |

## Recommendation

```forecast_risk_scoring_diagnostic_consistent```

> This is a diagnostic-only comparison. No safety claim is made. claim_boundary=diagnostic_only_not_benchmark_evidence
