# Issue #2869 Forecast Risk Calibration Filter Diagnostic

## Scope

This bundle compares five forecast-risk scoring modes before trusting planner coupling:
`no_risk`, `raw_risk`, `calibration_filtered`, `actor_class_aware`, and `observation_tier_aware`.

## Evidence status

- `schema`: `forecast_risk_calibration_filter.diagnostic_comparison.v1`
- `calibration_report`: [docs/context/evidence/issue_2865_forecast_calibration_report_2026-06-15/calibration_report.json](../issue_2865_forecast_calibration_report_2026-06-15/calibration_report.json)
- `report`: [report.json](report.json) and [report.md](report.md)
- `claim_boundary`: diagnostic_only_not_benchmark_evidence
- `recommendation`: `wait`

## Reproducible command

```
uv run python scripts/validation/validate_forecast_risk_calibration_filter.py \
  --out-dir docs/context/evidence/issue_2869_forecast_risk_calibration_filter_2026-06-15
```

## Validation

```
uv run pytest tests/validation/test_forecast_risk_calibration_filter.py
```

## Claim boundary

This report is diagnostic-only. It does not establish safety, navigation benefit, human realism, benchmark-strength predictor quality, or paper/dissertation claims. Forecast-risk scoring remains opt-in with `forecast_risk_weight=0.0` by default.
