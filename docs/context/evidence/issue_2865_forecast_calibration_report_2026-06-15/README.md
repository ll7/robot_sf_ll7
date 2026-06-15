# Issue #2865 Forecast Calibration Report

## Scope

This analysis-only bundle converts the Issue #2868 forecast comparison rows into
`ForecastCalibrationReport.v1` reliability rows for the five deterministic baselines.

## Evidence status

- `schema`: `ForecastCalibrationReport.v1`
- `source`: `docs/context/evidence/issue_2868_semantic_metadata_fixtures_2026-06-15/comparison_report.json`
- `report`: [calibration_report.json](calibration_report.json) and
  [calibration_report.md](calibration_report.md)
- `claim_boundary`: diagnostic-only / analysis-only
- `reason`: the report separates cv, signal_aware, goal_aware, semantic, and
  interaction_aware rows, but actor class is unavailable in the source comparison rows and two
  legacy families have zero denominators.

## Reproducible command

```
uv run python scripts/benchmark/build_forecast_calibration_from_cv_comparison.py \
  docs/context/evidence/issue_2868_semantic_metadata_fixtures_2026-06-15/comparison_report.json \
  --report-id issue_2865_forecast_calibration_report_2026-06-15 \
  --generated-at-utc 2026-06-15T00:00:00+00:00 \
  --out-json docs/context/evidence/issue_2865_forecast_calibration_report_2026-06-15/calibration_report.json \
  --out-md docs/context/evidence/issue_2865_forecast_calibration_report_2026-06-15/calibration_report.md
```

## Validation

```
uv run pytest tests/benchmark/test_forecast_calibration_report.py
```

## Claim boundary

The report is useful for deciding that forecast-risk scoring should remain gated by better
calibration evidence. It does not prove navigation improvement, safety, human realism,
benchmark-strength predictor quality, or paper-facing/dissertation claims.
