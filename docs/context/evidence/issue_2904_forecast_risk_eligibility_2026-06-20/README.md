# Issue #2904 — Forecast-Risk Eligibility Fixtures

## Scope

Durable fixtures + regenerated evidence that unblock the forecast-risk calibration filter. Before
this change the canonical calibration report (Issue #2865) contained only
`actor_class=unavailable` rows on a single observation tier (`deployable_tracked`), so the filter
returned `wait`/blocked for lack of risk-scoring-eligible rows. These fixtures add calibrated rows
with explicit actor classes across two observation tiers so the calibration filter becomes
meaningful.

This is diagnostic / analysis-only evidence. It does **not** claim forecast-variant superiority,
planner safety, navigation benefit, or benchmark/paper-grade quality.

## Source fixtures (`ForecastMetrics.v1`)

- [`metric_pedestrian_deployable_tracked.json`](../../../../tests/fixtures/benchmark/forecast_risk_eligibility/metric_pedestrian_deployable_tracked.json)
  — `actor_class=pedestrian`, `observation_tier=deployable_tracked`, coverage `0.91`, denominator `12`.
- [`metric_bicycle_deployable_observation.json`](../../../../tests/fixtures/benchmark/forecast_risk_eligibility/metric_bicycle_deployable_observation.json)
  — `actor_class=bicycle`, `observation_tier=deployable_observation`, coverage `0.88`, denominator `8`.

Both rows are calibrated within tolerance (coverage in `[0.85, 0.95]` around target `0.90`),
carry `semantic_metadata_present="present"`, and therefore resolve to
`risk_scoring_eligibility=eligible_analysis_only`.

## Artifacts in this bundle

- [`calibration_report.json`](calibration_report.json) / [`calibration_report.md`](calibration_report.md)
  — `ForecastCalibrationReport.v1`; `recommendation.decision=continue` (`claim_status=analysis-only`),
  2 reliability rows, 0 limitation rows.
- [`report.json`](report.json) / [`report.md`](report.md) / [`summary.json`](summary.json)
  — `forecast_risk_calibration_filter.diagnostic_comparison.v1`; all five modes `available`,
  overall `recommendation=diagnostic_only`.

## Coverage achieved (Definition of Done)

| Requirement | Result |
| --- | --- |
| Pedestrian row exposes `actor_class` | `pedestrian` (denominator 12) |
| Bicycle/dynamic-agent row exposes `actor_class` | `bicycle` (denominator 8) |
| ≥2 observation tiers | `deployable_tracked`, `deployable_observation` |
| Risk-scoring-eligible rows | 2 × `eligible_analysis_only` |
| Calibration-filtered mode not blocked | `available`; overall `diagnostic_only` (was `wait`) |

## Reproducible commands

```bash
# 1. Build the calibration report from the durable fixtures.
uv run python scripts/benchmark/build_forecast_calibration_report.py \
  tests/fixtures/benchmark/forecast_risk_eligibility/metric_pedestrian_deployable_tracked.json \
  tests/fixtures/benchmark/forecast_risk_eligibility/metric_bicycle_deployable_observation.json \
  --report-id issue_2904_forecast_risk_eligibility_2026-06-20 \
  --generated-at-utc 2026-06-20T00:00:00+00:00 \
  --out-json docs/context/evidence/issue_2904_forecast_risk_eligibility_2026-06-20/calibration_report.json \
  --out-md docs/context/evidence/issue_2904_forecast_risk_eligibility_2026-06-20/calibration_report.md

# 2. Run the forecast-risk calibration filter against the regenerated report.
uv run python scripts/validation/validate_forecast_risk_calibration_filter.py \
  --calibration-report docs/context/evidence/issue_2904_forecast_risk_eligibility_2026-06-20/calibration_report.json \
  --out-dir docs/context/evidence/issue_2904_forecast_risk_eligibility_2026-06-20
```

## Note on the filter gate

`scripts/validation/validate_forecast_risk_calibration_filter.py::_any_eligible_for_risk_scoring`
previously matched only the legacy tokens `{"eligible", "calibrated"}`, which the canonical
calibration report never emits — it emits `eligible_analysis_only`. The gate now also recognizes
`eligible_analysis_only`, so eligible rows actually unblock the `calibration_filtered` mode. The
legacy tokens are retained for backward compatibility. This is an eligibility repair only; it does
not change `_risk_scoring_eligibility`, calibration thresholds, or planner risk coupling
(`calibration_filtered` still runs with `forecast_risk_weight=0.0`).

## Validation

```bash
uv run pytest tests/benchmark/test_forecast_risk_eligibility_fixtures.py
uv run pytest tests/validation/test_forecast_risk_calibration_filter.py
```

## Claim boundary

Diagnostic-only. Forecast-risk scoring remains opt-in with `forecast_risk_weight=0.0` by default.
This bundle does not establish safety, navigation benefit, human realism, benchmark-strength
predictor quality, or paper/dissertation claims.
