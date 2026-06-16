# Issue #2902 Live Same-Seed Forecast Replay Gate

## Claim Boundary

Diagnostic-only same-seed forecast replay gate. Forecast variants are evaluated open-loop against the recorded trace; closed-loop metrics are the baseline trace metrics and do not vary by variant because no native planner consumes selectable baseline forecasts.

## Provenance

- **Trace:** issue_2765_dense_pedestrian_stress_seed2765_ep0000
- **Scenario:** dense_pedestrian_stress
- **Seed:** 2765
- **Planner:** hybrid_rule_v0_minimal
- **Repo HEAD:** `78fed01e`
- **Generated at (UTC):** 2026-06-16T10:08:46.851992+00:00

## Native Path Eligibility

- **Live path available:** False
- **Predictor implementations:** 0
- **Forecast variant config key present:** False

### Missing Components
- no ProbabilisticPredictor implementation registered for baseline forecast variants
- no environment or planner config key for selecting forecast_variant

## Baseline Closed-Loop Metrics (none variant)

- **Collision:** True
- **Near miss timesteps:** 9
- **Min distance (m):** 0.15620499351813316
- **Stop/yield steps:** 0
- **Progress (m):** 1.9
- **False-positive stops:** 0
- **Runtime (s):** 1.9

## Variant Results

| Variant | Actors | Forecast Metrics |
|---|---|---|
| none | 0 | not_applicable |
| cv | 3 | ok |
| semantic | 3 | ok |
| interaction_aware | 3 | ok |
| risk_filtered | 3 | ok |

## Limitations
- Closed-loop metrics are invariant across variants because the repository does not yet expose a planner that consumes ProbabilisticPredictor baseline variants.
- Open-loop forecast metrics are computed from a single frame per trace by default.
- Risk-filtered variant uses a deterministic distance-based relevance filter.
