# Issue #2941 Native Forecast-Variant Replay Smoke

## Claim Boundary

Issue #2941 native forecast-variant replay.  The gate evaluates the none and cv forecast variants on the same recorded trace using a minimal forecast-aware brake replay policy.  A BaselineProbabilisticPredictor implementation and a forecast_variant config key are now present, so the run is classified as native when cv closed-loop metrics differ from the integrated no-forecast baseline, and degraded when they match.  It does not claim that cv improves safety, success, or runtime in a full planner stack.

## Classification

- **Classification:** native
- **Reason:** cv forecast produces closed-loop metrics that differ from the integrated no-forecast baseline
- **Full-matrix expansion recommended:** True

## Provenance

- **Trace:** issue_2765_dense_pedestrian_stress_seed2765_ep0000
- **Scenario:** dense_pedestrian_stress
- **Seed:** 2765
- **Planner:** hybrid_rule_v0_minimal
- **Repo HEAD:** `e52f0323`
- **Generated at (UTC):** 2026-06-16T00:00:00+00:00
- **Variants:** ['none', 'cv', 'semantic', 'interaction_aware', 'risk_filtered']

## Native Path Eligibility

- **Live path available:** True
- **Predictor implementations:** 1
- **Forecast variant config key present:** True

## Recorded Trace Closed-Loop Metrics

- **Collision:** True
- **Near miss timesteps:** 9
- **Min distance (m):** 0.15620499351813316
- **Stop/yield steps:** 0
- **Progress (m):** 1.9
- **False-positive stops:** 0
- **Runtime (s):** 1.9

## Variant Results

| Variant | Actors | Forecast Metrics | Closed-Loop Source |
|---|---|---|---|
| none | 0 | not_applicable | no_forecast_replay |
| cv | 3 | ok | forecast_brake_replay |
| semantic | 3 | ok | forecast_brake_replay |
| interaction_aware | 3 | ok | forecast_brake_replay |
| risk_filtered | 3 | ok | forecast_brake_replay |

## Limitations
- Non-none variants use a minimal forecast-brake replay policy, not a production planner.  The gate proves the forecast can influence closed-loop metrics but does not prove benefit in a full planner stack.
- Open-loop forecast metrics are computed from a single frame per trace by default.
- Full-matrix expansion is gated by the run classification; only native runs should expand to the full variant matrix.
