# Issue #2944 Native CV-Only Closed-Loop Replay Smoke

## Claim Boundary

Issue #2944 native CV-only closed-loop replay smoke.  The gate evaluates the none and cv forecast variants on the same recorded trace.  Because the repository does not yet expose a planner that consumes selectable baseline forecast variants, closed-loop metrics are copied from the recorded trace and the run is classified fail-closed as blocked, degraded, or diagnostic_only.  It does not claim that cv improves safety, success, or runtime.

## Classification

- **Classification:** blocked
- **Reason:** native live path blocked: no ProbabilisticPredictor implementation registered for baseline forecast variants; no environment or planner config key for selecting forecast_variant
- **Full-matrix expansion recommended:** False

## Provenance

- **Trace:** issue_2765_dense_pedestrian_stress_seed2765_ep0000
- **Scenario:** dense_pedestrian_stress
- **Seed:** 2765
- **Planner:** hybrid_rule_v0_minimal
- **Repo HEAD:** `b3adfbb6`
- **Generated at (UTC):** 2026-06-16T11:05:00+00:00
- **Variants:** ['none', 'cv']

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

| Variant | Actors | Forecast Metrics | Closed-Loop Source |
|---|---|---|---|
| none | 0 | not_applicable | baseline_recorded_trace |
| cv | 3 | ok | baseline_recorded_trace |

## Limitations
- Closed-loop metrics are copied from the recorded trace for all variants because the repository does not yet expose a planner that consumes ProbabilisticPredictor baseline variants.
- Open-loop forecast metrics are computed from a single frame per trace by default.
- Full-matrix expansion is gated by the run classification; only native runs should expand to the full variant matrix.
