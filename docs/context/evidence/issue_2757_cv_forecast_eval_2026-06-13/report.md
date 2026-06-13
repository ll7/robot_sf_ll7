# CV Forecast Baseline Evaluation

## Claim Boundary

**Diagnostic-only, not paper-facing evidence.** This evaluates the constant-velocity Gaussian forecast baseline on a bounded set of existing repository trace fixtures. Results are per-family rollups, not statistically powered population claims.

## Reproducibility

- **Issue:** #2757
- **Generated at (UTC):** 2026-06-13T09:40:45.509121+00:00
- **Command:** `uv run python scripts/benchmark/run_cv_forecast_eval.py --output-dir docs/context/evidence/issue_2757_cv_forecast_eval_2026-06-13`
- **Repo HEAD:** `2a87f526`
- **Forecast horizons:** [0.5, 1.0, 2.0]

## Selected Trace Families

| Family | Label | Scenario | Frames | Peds | Motion | dt (s) | Status |
|--------|-------|----------|--------|------|--------|--------|--------|
| corridor_interaction | default_social_force | classic_head_on_corridor_low | 20 | 2 | yes | 0.1 | evaluated |
| corridor_interaction | ammv_social_force | classic_head_on_corridor_low | 20 | 2 | yes | 0.1 | evaluated |
| crossing_proxy | synthetic_crossing_proxy_orca | crossing_proxy | 5 | 1 | no | - | limited_no_pedestrian_motion |
| bottleneck | minimal_fixture | classic_bottleneck_medium | 2 | 1 | no | - | limited_no_pedestrian_motion |

## Missing Trace Families

These scenario families have no durable trace fixtures in the repository and were not evaluated:

- **signalized_crossing**: no durable trace fixture available
- **occluded_emergence**: no durable trace fixture available
- **dense_pedestrian_interaction**: no durable trace fixture available
- **bottleneck_with_motion**: existing bottleneck fixture has zero pedestrian velocity

## Aggregate Metrics by Trace Family

### corridor_interaction / default_social_force

- Evaluable samples: 30
- **Horizon 0.5s:**
  - ADE: 0.0243 m
  - NLL: 1.2677
  - Miss rate: 0.00%
  - Within 95% CI: 100.00%
  - Calibration error: 0.0500
- **Horizon 1s:**
  - ADE: 0.0769 m
  - NLL: 1.9522
  - Miss rate: 0.00%
  - Within 95% CI: 100.00%
  - Calibration error: 0.0500
- **Horizon 2s:**
  - Not available for this trace length.

### corridor_interaction / ammv_social_force

- Evaluable samples: 30
- **Horizon 0.5s:**
  - ADE: 0.0243 m
  - NLL: 1.2677
  - Miss rate: 0.00%
  - Within 95% CI: 100.00%
  - Calibration error: 0.0500
- **Horizon 1s:**
  - ADE: 0.0769 m
  - NLL: 1.9522
  - Miss rate: 0.00%
  - Within 95% CI: 100.00%
  - Calibration error: 0.0500
- **Horizon 2s:**
  - Not available for this trace length.


## Interpretation

The constant-velocity Gaussian baseline is adequate for short horizons (0.5s) on traces with relatively smooth, low-acceleration pedestrian motion (such as the corridor_interaction family). In this bounded evidence set, 1s metrics remain available and low-error, while 2s metrics are unavailable because the durable traces are too short for that forecast horizon.

The available crossing_proxy and bottleneck fixtures are limitation records, not forecast-quality evidence: they contain zero pedestrian motion, so they do not test whether constant velocity handles crossing, bottleneck, occluded, or dense interaction dynamics.

All traces in this evaluation lack intent and signal context metadata, so the forecast operates in 'uncertain' mode with 1.5x widened standard deviation. This is a systematic limitation of the available trace fixtures.