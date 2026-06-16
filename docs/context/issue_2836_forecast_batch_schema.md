# Issue #2836 / #2913: ForecastBatch.v1 Schema (2026-06-14)

`ForecastBatch.v1` is a benchmark artifact contract for exchanging pedestrian
forecast outputs without making prediction-quality, calibration, or planning
benefit claims. Those claims still require separate benchmark evidence.

The required provenance records predictor identity and family, observation tier,
forecast origin `timestamp`, coordinate frame, units, axis semantics, `dt_s`,
forecast horizons in seconds, scenario id, seed, fallback/degraded status, actor
ids, actor mask metadata, and feature schema. Positions are two-dimensional
coordinates in meters. The coordinate frame must name the frame and its axes so
downstream evaluators do not mix world-frame, robot-frame, or map-frame forecasts
silently.

The payload is additive. Producers may emit deterministic point forecasts,
sampled trajectories, Gaussian parameters, mode probabilities, occupancy
summaries, reachable/conformal sets, and uncertainty metadata. They do not need
to emit every optional field. Missing actor payloads are allowed only when the
actor mask and mask metadata describe the missingness semantics.

Oracle or deployable-oracle feature fields are rejected unless the artifact is
explicitly marked with `oracle_state=True`. This keeps deployable-observation
artifacts separate from diagnostic oracle-state artifacts.

## JSON Schema

The canonical schema lives at
`robot_sf/benchmark/schemas/forecast_batch.schema.v1.json`. The
`ForecastBatchSchema` structural validator is in
`robot_sf/benchmark/schemas/forecast_batch_schema.py`; the domain dataclasses
and stricter typed loader are in `robot_sf/benchmark/forecast_batch.py`. A CLI
validator is available at `scripts/validation/validate_forecast_batch.py`.

## Minimal deterministic example

```json
{
  "schema_version": "ForecastBatch.v1",
  "provenance": {
    "predictor_id": "cv-baseline-v1",
    "predictor_family": "constant_velocity",
    "observation_tier": "tracked_agents",
    "timestamp": "2026-06-15T12:00:00Z",
    "frame": {"name": "world", "units": "m", "axes": ["x", "y"]},
    "dt_s": 0.5,
    "horizons_s": [0.5, 1.0],
    "scenario_id": "crosswalk_001",
    "seed": 7,
    "fallback_status": "native",
    "degraded_status": "none",
    "actor_ids": ["ped_1"],
    "actor_mask": [true],
    "actor_mask_metadata": {
      "semantics": "true means forecast payload is available for actor_id",
      "missing_actor_reasons": {}
    },
    "feature_schema": {"name": "socnav_observation_v1", "features": ["position_m"]}
  },
  "forecasts": [
    {"actor_id": "ped_1", "deterministic": [[1.0, 0.0], [1.5, 0.0]]}
  ]
}
```

## Probabilistic example with samples, Gaussian, and reachable set

```json
{
  "schema_version": "ForecastBatch.v1",
  "provenance": {
    "predictor_id": "gmm-forecast-v1",
    "predictor_family": "gaussian_mixture",
    "observation_tier": "tracked_agents",
    "timestamp": "2026-06-15T12:00:00Z",
    "frame": {"name": "world", "units": "m", "axes": ["x", "y"]},
    "dt_s": 0.5,
    "horizons_s": [0.5, 1.0],
    "scenario_id": "crosswalk_001",
    "seed": 7,
    "fallback_status": "native",
    "degraded_status": "none",
    "actor_ids": ["ped_1"],
    "actor_mask": [true],
    "actor_mask_metadata": {
      "semantics": "true means forecast payload is available for actor_id",
      "missing_actor_reasons": {}
    },
    "feature_schema": {"name": "socnav_observation_v1", "features": ["position_m"]}
  },
  "forecasts": [
    {
      "actor_id": "ped_1",
      "samples": [
        [[1.0, 0.0], [1.5, 0.0]],
        [[1.0, 0.1], [1.4, 0.2]]
      ],
      "mode_probabilities": [0.6, 0.4],
      "gaussian": [
        {"mean": [1.0, 0.0], "cov": [[0.1, 0.0], [0.0, 0.1]]},
        {"mean": [1.5, 0.0], "cov": [[0.2, 0.0], [0.0, 0.2]]}
      ],
      "reachable_set": [
        {"center": [1.0, 0.0], "radius_m": 0.5, "set_type": "conformal_tube"},
        {"center": [1.5, 0.0], "semi_axes_m": [0.6, 0.4], "set_type": "confidence_ellipse"}
      ]
    }
  ]
}
```
