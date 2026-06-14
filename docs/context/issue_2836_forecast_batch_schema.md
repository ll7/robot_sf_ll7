# Issue #2836: ForecastBatch.v1 Schema (2026-06-14)

`ForecastBatch.v1` is a benchmark artifact contract for exchanging pedestrian
forecast outputs without making prediction-quality, calibration, or planning
benefit claims. Those claims still require separate benchmark evidence.

The required provenance records predictor identity and family, observation tier,
coordinate frame, units, axis semantics, `dt_s`, forecast horizons in seconds,
scenario id, seed, fallback/degraded status, actor ids, actor mask metadata, and
feature schema. Positions are two-dimensional coordinates in meters. The
coordinate frame must name the frame and its axes so downstream evaluators do not
mix world-frame, robot-frame, or map-frame forecasts silently.

The payload is additive. Producers may emit deterministic trajectories, sampled
trajectories, mode probabilities, occupancy summaries, and uncertainty metadata.
They do not need to emit every optional field. Missing actor payloads are allowed
only when the actor mask and mask metadata describe the missingness semantics.

Oracle or deployable-oracle feature fields are rejected unless the artifact is
explicitly marked with `oracle_state=True`. This keeps deployable-observation
artifacts separate from diagnostic oracle-state artifacts.
