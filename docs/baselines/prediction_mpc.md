# Prediction-Aware Model Predictive Control Planner

`prediction_mpc` is an experimental local planner for Robot SF map benchmarks. It uses the
structured Social Navigation (`socnav_state`) observation, predicts pedestrian futures with a
constant-velocity backend, and passes those futures to a short-horizon model predictive controller
as hard time-varying pedestrian clearance constraints.

Aliases:

- `prediction_mpc`
- `prediction_aware_mpc`
- `cv_prediction_mpc`

This first slice is not a learned-prediction result and does not change benchmark metrics. The
learned predictor backend remains deferred; missing learned-predictor assets must not silently
fallback unless a future configuration explicitly opts into that behavior.

## Configuration

Use the committed constant-velocity configuration:

```bash
configs/algos/prediction_mpc_cv.yaml
```

The planner is registered as experimental and requires:

```yaml
allow_testing_algorithms: true
```

## Smoke Command

```bash
uv run robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --out output/benchmarks/prediction_mpc_cv_smoke/episodes.jsonl \
  --algo prediction_mpc \
  --algo-config configs/algos/prediction_mpc_cv.yaml \
  --repeats 1 \
  --horizon 300 \
  --workers 1 \
  --no-video \
  --benchmark-profile experimental
```

Generated `output/` files are local validation artifacts, not durable benchmark evidence.
