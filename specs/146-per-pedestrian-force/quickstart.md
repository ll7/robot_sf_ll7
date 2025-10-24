# Quickstart: Per-Pedestrian Force Quantiles

## Run unit tests

```
uv run pytest tests/test_metrics.py -q
```

## Where to implement

- Add function `per_ped_force_quantiles(...)` to `robot_sf/benchmark/metrics.py`
- Add keys to `METRIC_NAMES` and call from `compute_all_metrics()`
- Update `docs/dev/issues/social-navigation-benchmark/metrics_spec.md`

## Expected behavior

- Returns NaN for all keys when there are no pedestrians
- For single pedestrian, per-ped quantiles equal that pedestrian's quantiles
- For multiple pedestrians, computes per-ped quantiles then mean across pedestrians
