# Issue #593 Predictive Ego-Conditioned v2 Note

## Summary

`prediction_planner` already had most of the ego-conditioned v2 path implemented:

- planner inference accepted 9D per-agent state when the predictor model advertised `input_dim=9`
- the training pipeline already propagated `--ego-conditioning`
- ego-conditioned model artifacts already existed locally

The missing piece was the standalone rollout collector in
`scripts/training/collect_predictive_planner_data.py`, which still emitted only the legacy 4D
state contract. This issue closes that gap so the base collector and hardcase collector now expose
the same feature width and CLI surface.

## Implemented Delta

The standalone collector now:

- records `robot_speed`
- records `goal_current`
- supports `--ego-conditioning`
- emits 9D state rows when enabled
- records `state_dim` and `ego_conditioning` in the dataset summary sidecar

The 9D state layout matches the existing planner/training path:

1. pedestrian relative position `(x_rel, y_rel)`
2. pedestrian relative velocity `(vx_rel, vy_rel)`
3. robot planar self velocity `(v_x, v_y)`
4. normalized goal direction `(goal_dir_x, goal_dir_y)`
5. goal distance `(goal_dist)`

## Proof

### Code-path proof

Validated with:

```bash
uv run pytest -q \
  tests/training/test_collect_predictive_planner_data.py \
  tests/test_predictive_model.py \
  tests/training/test_run_predictive_training_pipeline.py \
  -k 'ego_condition or predictive_model_forward_accepts_ego_conditioned_state'
```

This covers:

- standalone collector 4D legacy compatibility
- standalone collector 9D ego-conditioned samples
- predictive model forward support for `input_dim=9`
- pipeline propagation of `--ego-conditioning`

### Existing artifact evidence

Local artifacts already show the ego-conditioned predictor substantially improved forecasting
quality but did not yet improve hard-seed navigation outcomes:

| Run | Model ID | Val ADE | Val FDE | Hard-seed success/failure |
| --- | --- | ---: | ---: | --- |
| baseline full | `predictive_proxy_selected_v2_full` | `0.0629` | `0.1155` | `1 success / 6 failure` |
| xl ego | `predictive_proxy_selected_v2_xl_ego` | `0.0162` | `0.0314` | `1 success / 6 failure` |

Sources:

- `output/tmp/predictive_planner/pipeline/predictive_br07_all_maps_randomized_full_20260306T210912Z/training/training_summary.json`
- `output/tmp/predictive_planner/pipeline/predictive_br07_all_maps_randomized_xl_ego_20260308T145000Z/training/training_summary.json`
- corresponding `final_performance_summary.json` files in the same run roots

## Interpretation

The repository now has a complete config-first ego-conditioned v2 path from collection through
training and inference. That closes a reproducibility gap.

It does **not** prove benchmark promotion yet:

- supervised trajectory quality improved sharply
- hard-seed planner outcomes did not improve materially in the existing runs

So the correct current claim is:

- `prediction_planner` v2 ego conditioning is implemented and reproducible
- planner-quality improvement remains unproven and still requires benchmark-facing evaluation

## Follow-up

The next valuable predictive-planner step is not more plumbing. It is a benchmark-facing rerun or
promotion study that tests whether the improved predictor can actually raise success or reduce
collisions on the hard-seed and canonical benchmark surfaces.
