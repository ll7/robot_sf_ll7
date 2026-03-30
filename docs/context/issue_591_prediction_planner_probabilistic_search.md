# Issue 591: Prediction Planner Probabilistic Search

## Goal

Add first-class probabilistic forecasting and search modes to `prediction_planner` without
breaking the deterministic baseline:

- heuristic probabilistic rollout sampling over the existing predictive model,
- risk-aware aggregation via mean/CVaR,
- an optional bounded MCTS-lite search path,
- and benchmark metadata that records uncertainty/search mode explicitly.

The proof obligation for this issue was not just code-level determinism. The new modes had to run
on the canonical paper surface and show whether they improve the current predictive planner tradeoff
in this repository.

## Scope Implemented

- Added heuristic probabilistic future sampling to
  `robot_sf/planner/socnav.py::PredictionPlannerAdapter`.
- Added risk aggregation for:
  - `mean`
  - `cvar`
- Added optional bounded `predictive_mcts_enabled` search mode.
- Added explicit benchmark metadata exposure in `robot_sf/benchmark/map_runner.py`:
  - `prediction_mode`
  - `predictive_uncertainty_mode`
  - `predictive_risk_objective`
  - `predictive_risk_sample_count`
  - `predictive_search_mode`
- Added benchmark configs:
  - `configs/algos/prediction_planner_probabilistic_cvar.yaml`
  - `configs/algos/prediction_planner_mcts_lite.yaml`
  - `configs/benchmarks/paper_experiment_matrix_v1_prediction_planner_probabilistic_compare.yaml`

## Validation

Code-level validation:

- `uv run pytest -q tests/test_socnav_planner_adapter.py -k 'prediction_adapter' tests/benchmark/test_prediction_planner_audit_contract.py`
- `uv run ruff check robot_sf/planner/socnav.py robot_sf/benchmark/map_runner.py tests/test_socnav_planner_adapter.py tests/benchmark/test_prediction_planner_audit_contract.py`

Benchmark proof:

- `uv run python scripts/tools/run_camera_ready_benchmark.py --config configs/benchmarks/paper_experiment_matrix_v1_prediction_planner_probabilistic_compare.yaml --label issue591_prediction_planner_probabilistic_compare --log-level WARNING`

Canonical artifact root:

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_prediction_planner_probabilistic_compare_issue591_prediction_planner_probabilistic_compare_20260321_192353`

## Result

### Canonical paper-surface comparison

| planner | success | collision | timeout | runtime (s) | SNQI |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prediction_planner_v2_xl_ego` | `0.0709` | `0.2270` | `0.7021` | `334.5553` | `-0.1635` |
| `prediction_planner_probabilistic_cvar` | `0.0567` | `0.2199` | `0.7234` | `1359.5806` | `-0.1611` |
| `prediction_planner_mcts_lite` | `0.0567` | `0.2199` | `0.7234` | `1722.4329` | `-0.1597` |

### Interpretation

- Both probabilistic modes are slightly safer than the deterministic XL-ego baseline on collision
  termination rate.
- Both probabilistic modes also improve SNQI slightly.
- Neither mode improves success; both lose goal-reaching relative to the deterministic baseline.
- Both modes are materially slower:
  - CVaR is about `4.1x` slower.
  - MCTS-lite is about `5.1x` slower.

## Assessment

This issue succeeded as an implementation and auditability improvement, not as a planner-quality
upgrade.

What changed for the better:

- the benchmark can now distinguish deterministic vs probabilistic predictive planner runs,
- risk objective and search mode are explicit in algorithm metadata,
- the predictive planner now supports a bounded probabilistic search family without retraining.

What did not improve:

- the new modes did not beat the current deterministic predictive baseline on the canonical paper
  surface,
- and they are too expensive to justify as default benchmark replacements.

## Conclusion

`prediction_planner_probabilistic_cvar` and `prediction_planner_mcts_lite` should remain
`experimental` only.

The observed evidence does **not** support promoting either mode over
`prediction_planner_v2_xl_ego` in the benchmark.

## Remaining Risk

The current probabilistic layer is heuristic rather than checkpoint-native. That was intentional for
this issue, but it means the uncertainty model itself is still an approximation on top of a
deterministic predictor rather than a learned uncertainty estimate.
