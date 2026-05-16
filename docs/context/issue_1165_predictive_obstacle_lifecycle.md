# Issue #1165 Predictive Obstacle-Feature Lifecycle

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1165>
- <https://github.com/ll7/robot_sf_ll7/issues/1218>

## Decision

Wire `predictive_obstacle_features_v1` through the predictive-planner lifecycle as a schema and
dimension contract before running any training campaign or benchmark-promotion comparison.

This issue does not claim that obstacle features improve performance. It only makes dataset,
training, checkpoint, and runtime loading fail closed when feature schemas or input dimensions do
not match.

## Implemented contract

- Datasets can record `feature_schema_json` in `.npz` payloads and a sibling
  `.npz.manifest.json` with model-family/schema metadata.
- Training loads schema metadata from dataset payloads or manifests instead of guessing input
  dimension from CLI flags.
- `PredictiveModelConfig` stores `feature_schema_name`.
- Checkpoints store `feature_schema` and validate it against model `input_dim`.
- Runtime config carries `predictive_feature_schema_name`; checkpoint loading rejects mismatched
  schemas before inference.
- Legacy 4D and ego-conditioned 9D checkpoints remain loadable through inferred legacy schema
  metadata.

## Current obstacle-feature data source

The lifecycle path appends the stable six-element obstacle-feature schema. As of #1218, the
collector and runtime adapter can use real static map geometry:

- `robot_sf.planner.obstacle_features.obstacle_lines_from_map(...)` normalizes
  `MapDefinition.obstacles` and `MapDefinition.bounds` into deterministic line segments.
- `scripts/training/collect_predictive_planner_data.py` passes those map-derived lines into
  `LocalObstacleFeatureExtractor` when collecting `predictive_obstacle_features_v1` datasets.
- `robot_sf.planner.socnav.PredictionPlannerAdapter.bind_env(...)` binds live benchmark
  environment map geometry for runtime feature construction, while explicit observation
  `map.obstacle_lines` payloads can override it.
- Missing or malformed geometry still degrades to the explicit unavailable-obstacle sentinel:
  `[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]`.

This is a data-source wiring change, not a benchmark-performance claim.

## Validation

Targeted validation:

```bash
uv run pytest tests/test_predictive_model.py \
  tests/training/test_collect_predictive_planner_data.py \
  tests/training/test_train_predictive_planner_dataset_diagnostics.py \
  tests/benchmark/test_prediction_planner_audit_contract.py \
  tests/planner/test_predictive_obstacle_runtime_contract.py -q
```

Issue #1218 Targeted Validation on 2026-05-15:

```bash
rtk uv run pytest tests/planner/test_obstacle_features.py \
  tests/training/test_collect_predictive_planner_data.py \
  tests/planner/test_predictive_obstacle_runtime_contract.py -q
```

Result: `25 passed`.

Collector smoke on 2026-05-15:

```bash
rtk uv run python scripts/training/collect_predictive_planner_data.py \
  --episodes 1 \
  --max-steps 8 \
  --horizon-steps 1 \
  --max-agents 8 \
  --model-family predictive_obstacle_features_v1 \
  --min-goal-distance 0 \
  --max-reset-attempts 3 \
  --output output/tmp/issue1218/predictive_obstacle_smoke.npz
```

Observed output: `7` samples, state shape `[7, 8, 10]`, `obstacle_feature_source:
map_geometry`, `obstacle_line_count: 862`, and `56` obstacle rows with `valid_mask=1.0`.
The first valid obstacle row was approximately `[4.9238, 0.3212, 0.9470, 0.9997, -0.0258, 1.0]`.

Full PR readiness should use `origin/main`; #1138 has landed via #1160, so this branch is no
longer stacked on a feature base.

## Follow-up boundary

The same-seed performance comparison belongs to #1167. This note records executable evidence that
real map-derived features are available, but #1167 remains blocked until it runs its own
same-seed comparison and does not inherit a benchmark-success claim from this wiring proof.
