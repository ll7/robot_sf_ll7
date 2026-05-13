# Issue #1165 Predictive Obstacle-Feature Lifecycle

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1165>

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

The lifecycle path appends the stable six-element obstacle-feature schema. Until a later data-source
issue wires real map-line extraction into the predictive data collector/runtime, collected obstacle
rows use the extractor's unavailable-obstacle sentinel. That keeps the schema stable and prevents
silent input-dimension drift without overstating benchmark value.

## Validation

Targeted validation:

```bash
uv run pytest tests/test_predictive_model.py \
  tests/training/test_collect_predictive_planner_data.py \
  tests/training/test_train_predictive_planner_dataset_diagnostics.py \
  tests/benchmark/test_prediction_planner_audit_contract.py \
  tests/planner/test_predictive_obstacle_runtime_contract.py -q
```

Full PR readiness should use `origin/issue-1138-predictive-obstacle-features-v1` as the stacked
base while #1138 remains unmerged.

## Follow-up boundary

The same-seed performance comparison belongs to #1167 after this lifecycle contract lands. A future
collector/runtime issue can replace unavailable obstacle sentinels with map-derived obstacle-line
features without changing the schema name.
