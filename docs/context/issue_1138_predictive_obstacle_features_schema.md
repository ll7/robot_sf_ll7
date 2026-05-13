# Issue 1138 predictive obstacle feature schema

Date: 2026-05-12

## Scope implemented so far

This branch adds the first shared deterministic obstacle-feature extractor and schema metadata helpers for the predictive planner baseline. It does not yet wire the features into predictive data collection, training, checkpoint loading, or runtime planner inference.

Implemented files:

- `robot_sf/planner/obstacle_features.py`
- `tests/planner/test_obstacle_features.py`

## Feature schema

Schema name: `predictive_obstacle_features_v1`

Feature dimension: `6`

Feature order:

```text
[distance, normal_x, normal_y, tangent_x, tangent_y, valid_mask]
```

Semantics:

- `distance`: Euclidean distance from the query point to the nearest obstacle line segment.
- `normal_x`, `normal_y`: unit vector from nearest obstacle point toward the query point, or zeros for degenerate overlap.
- `tangent_x`, `tangent_y`: unit vector along the nearest obstacle segment.
- `valid_mask`: `1.0` when obstacle geometry is available, `0.0` for missing/unavailable geometry.

Unavailable map/obstacle geometry uses deterministic sentinel behavior:

```text
[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

The unavailable distance is configurable on `LocalObstacleFeatureExtractor` but must remain finite
and non-negative.

## Determinism

Nearest obstacle selection is deterministic. Equal-distance ties are broken by input line order.

## Schema metadata

`ObstacleFeatureSchema.to_metadata()` returns JSON-compatible metadata:

```json
{"name": "predictive_obstacle_features_v1", "feature_dim": 6}
```

`ObstacleFeatureSchema.validate_metadata(...)` fails closed on schema-name or dimension drift with `ObstacleFeatureSchemaError`.

## Remaining work

Deferred follow-up issues:

- `#1165` predictive planner: wire obstacle features through data, training, and runtime
- `#1167` predictive planner: compare obstacle-feature baseline on same seeds

- Add one shared call path in predictive data collection.
- Extend predictive dataset files/checkpoint metadata with `obstacle_feature_schema`.
- Update training/evaluation model input dimensions using the same schema metadata.
- Update runtime planner inference to consume the same feature rows.
- Add config-first path such as `predictive_obstacle_features_v1`.
- Add integration tests for collection/training/runtime schema consistency.
- Prove checkpoint/config dimension mismatches fail closed in the real loader path.

## Validation status

Validation commands run on this branch:

- `uv run pytest tests/planner/test_obstacle_features.py -q`
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`

Both passed for the PR branch before this note was refreshed; rerun them after any follow-up edits.

## Additional foundation: feature composition

Implemented artifact:

- `robot_sf/planner/obstacle_features.py::append_obstacle_features`

Design:

- Validates `ObstacleFeatureSchema` metadata before concatenating features.
- Requires agent feature rows and obstacle feature rows to have matching row counts.
- Requires obstacle feature width to match `PREDICTIVE_OBSTACLE_FEATURE_DIM`.
- Returns a float32 array with obstacle features appended to each agent row.

This is a lifecycle wiring primitive; the collector/trainer/runtime still need to call it from their concrete paths.
