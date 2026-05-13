"""Tests for deterministic local obstacle features."""

import numpy as np

from robot_sf.planner.obstacle_features import (
    PREDICTIVE_OBSTACLE_FEATURE_DIM,
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    LocalObstacleFeatureExtractor,
    ObstacleFeatureSchema,
    ObstacleFeatureSchemaError,
    append_obstacle_features,
)


def test_obstacle_feature_schema_and_shape():
    """Extractor should expose stable schema metadata and feature width."""
    extractor = LocalObstacleFeatureExtractor()

    features = extractor.extract((1.0, 1.0), [((0.0, 0.0), (2.0, 0.0))])

    assert extractor.schema == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA
    assert extractor.feature_dim == PREDICTIVE_OBSTACLE_FEATURE_DIM
    assert features.shape == (6,)


def test_obstacle_feature_nearest_distance_normal_and_tangent():
    """Feature should describe the nearest obstacle segment deterministically."""
    extractor = LocalObstacleFeatureExtractor()

    features = extractor.extract((1.0, 1.0), [((0.0, 0.0), (2.0, 0.0))])

    np.testing.assert_allclose(features, [1.0, 0.0, 1.0, 1.0, 0.0, 1.0])


def test_obstacle_feature_unavailable_map_uses_sentinel_mask():
    """Missing obstacle geometry should produce sentinel distance and mask=0."""
    extractor = LocalObstacleFeatureExtractor(unavailable_distance=-1.0)

    features = extractor.extract((1.0, 1.0), [])

    np.testing.assert_allclose(features, [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_obstacle_feature_many_reuses_same_schema():
    """Batch extraction should return one fixed-width row per query point."""
    extractor = LocalObstacleFeatureExtractor()

    features = extractor.extract_many(
        [(1.0, 1.0), (3.0, 0.0)],
        [((0.0, 0.0), (2.0, 0.0))],
    )

    assert features.shape == (2, PREDICTIVE_OBSTACLE_FEATURE_DIM)
    np.testing.assert_allclose(features[:, -1], [1.0, 1.0])


def test_obstacle_feature_tie_breaks_by_input_order():
    """Equal-distance obstacle ties should be stable by line order."""
    extractor = LocalObstacleFeatureExtractor()

    features = extractor.extract(
        (0.0, 0.0),
        [
            ((-1.0, 1.0), (1.0, 1.0)),
            ((-1.0, -1.0), (1.0, -1.0)),
        ],
    )

    np.testing.assert_allclose(features, [1.0, 0.0, -1.0, 1.0, 0.0, 1.0])


def test_obstacle_feature_schema_metadata_validates_expected_values():
    """Schema metadata should round-trip for datasets and checkpoints."""
    schema = ObstacleFeatureSchema()

    schema.validate_metadata(schema.to_metadata())

    assert LocalObstacleFeatureExtractor().schema_metadata == {
        "name": PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
        "feature_dim": PREDICTIVE_OBSTACLE_FEATURE_DIM,
    }


def test_obstacle_feature_schema_metadata_fails_closed_on_dimension_mismatch():
    """Dimension drift should raise an actionable schema error."""
    schema = ObstacleFeatureSchema()

    try:
        schema.validate_metadata({"name": PREDICTIVE_OBSTACLE_FEATURE_SCHEMA, "feature_dim": 7})
    except ObstacleFeatureSchemaError as exc:
        assert "dimension mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ObstacleFeatureSchemaError")


def test_obstacle_feature_schema_metadata_fails_closed_on_name_mismatch():
    """Schema-name drift should raise an actionable schema error."""
    schema = ObstacleFeatureSchema()

    try:
        schema.validate_metadata({"name": "legacy", "feature_dim": PREDICTIVE_OBSTACLE_FEATURE_DIM})
    except ObstacleFeatureSchemaError as exc:
        assert "schema mismatch" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ObstacleFeatureSchemaError")


def test_append_obstacle_features_concatenates_validated_rows():
    """Agent and obstacle rows should concatenate only after schema validation."""
    agent_features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    obstacle_features = np.asarray(
        [[1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [2.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
        dtype=np.float32,
    )

    combined = append_obstacle_features(
        agent_features,
        obstacle_features,
        schema_metadata=ObstacleFeatureSchema().to_metadata(),
    )

    assert combined.shape == (2, 8)
    np.testing.assert_allclose(combined[:, :2], agent_features)
    np.testing.assert_allclose(combined[:, 2:], obstacle_features)


def test_append_obstacle_features_rejects_row_count_mismatch():
    """Feature composition should fail closed on agent/obstacle row drift."""
    try:
        append_obstacle_features(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((1, PREDICTIVE_OBSTACLE_FEATURE_DIM), dtype=np.float32),
            schema_metadata=ObstacleFeatureSchema().to_metadata(),
        )
    except ValueError as exc:
        assert "same number of rows" in str(exc)
    else:  # pragma: no cover - defensive assertion style for clearer failure
        raise AssertionError("expected ValueError")
