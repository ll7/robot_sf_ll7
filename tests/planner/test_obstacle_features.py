"""Tests for deterministic local obstacle features."""

import numpy as np

from robot_sf.nav.obstacle import Obstacle
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_OBSTACLE_FEATURE_DIM,
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    LocalObstacleFeatureExtractor,
    ObstacleFeatureSchema,
    ObstacleFeatureSchemaError,
    append_obstacle_features,
    obstacle_lines_from_map,
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


def test_obstacle_feature_unavailable_map_uses_default_sentinel_mask():
    """Missing obstacle geometry should use the v1 default sentinel distance and mask=0."""
    extractor = LocalObstacleFeatureExtractor()

    features = extractor.extract((1.0, 1.0), [])

    np.testing.assert_allclose(features, [50.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_obstacle_feature_unavailable_map_allows_custom_sentinel_distance():
    """Callers may override the default unavailable-distance sentinel when needed."""
    extractor = LocalObstacleFeatureExtractor(unavailable_distance=25.0)

    features = extractor.extract((1.0, 1.0), [])

    np.testing.assert_allclose(features, [25.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_obstacle_feature_unavailable_distance_must_be_finite_non_negative():
    """Unavailable-distance sentinel values should preserve finite v1 schema semantics."""
    for unavailable_distance in (-1.0, float("nan"), float("inf")):
        try:
            LocalObstacleFeatureExtractor(unavailable_distance=unavailable_distance)
        except ValueError as exc:
            assert "finite, non-negative" in str(exc)
        else:  # pragma: no cover - defensive assertion style for clearer failure
            raise AssertionError("expected ValueError")


def test_obstacle_feature_many_reuses_same_schema():
    """Batch extraction should return one fixed-width row per query point."""
    extractor = LocalObstacleFeatureExtractor()

    features = extractor.extract_many(
        [(1.0, 1.0), (3.0, 0.0)],
        [((0.0, 0.0), (2.0, 0.0))],
    )

    assert features.shape == (2, PREDICTIVE_OBSTACLE_FEATURE_DIM)
    np.testing.assert_allclose(features[:, -1], [1.0, 1.0])


def test_obstacle_feature_many_empty_input_keeps_feature_width():
    """Empty batch extraction should preserve the fixed feature width."""
    extractor = LocalObstacleFeatureExtractor()

    features = extractor.extract_many([], [((0.0, 0.0), (2.0, 0.0))])

    assert features.shape == (0, PREDICTIVE_OBSTACLE_FEATURE_DIM)
    assert features.dtype == np.float32


def test_obstacle_lines_from_map_converts_obstacles_and_legacy_flat_bounds():
    """Map geometry should preserve obstacle/bound line order for predictive features."""
    map_def = type(
        "MapDef",
        (),
        {
            "obstacles": [Obstacle([(0.0, 2.0), (2.0, 2.0), (2.0, 3.0), (0.0, 3.0)])],
            "bounds": [(0.0, 4.0, 0.0, 0.0)],
        },
    )()

    lines = obstacle_lines_from_map(map_def)

    assert lines[0] == ((0.0, 2.0), (2.0, 2.0))
    assert lines[-1] == ((0.0, 0.0), (4.0, 0.0))
    features = LocalObstacleFeatureExtractor().extract((1.0, 1.0), lines)
    np.testing.assert_allclose(features, [1.0, 0.0, -1.0, 1.0, 0.0, 1.0])


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


def test_vectorized_extract_many_matches_scalar_parity():
    """Vectorized batch extraction must match scalar per-point results exactly."""
    extractor = LocalObstacleFeatureExtractor()
    lines = [
        ((0.0, 0.0), (2.0, 0.0)),
        ((2.0, 0.0), (2.0, 2.0)),
        ((2.0, 2.0), (0.0, 2.0)),
        ((0.0, 2.0), (0.0, 0.0)),
    ]
    query_points = [
        (1.0, 1.0),
        (0.5, 0.5),
        (1.5, 1.5),
        (3.0, 1.0),
        (0.0, 0.0),
    ]

    # Scalar reference
    scalar_rows = [extractor.extract(p, lines) for p in query_points]
    scalar = np.asarray(scalar_rows, dtype=np.float32)

    # Vectorized batch
    vectorized = extractor.extract_many(query_points, lines)

    np.testing.assert_array_equal(vectorized, scalar)


def test_vectorized_extract_many_tie_breaks_by_input_order():
    """Vectorized tie-breaking must match scalar: lowest index wins on equal distance."""
    extractor = LocalObstacleFeatureExtractor()
    lines = [
        ((-1.0, 1.0), (1.0, 1.0)),
        ((-1.0, -1.0), (1.0, -1.0)),
    ]
    query_points = [(0.0, 0.0)]

    scalar = extractor.extract((0.0, 0.0), lines)
    vectorized = extractor.extract_many(query_points, lines)

    np.testing.assert_array_equal(vectorized[0], scalar)


def test_vectorized_extract_many_empty_lines_returns_sentinel():
    """Vectorized batch with no lines should return sentinel for all points."""
    extractor = LocalObstacleFeatureExtractor()
    query_points = [(1.0, 1.0), (2.0, 2.0)]

    vectorized = extractor.extract_many(query_points, [])

    assert vectorized.shape == (2, PREDICTIVE_OBSTACLE_FEATURE_DIM)
    np.testing.assert_allclose(vectorized[:, 0], [50.0, 50.0])
    np.testing.assert_allclose(vectorized[:, 5], [0.0, 0.0])


def test_vectorized_extract_many_single_line_multiple_points():
    """Vectorized batch with one line should compute correct distances for all points."""
    extractor = LocalObstacleFeatureExtractor()
    lines = [((0.0, 0.0), (2.0, 0.0))]
    query_points = [(1.0, 1.0), (1.0, -1.0), (3.0, 0.0)]

    vectorized = extractor.extract_many(query_points, lines)

    # Distances: 1.0, 1.0, 1.0
    np.testing.assert_allclose(vectorized[:, 0], [1.0, 1.0, 1.0], atol=1e-7)
    # All valid
    np.testing.assert_allclose(vectorized[:, 5], [1.0, 1.0, 1.0])
