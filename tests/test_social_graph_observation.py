"""Tests for the shared social graph observation adapter."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.sensor.social_graph_observation import (
    PEDESTRIAN_FEATURE_NAMES,
    SocialGraphObservationAdapter,
    SocialGraphObservationConfig,
    build_social_graph_observation,
)


def _nested_obs(
    *,
    ped_positions: list[list[float]] | None = None,
    ped_velocities: list[list[float]] | None = None,
    ped_count: int | None = None,
    heading: float = 0.0,
) -> dict:
    """Return a minimal nested SocNav observation fixture."""
    positions = [] if ped_positions is None else ped_positions
    velocities = [[0.0, 0.0] for _ in positions] if ped_velocities is None else ped_velocities
    count = len(positions) if ped_count is None else ped_count
    return {
        "robot": {
            "position": np.array([0.0, 0.0], dtype=np.float32),
            "heading": np.array([heading], dtype=np.float32),
            "velocity_xy": np.array([0.5, 0.0], dtype=np.float32),
            "radius": np.array([0.3], dtype=np.float32),
        },
        "goal": {"current": np.array([4.0, 0.0], dtype=np.float32)},
        "pedestrians": {
            "positions": np.asarray(positions, dtype=np.float32).reshape(-1, 2),
            "velocities": np.asarray(velocities, dtype=np.float32).reshape(-1, 2),
            "count": np.array([count], dtype=np.float32),
            "radius": np.array([0.25], dtype=np.float32),
        },
    }


def _flat_obs(nested: dict) -> dict:
    """Flatten the nested fixture with RobotEnv's SocNav naming convention."""
    return {
        "robot_position": nested["robot"]["position"],
        "robot_heading": nested["robot"]["heading"],
        "robot_velocity_xy": nested["robot"]["velocity_xy"],
        "robot_radius": nested["robot"]["radius"],
        "goal_current": nested["goal"]["current"],
        "pedestrians_positions": nested["pedestrians"]["positions"],
        "pedestrians_velocities": nested["pedestrians"]["velocities"],
        "pedestrians_count": nested["pedestrians"]["count"],
        "pedestrians_radius": nested["pedestrians"]["radius"],
    }


def test_empty_pedestrians_emit_zero_features_and_masks() -> None:
    """Empty scenes should keep fixed shapes and explicit inactive masks."""
    graph = build_social_graph_observation(
        _nested_obs(),
        config=SocialGraphObservationConfig(max_pedestrians=3),
    )

    assert graph["pedestrian_features"].shape == (3, len(PEDESTRIAN_FEATURE_NAMES))
    assert graph["pedestrian_mask"].tolist() == [False, False, False]
    assert graph["pedestrian_count"].tolist() == [0]
    assert graph["edge_index"].shape == (2, 0)
    assert graph["edge_type"].shape == (0,)
    np.testing.assert_allclose(graph["pedestrian_features"], 0.0)


def test_pedestrians_are_ordered_by_distance_with_geometric_tie_breaks() -> None:
    """Ordering should not depend on source array order when positions differ."""
    config = SocialGraphObservationConfig(max_pedestrians=4)
    obs_a = _nested_obs(
        ped_positions=[[2.0, 0.0], [1.0, 1.0], [1.0, -1.0]],
        ped_velocities=[[0.2, 0.0], [0.0, 0.1], [0.0, -0.1]],
    )
    obs_b = _nested_obs(
        ped_positions=[[1.0, -1.0], [2.0, 0.0], [1.0, 1.0]],
        ped_velocities=[[0.0, -0.1], [0.2, 0.0], [0.0, 0.1]],
    )

    graph_a = build_social_graph_observation(obs_a, config=config)
    graph_b = build_social_graph_observation(obs_b, config=config)

    np.testing.assert_allclose(graph_a["pedestrian_features"], graph_b["pedestrian_features"])
    np.testing.assert_allclose(
        graph_a["pedestrian_features"][:3, :2],
        np.array([[1.0, -1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32),
    )


def test_exact_position_ties_are_ordered_by_relative_velocity() -> None:
    """Duplicate positions with different velocities should stay permutation invariant."""
    config = SocialGraphObservationConfig(max_pedestrians=2)
    obs_a = _nested_obs(
        ped_positions=[[1.0, 0.0], [1.0, 0.0]],
        ped_velocities=[[0.3, 0.0], [0.1, 0.0]],
    )
    obs_b = _nested_obs(
        ped_positions=[[1.0, 0.0], [1.0, 0.0]],
        ped_velocities=[[0.1, 0.0], [0.3, 0.0]],
    )

    graph_a = build_social_graph_observation(obs_a, config=config)
    graph_b = build_social_graph_observation(obs_b, config=config)

    np.testing.assert_allclose(graph_a["pedestrian_features"], graph_b["pedestrian_features"])
    np.testing.assert_allclose(graph_a["pedestrian_features"][:, 2], [-0.4, -0.2])


def test_cap_truncates_nearest_pedestrians_and_preserves_mask() -> None:
    """The cap should keep nearest pedestrians and expose active rows through a mask."""
    graph = build_social_graph_observation(
        _nested_obs(ped_positions=[[5.0, 0.0], [1.0, 0.0], [3.0, 0.0]]),
        config=SocialGraphObservationConfig(max_pedestrians=2),
    )

    assert graph["pedestrian_mask"].tolist() == [True, True]
    assert graph["pedestrian_count"].tolist() == [2]
    np.testing.assert_allclose(graph["pedestrian_features"][:, :2], [[1.0, 0.0], [3.0, 0.0]])


def test_pedestrian_velocity_uses_socnav_robot_frame_contract() -> None:
    """Pedestrian velocity input is already robot-frame in SocNav observations."""
    graph = build_social_graph_observation(
        _nested_obs(
            ped_positions=[[0.0, 2.0]],
            ped_velocities=[[0.0, 1.0]],
            heading=np.pi / 2.0,
        ),
        config=SocialGraphObservationConfig(max_pedestrians=1),
    )

    np.testing.assert_allclose(graph["pedestrian_features"][0, :2], [2.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(graph["pedestrian_features"][0, 2:4], [0.0, 1.5], atol=1e-6)


def test_missing_robot_velocity_xy_fails_closed() -> None:
    """Angular speed must not be silently treated as lateral velocity."""
    obs = _nested_obs(ped_positions=[[1.0, 0.0]])
    del obs["robot"]["velocity_xy"]
    obs["robot"]["speed"] = np.array([0.5, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="robot.velocity_xy"):
        build_social_graph_observation(obs)


def test_optional_static_obstacle_tokens_emit_features_and_edges() -> None:
    """Static obstacle segments should be optional graph nodes when configured."""
    graph = build_social_graph_observation(
        _nested_obs(ped_positions=[[2.0, 0.0]]),
        config=SocialGraphObservationConfig(
            max_pedestrians=2,
            include_static_obstacles=True,
            max_static_obstacles=2,
        ),
        obstacle_segments=[[1.0, -1.0, 1.0, 1.0], [3.0, 0.0, 5.0, 0.0]],
    )

    assert graph["static_obstacle_mask"].tolist() == [True, True]
    assert graph["static_obstacle_count"].tolist() == [2]
    np.testing.assert_allclose(graph["static_obstacle_features"][0, :2], [1.0, 0.0])
    assert graph["edge_type"].tolist() == [0, 1, 1]


def test_history_stack_fills_then_shifts_and_resets() -> None:
    """Stateful adapter should fill initial history, shift over time, and reset per episode."""
    stateless = build_social_graph_observation(
        _nested_obs(ped_positions=[[1.0, 0.0]]),
        config=SocialGraphObservationConfig(max_pedestrians=1, history_steps=3),
    )
    assert stateless["pedestrian_history"].shape == (3, 1, len(PEDESTRIAN_FEATURE_NAMES))
    np.testing.assert_allclose(stateless["pedestrian_history"][:, 0, 0], [1.0, 1.0, 1.0])

    adapter = SocialGraphObservationAdapter(
        SocialGraphObservationConfig(max_pedestrians=1, history_steps=3)
    )
    first = adapter.build(_nested_obs(ped_positions=[[1.0, 0.0]]))
    second = adapter.build(_nested_obs(ped_positions=[[2.0, 0.0]]))

    assert first["pedestrian_history"].shape == (3, 1, len(PEDESTRIAN_FEATURE_NAMES))
    np.testing.assert_allclose(first["pedestrian_history"][:, 0, 0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(second["pedestrian_history"][:, 0, 0], [1.0, 1.0, 2.0])

    adapter.reset()
    after_reset = adapter.build(_nested_obs(ped_positions=[[3.0, 0.0]]))
    np.testing.assert_allclose(after_reset["pedestrian_history"][:, 0, 0], [3.0, 3.0, 3.0])


def test_flat_and_nested_socnav_inputs_match() -> None:
    """The adapter should accept both nested and flattened SocNav observation layouts."""
    nested = _nested_obs(
        ped_positions=[[2.0, 0.0], [0.0, 2.0]],
        ped_velocities=[[0.1, 0.0], [0.0, 0.1]],
    )
    config = SocialGraphObservationConfig(max_pedestrians=3)
    nested_graph = build_social_graph_observation(nested, config=config)
    flat_graph = build_social_graph_observation(_flat_obs(nested), config=config)

    for key in nested_graph:
        np.testing.assert_array_equal(nested_graph[key], flat_graph[key])


def test_edge_index_orders_pedestrians_first_then_static_obstacles() -> None:
    """Edge index must list active pedestrian sources first, then static obstacle sources."""
    graph = build_social_graph_observation(
        _nested_obs(ped_positions=[[2.0, 0.0], [3.0, 0.0]]),
        config=SocialGraphObservationConfig(
            max_pedestrians=4,
            include_static_obstacles=True,
            max_static_obstacles=2,
        ),
        obstacle_segments=[[1.0, -1.0, 1.0, 1.0]],
    )
    assert graph["edge_type"].tolist() == [0, 0, 1]
    assert graph["edge_index"][0].tolist() == [1, 2, 5]
    assert graph["edge_index"][1].tolist() == [0, 0, 0]


def test_edge_index_supports_static_obstacles_without_pedestrians() -> None:
    """Static obstacle edges should be valid when no pedestrian rows are active."""
    graph = build_social_graph_observation(
        _nested_obs(),
        config=SocialGraphObservationConfig(
            max_pedestrians=3,
            include_static_obstacles=True,
            max_static_obstacles=2,
        ),
        obstacle_segments=[[1.0, -1.0, 1.0, 1.0]],
    )

    assert graph["edge_type"].tolist() == [1]
    assert graph["edge_index"][0].tolist() == [4]
    assert graph["edge_index"][1].tolist() == [0]


def test_future_like_deployment_fields_fail_closed() -> None:
    """Future trajectory labels should be rejected by the deployment adapter."""
    obs = _nested_obs(ped_positions=[[1.0, 0.0]])
    obs["pedestrians"]["future_positions"] = [[2.0, 0.0]]

    with pytest.raises(ValueError, match="future_positions"):
        build_social_graph_observation(obs)


def test_history_return_not_mutated_by_later_build_calls() -> None:
    """Returned pedestrian_history must be isolated from subsequent adapter writes."""
    adapter = SocialGraphObservationAdapter(
        SocialGraphObservationConfig(max_pedestrians=1, history_steps=3)
    )
    adapter.build(_nested_obs(ped_positions=[[1.0, 0.0]]))

    second = adapter.build(_nested_obs(ped_positions=[[2.0, 0.0]]))
    hist_2 = second["pedestrian_history"]

    third = adapter.build(_nested_obs(ped_positions=[[3.0, 0.0]]))
    np.testing.assert_allclose(hist_2[:, 0, 0], [1.0, 1.0, 2.0])
    np.testing.assert_allclose(third["pedestrian_history"][:, 0, 0], [1.0, 2.0, 3.0])

    hist_2[0, 0, 0] = 999.0

    fourth = adapter.build(_nested_obs(ped_positions=[[4.0, 0.0]]))
    np.testing.assert_allclose(fourth["pedestrian_history"][:, 0, 0], [2.0, 3.0, 4.0])
