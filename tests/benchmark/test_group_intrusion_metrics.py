"""Tests for group-space intrusion metrics (issue #3972).

Cover the pure geometric reductions in
:mod:`robot_sf.benchmark.group_space_metrics`:

- no declared groups -> unavailable, zero intrusion, NaN min distances;
- distinct episode-rate vs time-ratio semantics;
- no-intrusion trajectories stay zero;
- signed boundary clearance is negative inside the o-space;
- polygon o-space inside/outside detection.
"""

from __future__ import annotations

import math

import numpy as np

from robot_sf.benchmark.group_space_metrics import compute_group_space_metrics


def _circular_group(centroid=(5.0, 5.0), radius=1.0, group_id="g1"):
    """Return a minimal circular group spec (no explicit polygon)."""
    return {
        "group_id": group_id,
        "type": "conversation",
        "members": ["ped_a", "ped_b"],
        "formation": "circular_conversation",
        "centroid": [float(centroid[0]), float(centroid[1])],
        "radius": float(radius),
        "o_space_polygon": None,
    }


def test_no_groups_reports_unavailable_and_nan_distances():
    """With no declared groups the metric is unavailable with NaN min distances."""
    robot_pos = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = compute_group_space_metrics(robot_pos, [])

    assert result["group_space_available"] == 0.0
    assert result["group_count"] == 0.0
    assert result["group_intrusion_episode_rate"] == 0.0
    assert result["group_intrusion_time_ratio"] == 0.0
    assert result["group_intrusion_step_count"] == 0.0
    assert math.isnan(result["min_distance_to_group_centroid"])
    assert math.isnan(result["min_distance_to_group_boundary"])
    assert result["nearest_group_id"] is None


def test_episode_rate_and_time_ratio_are_distinct():
    """A 10-step trajectory with 2 intrusive steps distinguishes the two metrics."""
    # Group centered at origin, radius 1.0. Two of ten steps lie inside.
    group = _circular_group(centroid=(0.0, 0.0), radius=1.0)
    xs = np.array([5.0, 4.0, 3.0, 2.0, 0.3, 0.2, 1.5, 2.5, 3.5, 4.5])
    robot_pos = np.stack([xs, np.zeros_like(xs)], axis=1)

    result = compute_group_space_metrics(robot_pos, [group])

    assert result["group_space_available"] == 1.0
    assert result["group_count"] == 1.0
    assert result["group_intrusion_step_count"] == 2.0
    assert result["group_metric_timestep_count"] == 10.0
    assert result["group_intrusion_episode_rate"] == 1.0
    assert result["group_intrusion_time_ratio"] == 0.2
    assert result["nearest_group_id"] == "g1"


def test_no_intrusion_trajectory_stays_zero():
    """A trajectory that never enters the o-space reports no intrusion."""
    group = _circular_group(centroid=(0.0, 0.0), radius=1.0)
    xs = np.array([5.0, 4.0, 3.0, 2.0, 1.5])
    robot_pos = np.stack([xs, np.zeros_like(xs)], axis=1)

    result = compute_group_space_metrics(robot_pos, [group])

    assert result["group_space_available"] == 1.0
    assert result["group_intrusion_episode_rate"] == 0.0
    assert result["group_intrusion_time_ratio"] == 0.0
    # Nearest approach is at x=1.5 -> centroid distance 1.5, boundary clearance 0.5.
    assert result["min_distance_to_group_centroid"] == 1.5
    assert result["min_distance_to_group_boundary"] > 0.0
    assert math.isclose(result["min_distance_to_group_boundary"], 0.5, abs_tol=1e-9)


def test_signed_boundary_distance_is_negative_inside():
    """A point inside the radius yields a negative signed boundary clearance."""
    group = _circular_group(centroid=(0.0, 0.0), radius=2.0)
    robot_pos = np.array([[0.0, 0.0]])  # dead center

    result = compute_group_space_metrics(robot_pos, [group])

    assert result["group_intrusion_episode_rate"] == 1.0
    assert result["group_intrusion_time_ratio"] == 1.0
    assert result["min_distance_to_group_centroid"] == 0.0
    # center distance 0 - radius 2 => -2.0
    assert math.isclose(result["min_distance_to_group_boundary"], -2.0, abs_tol=1e-9)


def test_polygon_o_space_detects_inside_and_outside():
    """A square o-space polygon detects inside vs outside timesteps."""
    group = {
        "group_id": "square_a",
        "type": "static_group",
        "members": ["ped_a"],
        "formation": "cluster",
        "centroid": [1.0, 1.0],
        "radius": 0.5,  # circular proxy would NOT be used because polygon is set
        "o_space_polygon": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
    }
    # Step 0 far outside, step 1 inside the square, step 2 outside again.
    robot_pos = np.array([[5.0, 5.0], [1.0, 1.0], [3.0, 3.0]])

    result = compute_group_space_metrics(robot_pos, [group])

    assert result["group_space_available"] == 1.0
    assert result["group_intrusion_step_count"] == 1.0
    assert result["group_intrusion_episode_rate"] == 1.0
    assert math.isclose(result["group_intrusion_time_ratio"], 1.0 / 3.0, abs_tol=1e-9)
    # Inside the square center, boundary clearance is negative (distance 1.0 to edge).
    assert result["min_distance_to_group_boundary"] < 0.0
    assert result["nearest_group_id"] == "square_a"


def test_nearest_group_id_is_most_intruded_group():
    """With two groups, the reported nearest group has the deepest intrusion."""
    shallow = _circular_group(centroid=(0.0, 0.0), radius=1.0, group_id="shallow")
    deep = _circular_group(centroid=(10.0, 0.0), radius=3.0, group_id="deep")
    # Robot sits at the center of the "deep" group (clearance -3) and skims the
    # boundary of the "shallow" group.
    robot_pos = np.array([[10.0, 0.0], [1.0, 0.0]])

    result = compute_group_space_metrics(robot_pos, [shallow, deep])

    assert result["nearest_group_id"] == "deep"
    assert math.isclose(result["min_distance_to_group_boundary"], -3.0, abs_tol=1e-9)
