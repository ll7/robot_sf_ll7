"""Group-space intrusion metrics for scenario-declared social groups.

This module (issue #3972) computes how much a robot trajectory intrudes into the
shared "o-space" of declared social pedestrian groups (for example standing
conversations). It complements the existing ``group_split_intrusion`` diagnostic
in :mod:`robot_sf.benchmark.metrics`, which was previously left uncomputed
because episodes carried no social-group labels.

The metrics are pure geometric reductions over the robot center trajectory and
the declared group geometry. They are diagnostic social-space metrics, not
validated human-comfort or safety metrics.

Definitions
-----------
- ``group_intrusion_episode_rate``: per-episode binary indicator. ``1.0`` when
  the robot center lies inside any group o-space for at least one timestep, else
  ``0.0``. Aggregated across rows, the mean is the fraction of episodes with an
  intrusion.
- ``group_intrusion_time_ratio``: ``intrusion_step_count / timestep_count``,
  where a timestep is intrusive when the robot center lies inside any group
  o-space.
- ``min_distance_to_group_centroid``: minimum Euclidean distance from the robot
  center to any group centroid over all timesteps (meters).
- ``min_distance_to_group_boundary``: signed boundary clearance in meters,
  minimized over all timesteps and groups. Positive outside the o-space, ``0.0``
  on the boundary, negative inside. The minimum reflects worst-case intrusion
  severity.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon


def group_specs_from_map(map_def: Any) -> list[dict[str, Any]]:
    """Return JSON-safe group specifications from a map definition.

    Args:
        map_def: A ``MapDefinition``-like object exposing ``social_groups``.

    Returns:
        list[dict[str, Any]]: One JSON-safe spec per declared social group.
    """
    groups = getattr(map_def, "social_groups", None) or []
    specs: list[dict[str, Any]] = []
    for group in groups:
        as_spec = getattr(group, "as_spec", None)
        if callable(as_spec):
            specs.append(as_spec())
        elif isinstance(group, Mapping):
            specs.append(dict(group))
    return specs


def _group_geometry(spec: Mapping[str, Any]) -> tuple[Any, np.ndarray, float, Polygon | None]:
    """Extract ``(group_id, centroid, radius, polygon)`` from a group spec.

    Returns:
        tuple: Group id, centroid ``(2,)`` array, radius, and an optional
        Shapely polygon (``None`` when the group uses a circular proxy).
    """
    group_id = spec.get("group_id")
    centroid = np.asarray(spec.get("centroid", (0.0, 0.0)), dtype=float).reshape(-1)[:2]
    radius = float(spec.get("radius", 0.0))
    poly_points = spec.get("o_space_polygon")
    polygon = (
        Polygon([(float(x), float(y)) for x, y in poly_points])
        if isinstance(poly_points, (list, tuple)) and len(poly_points) >= 3
        else None
    )
    return group_id, centroid, radius, polygon


def _polygon_signed_clearance(polygon: Polygon, point_xy: np.ndarray) -> float:
    """Return signed clearance from a point to a polygon boundary.

    Negative inside the polygon, ``0.0`` on the boundary, positive outside.

    Returns:
        float: Signed boundary clearance in meters.
    """
    point = Point(float(point_xy[0]), float(point_xy[1]))
    boundary_distance = float(polygon.exterior.distance(point))
    if polygon.contains(point):
        return -boundary_distance
    return boundary_distance


def _empty_metrics(group_count: int, step_count: int) -> dict[str, Any]:
    """Return the default (no-intrusion / unavailable) metric payload.

    Returns:
        dict[str, Any]: Flat group-space metric defaults.
    """
    return {
        "group_space_available": 0.0,
        "group_count": float(group_count),
        "group_intrusion_episode_rate": 0.0,
        "group_intrusion_time_ratio": 0.0,
        "group_intrusion_step_count": 0.0,
        "group_metric_timestep_count": float(step_count),
        "min_distance_to_group_centroid": float("nan"),
        "min_distance_to_group_boundary": float("nan"),
        "nearest_group_id": None,
    }


def compute_group_space_metrics(
    robot_pos: np.ndarray,
    groups: Any,
) -> dict[str, Any]:
    """Compute group-space intrusion metrics for one episode.

    Args:
        robot_pos: Robot center trajectory of shape ``(T, 2)``.
        groups: Iterable of JSON-safe group specs (mappings) or objects exposing
            ``as_spec()`` / ``social_groups``-style geometry.

    Returns:
        dict[str, Any]: Flat group-space metric columns. When no groups are
        declared or the trajectory is empty, intrusion metrics are ``0.0`` and
        min distances are ``NaN`` with ``group_space_available == 0.0``.
    """
    if isinstance(groups, Mapping):
        specs: list[Mapping[str, Any]] = [groups]
    else:
        specs = [g for g in (groups or []) if isinstance(g, Mapping)]

    pos = np.asarray(robot_pos, dtype=float)
    step_count = int(pos.shape[0]) if pos.ndim == 2 and pos.shape[1] >= 2 else 0

    if not specs or step_count == 0:
        return _empty_metrics(len(specs), step_count)

    xy = pos[:, :2]
    finite_mask = np.all(np.isfinite(xy), axis=1)
    xy_finite = xy[finite_mask]
    if xy_finite.shape[0] == 0:
        return _empty_metrics(len(specs), step_count)
    intrusive_steps = np.zeros(step_count, dtype=bool)
    min_centroid = float("inf")
    min_boundary = float("inf")
    nearest_group_id: Any = None

    for spec in specs:
        group_id, centroid, radius, polygon = _group_geometry(spec)
        centroid_dist = np.linalg.norm(xy_finite - centroid[None, :], axis=1)
        group_min_centroid = float(np.min(centroid_dist))
        min_centroid = min(min_centroid, group_min_centroid)

        if polygon is None:
            signed = centroid_dist - radius
        else:
            signed = np.array(
                [_polygon_signed_clearance(polygon, p) for p in xy_finite], dtype=float
            )

        group_min_boundary = float(np.min(signed))
        if group_min_boundary < min_boundary:
            min_boundary = group_min_boundary
            nearest_group_id = group_id

        intrusive_steps[finite_mask] |= signed < 0.0

    intrusion_step_count = int(np.count_nonzero(intrusive_steps))
    return {
        "group_space_available": 1.0,
        "group_count": float(len(specs)),
        "group_intrusion_episode_rate": 1.0 if intrusion_step_count > 0 else 0.0,
        "group_intrusion_time_ratio": float(intrusion_step_count / step_count),
        "group_intrusion_step_count": float(intrusion_step_count),
        "group_metric_timestep_count": float(step_count),
        "min_distance_to_group_centroid": min_centroid,
        "min_distance_to_group_boundary": min_boundary,
        "nearest_group_id": nearest_group_id,
    }


__all__ = [
    "compute_group_space_metrics",
    "group_specs_from_map",
]
