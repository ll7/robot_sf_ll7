"""Return-annotation contracts for the optimized Theta* adapter."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from robot_sf.planner import theta_star_v2
from robot_sf.planner.theta_star_v2 import HighPerformanceThetaStar, _bind_fast_in_collision


def test_plan_annotation_matches_upstream_runtime_tuple() -> None:
    """Describe the path-and-metadata tuple returned by the upstream planner."""
    assert (
        inspect.signature(HighPerformanceThetaStar.plan).return_annotation
        == "tuple[list[tuple[float, ...]], dict[str, Any]]"
    )


def test_python_collision_fallback_is_annotated_as_bool(monkeypatch) -> None:
    """Exercise the non-Numba nested function and its concrete return annotation."""
    monkeypatch.setattr(theta_star_v2, "njit", None)
    grid = SimpleNamespace(type_map=SimpleNamespace(array=np.zeros((3, 3), dtype=np.int64)))

    _bind_fast_in_collision(grid)

    assert inspect.signature(grid.in_collision).return_annotation == "bool"
    assert grid.in_collision((0, 0), (2, 2)) is False
