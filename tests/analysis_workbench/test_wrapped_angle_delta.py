"""Fast coverage for the canonical wrapped-angle delta helpers."""

from __future__ import annotations

import math


def test_wrapped_angle_helpers_use_canonical_wrap() -> None:
    """Helper wrappers match the canonical [-pi, pi) wrap on known inputs."""
    from robot_sf.analysis_workbench.episode_phases import _wrapped_angle_delta as episode_delta
    from robot_sf.analysis_workbench.event_alignment import _wrapped_angle_delta as alignment_delta

    for delta in (episode_delta, alignment_delta):
        assert abs(delta(0.0, 0.5) + 0.5) < 1e-12
        assert abs(delta(2.0 * math.pi + 0.25, 0.25)) < 1e-12
        assert abs(delta(math.pi, 0.0) + math.pi) < 1e-12
