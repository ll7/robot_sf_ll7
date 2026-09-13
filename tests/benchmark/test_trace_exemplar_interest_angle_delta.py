"""Focused coverage for the trace-exemplar angle-delta helper."""

from __future__ import annotations

import math


def test_angle_delta_wraps_onto_canonical_interval() -> None:
    """The interest-score angle delta matches the canonical [-pi, pi) wrap."""
    from robot_sf.benchmark.trace_exemplar_interest import _angle_delta

    assert abs(_angle_delta(0.0, 0.5) + 0.5) < 1e-12
    assert abs(_angle_delta(2.0 * math.pi + 0.25, 0.25)) < 1e-12
    assert abs(_angle_delta(math.pi, 0.0) + math.pi) < 1e-12
