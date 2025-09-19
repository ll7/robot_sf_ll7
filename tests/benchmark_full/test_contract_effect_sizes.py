"""Contract test T010 for `compute_effect_sizes`.

Expectations:
  - Produces list of EffectSizeReport objects (one per archetype).
  - Uses rate metrics (e.g., collision_rate) to compute standardized differences (Cohen's h).

Current state: compute_effect_sizes not implemented -> expect NotImplementedError.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.effects import compute_effect_sizes


def test_compute_effect_sizes_not_implemented():
    class _Metric:
        def __init__(self, name, mean):
            self.name = name
            self.mean = mean
            self.median = mean
            self.p95 = mean
            self.mean_ci = None
            self.median_ci = None

    class _Group:
        def __init__(self, archetype, density, mean_collision):
            self.archetype = archetype
            self.density = density
            self.count = 10
            self.metrics = {"collision_rate": _Metric("collision_rate", mean_collision)}

    groups = [
        _Group("crossing", "low", 0.10),
        _Group("crossing", "high", 0.25),
    ]

    class _Cfg:
        effect_size_reference_density = "low"

    with pytest.raises(NotImplementedError):  # until T032
        compute_effect_sizes(groups, _Cfg())
