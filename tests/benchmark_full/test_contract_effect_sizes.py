"""Contract test T010 for `compute_effect_sizes`.

Expectations:
  - Produces list of EffectSizeReport objects (one per archetype) once implemented.
  - Contains comparisons with standardized (Cohen's h) values for rate metrics.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.effects import compute_effect_sizes


def test_compute_effect_sizes_structure():
    class _Metric:
        def __init__(self, name, mean):
            self.name = name
            self.mean = mean
            self.median = mean
            self.p95 = mean
            self.mean_ci = (mean * 0.9, mean * 1.1)
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

    reports = compute_effect_sizes(groups, _Cfg())
    assert reports and len(reports) == 1
    rep = reports[0]
    assert rep.archetype == "crossing"
    assert rep.comparisons
    comp = rep.comparisons[0]
    for attr in ["metric", "density_low", "density_high", "diff", "standardized"]:
        assert hasattr(comp, attr)
