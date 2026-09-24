"""Contract test T010 for `compute_effect_sizes`.

Expectations:
  - Produces list of EffectSizeReport objects (one per archetype) once implemented.
  - Contains comparisons with standardized (Cohen's h) values for rate metrics.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.effects import compute_effect_sizes


def test_compute_effect_sizes_structure():
    """Verify compute_effect_sizes emits one archetype report with comparison fields."""

    class _Metric:
        """Minimal aggregate-metric double carrying mean, percentile, and CI fields."""

        def __init__(self, name, mean):
            """Store the metric name and mean used across percentile and CI fields.

            Args:
                name: Metric identifier reported in comparison entries.
                mean: Central value mirrored into median, p95, and the mean CI.
            """
            self.name = name
            self.mean = mean
            self.median = mean
            self.p95 = mean
            self.mean_ci = (mean * 0.9, mean * 1.1)
            self.median_ci = None

    class _Group:
        """Minimal aggregate-group double with identity, count, and one metric."""

        def __init__(self, archetype, density, mean_collision):
            """Store group identity and a single collision-rate metric.

            Args:
                archetype: Interaction archetype label used to group reports.
                density: Density label compared against the reference density.
                mean_collision: Mean collision rate for this group.
            """
            self.archetype = archetype
            self.density = density
            self.count = 10
            self.metrics = {"collision_rate": _Metric("collision_rate", mean_collision)}

    groups = [
        _Group("crossing", "low", 0.10),
        _Group("crossing", "high", 0.25),
    ]

    class _Cfg:
        """Minimal config stub selecting the low-density reference for comparisons."""

        effect_size_reference_density = "low"

    reports = compute_effect_sizes(groups, _Cfg())
    assert reports and len(reports) == 1
    rep = reports[0]
    assert rep.archetype == "crossing"
    assert rep.comparisons
    comp = rep.comparisons[0]
    for attr in ["metric", "density_low", "density_high", "diff", "standardized"]:
        assert hasattr(comp, attr)
