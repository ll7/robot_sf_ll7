"""Contract test T011 for `evaluate_precision`.

Expectation:
  - Returns StatisticalSufficiencyReport with `evaluations` list and `final_pass` bool.
  - Contains per-group entries with metric_status list including collision_rate & success_rate.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.precision import evaluate_precision


def test_evaluate_precision_structure():
    class _Metric:
        def __init__(self, name, mean, mean_ci):
            self.name = name
            self.mean = mean
            self.median = mean
            self.p95 = mean
            self.mean_ci = mean_ci
            self.median_ci = None

    class _Group:
        def __init__(self, archetype, density):
            self.archetype = archetype
            self.density = density
            self.count = 40
            self.metrics = {
                "collision_rate": _Metric("collision_rate", 0.1, (0.08, 0.12)),
                "success_rate": _Metric("success_rate", 0.9, (0.85, 0.94)),
            }

    groups = [_Group("crossing", "low")]

    class _Cfg:
        smoke = True
        collision_ci = 0.05
        success_ci = 0.05

    report = evaluate_precision(groups, _Cfg())
    assert hasattr(report, "evaluations") and hasattr(report, "final_pass")
    assert isinstance(report.evaluations, list) and len(report.evaluations) == 1
    ev = report.evaluations[0]
    assert ev.archetype == "crossing" and ev.density == "low"
    assert ev.metric_status and len(ev.metric_status) == 2
    for ms in ev.metric_status:
        assert ms.metric in {"collision_rate", "success_rate"}
        assert ms.half_width <= ms.target
