"""Integration of aggregation → effect size computation.

Verifies that aggregate_metrics output feeds into compute_effect_sizes using the
configured reference density, producing sensible diffs for rate and continuous
metrics without NaN values.
"""

from __future__ import annotations

import math

import pytest

from robot_sf.benchmark.full_classic.aggregation import aggregate_metrics
from robot_sf.benchmark.full_classic.effects import compute_effect_sizes


class _Cfg:
    bootstrap_samples = 50
    bootstrap_confidence = 0.95
    master_seed = 123
    smoke = True
    effect_size_reference_density = "low"


@pytest.fixture
def records():
    return [
        {
            "episode_id": "low1",
            "archetype": "crossing",
            "density": "low",
            "metrics": {
                "collision_rate": 0.1,
                "success_rate": 0.9,
                "time_to_goal": 10.0,
                "path_efficiency": 0.8,
                "avg_speed": 1.0,
                "snqi": 0.7,
            },
        },
        {
            "episode_id": "high1",
            "archetype": "crossing",
            "density": "high",
            "metrics": {
                "collision_rate": 0.3,
                "success_rate": 0.7,
                "time_to_goal": 12.0,
                "path_efficiency": 0.6,
                "avg_speed": 0.9,
                "snqi": 0.5,
            },
        },
    ]


def test_aggregate_then_effect_sizes(records):
    groups = aggregate_metrics(records, _Cfg())
    reports = compute_effect_sizes(groups, _Cfg())

    assert reports and len(reports) == 1
    report = reports[0]
    assert report.archetype == "crossing"
    assert report.comparisons

    comps = {c.metric: c for c in report.comparisons}
    assert set(comps) >= {
        "collision_rate",
        "success_rate",
        "time_to_goal",
        "path_efficiency",
        "avg_speed",
        "snqi",
    }

    assert comps["collision_rate"].diff == pytest.approx(0.2)
    assert math.isfinite(comps["collision_rate"].standardized)
    assert comps["time_to_goal"].diff == pytest.approx(2.0)
    assert comps["path_efficiency"].diff == pytest.approx(-0.2)
    assert comps["avg_speed"].diff == pytest.approx(-0.1)
    assert comps["snqi"].diff == pytest.approx(-0.2)
    assert all(math.isfinite(c.standardized) or c.standardized == 0.0 for c in comps.values())
