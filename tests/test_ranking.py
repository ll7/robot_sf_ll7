"""TODO docstring. Document this module."""

from __future__ import annotations

import math

import pytest

from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.ranking import compute_ranking, format_csv, format_markdown


def _rec(g, **m):
    """TODO docstring. Document this function.

    Args:
        g: TODO docstring.
        m: TODO docstring.
    """
    return {
        "scenario_params": {"algo": g},
        "algo": g,
        "scenario_id": g,
        "metrics": m,
    }


def test_compute_ranking_basic():
    """TODO docstring. Document this function."""
    records = [
        _rec("a", collisions=1),
        _rec("a", collisions=3),
        _rec("b", collisions=0),
        _rec("b", collisions=2),
        _rec("c", collisions=5),
    ]
    rows = compute_ranking(records, metric="collisions", ascending=True)
    # Means: a=2, b=1, c=5 -> ascending gives b, a, c
    assert [r.group for r in rows] == ["b", "a", "c"]
    assert [r.count for r in rows] == [2, 2, 1]


def test_compute_ranking_top_and_desc():
    """TODO docstring. Document this function."""
    records = [
        _rec("a", comfort_exposure=0.1),
        _rec("a", comfort_exposure=0.4),
        _rec("b", comfort_exposure=0.3),
        _rec("b", comfort_exposure=0.2),
        _rec("c", comfort_exposure=0.5),
    ]
    # Descending (higher is better for this synthetic case), limit top 2
    rows = compute_ranking(records, metric="comfort_exposure", ascending=False, top=2)
    # Means: a=0.25, b=0.25, c=0.5 -> descending gives c first, then a/b
    assert rows[0].group == "c"
    assert rows[0].count == 1
    assert len(rows) == 2


def test_formatters_return_strings():
    """TODO docstring. Document this function."""
    records = [_rec("a", collisions=1), _rec("a", collisions=3)]
    rows = compute_ranking(records, metric="collisions")
    md = format_markdown(rows, "collisions")
    csv = format_csv(rows, "collisions")
    assert "| Rank |" in md
    assert md.endswith("\n")
    assert csv.splitlines()[0].startswith("rank,group,mean_")
    assert csv.endswith("\n")


def test_compute_ranking_requires_explicit_cross_track_mode() -> None:
    """Ranking should share the aggregate guard against silent cross-track pooling."""
    records = [
        {
            "benchmark_track": "grid_socnav_v1",
            "scenario_params": {"algo": "a", "benchmark_track": "grid_socnav_v1"},
            "metrics": {"collisions": 1},
        },
        {
            "benchmark_track": "lidar_2d_v1",
            "scenario_params": {"algo": "a", "benchmark_track": "lidar_2d_v1"},
            "metrics": {"collisions": 3},
        },
    ]

    with pytest.raises(AggregationMetadataError):
        compute_ranking(records, metric="collisions")

    rows = compute_ranking(
        records,
        metric="collisions",
        observation_track_mode="diagnostic-cross-track",
    )
    assert [row.group for row in rows] == ["grid_socnav_v1 :: a", "lidar_2d_v1 :: a"]


def test_compute_ranking_excludes_explicitly_ineligible_records() -> None:
    """Ranking must share aggregate evidence admission and omit marked-out rows."""
    ineligible = _rec("a", collisions=100)
    ineligible["algorithm_metadata"] = {
        "foresight_prediction": {"evidence_eligible": False},
    }
    records = [_rec("a", collisions=0), ineligible, _rec("b", collisions=2)]

    rows = compute_ranking(records, metric="collisions")

    assert [(row.group, row.mean, row.count) for row in rows] == [
        ("a", 0.0, 1),
        ("b", 2.0, 1),
    ]


def test_compute_ranking_ignores_non_finite_metric_values() -> None:
    """Ranking must not emit non-finite means from NaN or infinite input metrics."""
    records = [
        _rec("a", collisions=float("nan")),
        _rec("a", collisions=2),
        _rec("b", collisions=float("inf")),
        _rec("b", collisions=1),
    ]

    rows = compute_ranking(records, metric="collisions")

    assert [(row.group, row.mean, row.count) for row in rows] == [
        ("b", 1.0, 1),
        ("a", 2.0, 1),
    ]


def test_compute_ranking_ignores_metric_conversion_overflow() -> None:
    """Integer-to-float overflow must be treated like any other unavailable metric."""
    rows = compute_ranking(
        [_rec("a", collisions=10**1000), _rec("a", collisions=2)],
        metric="collisions",
    )

    assert [(row.group, row.mean, row.count) for row in rows] == [("a", 2.0, 1)]


def test_compute_ranking_keeps_large_finite_means_finite() -> None:
    """Finite inputs must not overflow the aggregate mean calculation."""
    rows = compute_ranking(
        [_rec("a", collisions=1e308), _rec("a", collisions=1e308)],
        metric="collisions",
    )

    assert len(rows) == 1
    assert math.isfinite(rows[0].mean)
    assert rows[0].mean == pytest.approx(1e308)
