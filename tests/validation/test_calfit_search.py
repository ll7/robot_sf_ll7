"""Bounded search and honest censoring controls; no simulation episodes."""

from scripts.validation import calfit_search_10074 as search


def test_recorded_grid_has_114_distinct_opt_in_settings():
    points = search.coarse_grid()
    assert len(points) == len({p["id"] for p in points}) == 114
    assert {p["family"] for p in points} == {
        "legacy_v1",
        "calibrated_v2",
        "gradient_v3",
        "exponential_edge",
    }
    assert all(p["desired_mean_m_s"] == 1.29 and p["desired_sd_m_s"] == 0.19 for p in points)
    assert {p["radius_m"] for p in points} == {0.25, 0.28, 0.30}
    assert {p["cap_m_s"] for p in points} == {2.0, 3.0}


def test_pareto_excludes_censored_flow_and_removes_dominated_point():
    rows = [
        {"id": "safe", "narrow_complete": True, "dense_wall_penetration_m": 0.0, "flow_score": 0.6},
        {
            "id": "fast",
            "narrow_complete": True,
            "dense_wall_penetration_m": 0.02,
            "flow_score": 1.0,
        },
        {
            "id": "dominated",
            "narrow_complete": True,
            "dense_wall_penetration_m": 0.03,
            "flow_score": 0.7,
        },
        {
            "id": "censored",
            "narrow_complete": False,
            "dense_wall_penetration_m": 0.0,
            "flow_score": None,
        },
    ]
    assert [p["id"] for p in search.pareto_front(rows)] == ["safe", "fast"]
