from __future__ import annotations

from robot_sf.benchmark.seed_variance import compute_seed_variance


def _make_record(group: str, seed: int, metrics: dict[str, float]) -> dict:
    return {
        "episode_id": f"{group}-{seed}",
        "scenario_id": group,
        "seed": seed,
        "metrics": metrics,
    }


def test_compute_seed_variance_basic():
    records = [
        _make_record("g1", 0, {"a": 1.0, "b": 2.0}),
        _make_record("g1", 1, {"a": 3.0, "b": 2.0}),
        _make_record("g2", 0, {"a": 10.0}),
    ]
    res = compute_seed_variance(records, group_by="scenario_id")
    assert "g1" in res and "g2" in res
    a_stats = res["g1"]["a"]
    # mean for a in g1: (1+3)/2 = 2
    assert abs(a_stats["mean"] - 2.0) < 1e-9
    # std with population (ddof=0): sqrt(((1-2)^2 + (3-2)^2)/2) = 1.0
    assert abs(a_stats["std"] - 1.0) < 1e-9
    # cv = std/mean = 0.5
    assert abs(a_stats["cv"] - 0.5) < 1e-9
    # metric missing for some seeds should be ignored; count equals number of finite values
    assert res["g2"]["a"]["count"] == 1.0


def test_compute_seed_variance_metric_filter():
    records = [
        _make_record("g", 0, {"a": 1.0, "b": 2.0}),
        _make_record("g", 1, {"a": 3.0, "b": 4.0}),
    ]
    res = compute_seed_variance(records, metrics=["a"], group_by="scenario_id")
    assert set(res["g"].keys()) == {"a"}
