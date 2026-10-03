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


def test_real_acquisition_preserves_complete_raw_bank_and_rejects_corruption(tmp_path, monkeypatch):
    import copy
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pytest

    from robot_sf.evidence.writers import write_json, write_text
    from scripts.validation.calfit_preflight_10074 import ideal_gate_records

    point = search.candidate("gradient_v3", 0.001, 0.28, 0.28, 2.0)
    config = search.suite.load_config(search.suite.DEFAULT_CONFIG)
    config["V5"]["published_sd_m"] = 0.1
    config["V6"]["published_onset_sd_m"] = [0.1, 0.1, 0.1]
    config_path = tmp_path / "test_config.json"
    write_json(config_path, config)
    ideal = {(r["case"], r["variant"]): r for r in ideal_gate_records()}

    def synthetic(task):
        r = copy.deepcopy(ideal[task[0], task[2]])
        r["crossed"] = 60 if task[0] == "V3" else 350
        if task[0] == "V3":
            r["all_crossed"] = True
            r["flow_right_censored"] = False
        r["_positions"] = np.array([[[0.0, 0.0]], [[0.1, 0.0]]])
        r["_speeds"] = np.array([[1.0]])
        return r

    original = subprocess.check_output

    def checked(command, **kwargs):
        return "" if command[0] == "squeue" else original(command, **kwargs)

    monkeypatch.setattr(search.subprocess, "check_output", checked)
    monkeypatch.setattr(search, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(search.suite, "run_task", synthetic)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-no-simulation")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    out = tmp_path / point["id"]
    summary = search.run_candidate(point, [1001], out, config_path=config_path)
    assert summary["gate_exit"] == 0
    assert summary["flow_score"] == pytest.approx(1.0)
    assert len(list(out.glob("trajectory_*.npz"))) == 18
    identity = search.verify_run(out)
    assert identity["seeds"] == [1001]
    assert identity["candidate"] == point
    assert search.aggregate(tmp_path)["completed_n"] == 1
    raw = sorted(out.glob("trajectory_*.npz"))[0]
    assert np.array_equal(np.load(raw)["positions"], np.array([[[0.0, 0.0]], [[0.1, 0.0]]]))
    write_text(raw, "intentional corruption fixture", issue_ref="#10074")
    with pytest.raises(ValueError, match="digest mismatch"):
        search.verify_run(out)


def test_refinement_is_single_bounded_round_with_distinct_settings():
    rows = []
    for family in ["legacy_v1", "calibrated_v2", "gradient_v3", "exponential_edge"]:
        rows.append(
            {
                "candidate": search.candidate(family, 0.001, 0.28, 0.28, 2.0),
                "narrow_complete": False,
                "dense_wall_penetration_m": 0.01,
                "flow_score": None,
                "complete_narrow_measurements": 0,
            }
        )
    grid = search.refinement(rows)
    assert len(grid["candidates"]) == 32
    assert len({p["id"] for p in grid["candidates"]}) == 32
    assert set(grid["parents"]).isdisjoint(p["id"] for p in grid["candidates"])
    assert grid["seeds"] == [1001, 1002, 1003]
    assert grid["selection"] == "censored boundary screen; no numeric Pareto claim"
    assert all(0.25 <= p["radius_m"] <= 0.30 for p in grid["candidates"])
