"""Independent trajectory oracles for the dev-only manipulation diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.training.scenario_loader import build_robot_config_from_scenario


@pytest.mark.parametrize("finite", [True, False])
def test_advance_orders_behavior_before_physics_and_rejects_nonfinite(finite):
    """Behavior refresh must precede physics, and invalid state must fail closed."""
    from robot_sf.benchmark.pedestrian_manipulation_checks import advance

    events = []
    sim = SimpleNamespace(
        step=lambda: events.append("physics"),
        peds=SimpleNamespace(state=np.array([[0.0 if finite else np.nan]])),
    )
    behaviors = [
        SimpleNamespace(step=lambda: events.append("first")),
        SimpleNamespace(step=lambda: events.append("second")),
    ]
    if finite:
        advance(sim, behaviors)
    else:
        with pytest.raises(ValueError, match="nonfinite manipulation trajectory"):
            advance(sim, behaviors)
    assert events == ["first", "second", "physics"]


def _scripted_sim(actors, obstacles, passed, step_count):
    """Supply prescribed positions/velocities without replacing diagnostic arithmetic."""
    count = len(actors)

    def step():
        step_count[0] += 1

    def pos():
        if count == 32:
            return np.zeros((32, 2))
        if obstacles:
            if passed and step_count[0] >= 2:
                return np.array([[8.5, 3.0]])
            return np.array([[8.0, obstacles[0].geometry.bounds[3] + 0.1]])
        if passed and step_count[0] >= 2:
            return np.array([[8.0, 3.0], [7.5, 3.0]])
        return np.array([[7.0, 3.0], [8.0, 3.0]])

    def vel():
        if count == 32:
            # Warmup is deliberately different from the 5..10 s observation window.
            values = [9.0, 9.0] if step_count[0] <= 50 else [2.0, 4.0]
            return np.column_stack((np.tile(values, 16), np.zeros(32)))
        return np.tile([1.0, 0.0], (count, 1))

    return SimpleNamespace(
        step=step,
        peds=SimpleNamespace(
            state=np.zeros((count, 6)),
            max_speeds=np.tile([1.0, 3.0], count // 2) if count == 32 else np.ones(count),
            agent_radius=0.25,
            pos=pos,
            vel=vel,
        ),
        config=SimpleNamespace(
            obstacle_force_config=SimpleNamespace(
                law_version="fixture-law", factor=2.0, threshold=0.1
            )
        ),
    )


@pytest.mark.parametrize("passed", [True, False], ids=["passage", "timeout"])
def test_manipulation_report_uses_time_window_contact_and_surface_distance(monkeypatch, passed):
    """Hand traces distinguish warmup, contact, crossing, and capped non-passage."""
    from robot_sf.benchmark import pedestrian_manipulation_checks as checks

    calls = []
    steps = []
    profile = {"ped_radius": 0.25}

    def build_probe(bound_profile, seed, actors, obstacles, *, width=120, height=140):
        assert bound_profile == profile
        assert seed == 1001
        calls.append((actors, obstacles, width, height))
        step_count = [0]
        steps.append(step_count)
        return _scripted_sim(actors, obstacles, passed, step_count), [], 0.25

    monkeypatch.setattr(checks, "build_probe", build_probe)
    report = checks.manipulation_checks(profile, [1001])
    assert report["evidence_status"] == "diagnostic-only"
    speed = report["free_speed"]
    assert speed["desired_mean_m_s"] == 2.0
    assert speed["desired_sd_m_s"] == 1.0
    assert speed["realized_mean_m_s"] == 3.0
    assert speed["realized_sd_m_s"] == 1.0
    assert speed["raw"] == [
        {
            "seed": 1001,
            "desired_m_s": [1.0, 3.0] * 16,
            "realized_5_to_10_s_m_s": [2.0, 4.0] * 16,
            "force_radius_m": 0.25,
            "collision_radius_m": 0.25,
            "wall_law": "fixture-law",
            "wall_factor": 2.0,
            "wall_threshold_m": 0.1,
        }
    ]
    assert [row["width_m"] for row in report["doorway"]] == [0.8, 1.0, 1.2, 1.4]
    for row in report["doorway"]:
        assert row["seed"] == 1001
        assert row["passage_s"] == (pytest.approx(0.2) if passed else None)
        assert row["contact_steps"] == (1 if passed else 400)
        assert row["min_surface_clearance_m"] == pytest.approx(-0.15)
        assert len(row["trace_xy_speed"]) == (2 if passed else 400)
        assert row["trace_xy_speed"][0] == pytest.approx([8.0, 3.0 - row["width_m"] / 2 + 0.1, 1.0])
        if passed:
            assert row["trace_xy_speed"][-1] == [8.5, 3.0, 1.0]
    head_on = report["head_on"][0]
    assert head_on["seed"] == 1001
    assert head_on["passed"] is passed
    assert head_on["min_centre_distance_m"] == (0.5 if passed else 1.0)
    assert head_on["min_surface_clearance_m"] == (0.0 if passed else 0.5)
    assert head_on["trace_xy"] == (
        [[[7.0, 3.0], [8.0, 3.0]], [[8.0, 3.0], [7.5, 3.0]]]
        if passed
        else [[[7.0, 3.0], [8.0, 3.0]]] * 200
    )
    assert [step[0] for step in steps] == (
        [100, 2, 2, 2, 2, 2] if passed else [100, 400, 400, 400, 400, 200]
    )
    assert len(calls) == 6
    assert (len(calls[0][0]), calls[0][1:]) == (32, ([], 120, 140))
    for actors, obstacles, width, height in calls[1:5]:
        assert len(actors) == 1
        assert len(obstacles) == 2
        assert (width, height) == (16, 6)
    assert (len(calls[-1][0]), calls[-1][1:]) == (2, ([], 16, 6))
    assert report["targets"]["free_speed_mean_sd_m_s"] == [1.29, 0.19]
    assert "onset, not minimum" in report["targets"]["head_on"]
    assert "No body rotation" in report["limitations"]


def test_native_manipulation_report_binds_profile_and_retains_finite_traces():
    """Exercise all three diagnostics through the real scenario/physics substrate."""
    profile = {
        "ped_radius": 0.25,
        "ped_force_radius": 0.25,
        "desired_speed_mean": 1.0,
        "desired_speed_std": 0.0,
        "desired_speed_truncated": True,
    }
    # Check the existing loader first so base proof names the unsupported profile fields.
    config = build_robot_config_from_scenario(
        {"simulation_config": profile}, scenario_path=Path(__file__)
    ).sim_config
    assert config.ped_radius == config.ped_force_radius == 0.25
    from robot_sf.benchmark.pedestrian_manipulation_checks import manipulation_checks

    report = manipulation_checks(profile, [1001])
    speed = report["free_speed"]
    assert speed["desired_mean_m_s"] == 1.0
    assert speed["desired_sd_m_s"] == 0.0
    raw = speed["raw"][0]
    assert raw["seed"] == 1001
    assert raw["desired_m_s"] == [1.0] * 32
    assert raw["force_radius_m"] == raw["collision_radius_m"] == 0.25
    assert np.all(np.isfinite(raw["realized_5_to_10_s_m_s"]))
    assert all(value > 0 for value in raw["realized_5_to_10_s_m_s"])
    assert len(report["doorway"]) == 4
    for row in report["doorway"]:
        trace = np.asarray(row["trace_xy_speed"])
        assert np.all(np.isfinite(trace))
        assert 1 <= len(trace) <= 400
        aperture = row["width_m"]
        lower, upper = 3 - aperture / 2, 3 + aperture / 2
        x, y = trace[:, 0], trace[:, 1]
        dx = np.maximum(np.maximum(7.9 - x, x - 8.1), 0)
        lower_dy = np.maximum(np.maximum(-y, y - lower), 0)
        upper_dy = np.maximum(np.maximum(upper - y, y - 6), 0)
        gaps = np.minimum(np.hypot(dx, lower_dy), np.hypot(dx, upper_dy)) - 0.25
        assert row["min_surface_clearance_m"] == pytest.approx(float(np.min(gaps)))
        assert row["contact_steps"] == int(np.count_nonzero(gaps < 0))
        if row["passage_s"] is None:
            assert len(trace) == 400
            assert not np.any(x > 8.35)
        else:
            assert row["passage_s"] == pytest.approx(len(trace) * 0.1)
            assert x[-1] > 8.35
            assert not np.any(x[:-1] > 8.35)
    head_on = report["head_on"][0]
    trace = np.asarray(head_on["trace_xy"])
    assert np.all(np.isfinite(trace))
    centre_distances = np.sqrt(np.sum((trace[:, 0] - trace[:, 1]) ** 2, axis=1))
    assert head_on["min_centre_distance_m"] == pytest.approx(float(np.min(centre_distances)))
    assert head_on["min_surface_clearance_m"] == pytest.approx(
        float(np.min(centre_distances)) - 0.5
    )
    assert head_on["passed"] == bool(trace[-1, 0, 0] > trace[-1, 1, 0])
    if not head_on["passed"]:
        assert len(trace) == 200
