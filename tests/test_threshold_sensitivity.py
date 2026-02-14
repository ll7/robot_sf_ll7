"""Tests for threshold sensitivity helper calculations and replay parsing."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.metrics import EpisodeData
from robot_sf.benchmark.threshold_sensitivity import (
    SensitivityEpisode,
    analyze_threshold_sensitivity,
    comfort_exposure_ratio,
    near_miss_count,
    sensitivity_episodes_from_replay_records,
    speed_weighted_near_miss,
    ttc_gated_near_miss_count,
)


def _episode_for_distance_and_force_tests() -> EpisodeData:
    """Build a deterministic single-pedestrian episode for metric checks."""
    robot_pos = np.zeros((4, 2), dtype=float)
    robot_vel = np.zeros((4, 2), dtype=float)
    robot_acc = np.zeros((4, 2), dtype=float)
    peds_pos = np.zeros((4, 1, 2), dtype=float)
    peds_pos[:, 0, 0] = np.array([0.2, 0.3, 0.4, 0.6], dtype=float)
    ped_forces = np.zeros((4, 1, 2), dtype=float)
    ped_forces[:, 0, 0] = np.array([1.0, 2.5, 3.0, 0.5], dtype=float)
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=np.array([1.0, 0.0], dtype=float),
        dt=0.1,
        reached_goal_step=None,
    )


def _episode_for_speed_tests() -> EpisodeData:
    """Build an approaching episode with near-miss samples and finite TTC."""
    robot_pos = np.zeros((5, 2), dtype=float)
    robot_pos[:, 0] = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    robot_vel = np.zeros((5, 2), dtype=float)
    robot_vel[1:, 0] = 1.0
    robot_acc = np.zeros((5, 2), dtype=float)
    peds_pos = np.zeros((5, 1, 2), dtype=float)
    peds_pos[:, 0, 0] = 0.55
    ped_forces = np.zeros((5, 1, 2), dtype=float)
    return EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=robot_acc,
        peds_pos=peds_pos,
        ped_forces=ped_forces,
        goal=np.array([1.0, 0.0], dtype=float),
        dt=0.1,
        reached_goal_step=None,
    )


def test_near_miss_count_distance_band() -> None:
    """Near-miss counting should respect [collision, near) band semantics."""
    ep = _episode_for_distance_and_force_tests()
    result = near_miss_count(ep, collision_distance=0.25, near_miss_distance=0.5)
    assert result == 2.0


def test_comfort_exposure_ratio_uses_threshold() -> None:
    """Comfort-exposure ratio should change with the configured force threshold."""
    ep = _episode_for_distance_and_force_tests()
    result = comfort_exposure_ratio(ep, force_threshold=2.0)
    assert np.isclose(result, 0.5)


def test_speed_weighted_and_ttc_gated_near_miss() -> None:
    """Speed-aware near-miss alternatives should produce finite, non-zero values."""
    ep = _episode_for_speed_tests()
    weighted = speed_weighted_near_miss(
        ep,
        collision_distance=0.25,
        near_miss_distance=0.5,
        relative_speed_reference=1.0,
    )
    ttc_gated = ttc_gated_near_miss_count(
        ep,
        collision_distance=0.25,
        near_miss_distance=0.5,
        ttc_horizon_sec=0.4,
    )
    assert weighted > 0.0
    assert ttc_gated == 2.0


def test_analyze_threshold_sensitivity_groups_by_family() -> None:
    """Sensitivity analysis should aggregate outputs per scenario family."""
    ep_a = SensitivityEpisode("crossing:low", _episode_for_distance_and_force_tests(), "ep-a")
    ep_b = SensitivityEpisode("crossing:high", _episode_for_speed_tests(), "ep-b")
    report = analyze_threshold_sensitivity(
        [ep_a, ep_b],
        collision_grid=[0.2, 0.25],
        near_miss_grid=[0.5, 0.6],
        comfort_grid=[1.5, 2.0],
        ttc_horizons_sec=[0.3, 0.5],
        relative_speed_reference=1.0,
    )
    assert "families" in report
    assert "crossing:low" in report["families"]
    assert "crossing:high" in report["families"]


def test_sensitivity_episodes_from_replay_records_parses_family_and_forces() -> None:
    """Replay parser should emit sensitivity episodes with family metadata."""
    records = [
        {
            "episode_id": "ep-1",
            "replay_steps": [
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.0, 0.0],
            ],
            "replay_peds": [
                [[0.4, 0.0]],
                [[0.35, 0.0]],
            ],
            "replay_ped_forces": [
                [[1.0, 0.0]],
                [[2.0, 0.0]],
            ],
            "replay_dt": 0.1,
            "scenario_params": {
                "metadata": {
                    "archetype": "crossing",
                    "density": "low",
                },
            },
        },
    ]
    episodes = sensitivity_episodes_from_replay_records(records)
    assert len(episodes) == 1
    assert episodes[0].family == "crossing:low"
    assert episodes[0].data.ped_forces.shape == (2, 1, 2)
