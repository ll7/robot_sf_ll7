"""Tests for the completed-episode objective used by the #6921 open-loop arm."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, Pose2D
from robot_sf.adversarial.objectives import (
    get_objective,
    minimize_episode_min_robot_distance,
)


def _evaluation(path: Path) -> CandidateEvaluation:
    """Build an evaluation pointing at one local episode record."""
    candidate = CandidateSpec(
        start=Pose2D(0.0, 0.0),
        goal=Pose2D(2.0, 0.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=123,
    )
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=passed_status("objective test"),
        objective_value=None,
        failure_attribution=None,
        episode_record_path=path,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
        bundle_path=None,
    )


def test_episode_min_distance_objective_is_registered() -> None:
    """The open-loop objective name must resolve through the canonical registry."""
    assert get_objective("minimize_episode_min_robot_distance") is (
        minimize_episode_min_robot_distance
    )


def test_episode_min_distance_objective_negates_finite_metric(tmp_path: Path) -> None:
    """A finite non-negative canonical metric becomes a maximization score."""
    record_path = tmp_path / "episode.jsonl"
    record_path.write_text(json.dumps({"metrics": {"min_distance": 0.75}}) + "\n", encoding="utf-8")

    assert minimize_episode_min_robot_distance(_evaluation(record_path)) == pytest.approx(-0.75)


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"min_distance": None},
        {"min_distance": "0.75"},
        {"min_distance": -0.1},
        {"min_distance": float("nan")},
        {"min_distance": float("inf")},
    ],
)
def test_episode_min_distance_objective_fails_closed(tmp_path: Path, metrics: dict) -> None:
    """Missing, malformed, negative, or non-finite metrics are unavailable."""
    record_path = tmp_path / "episode.jsonl"
    record_path.write_text(json.dumps({"metrics": metrics, "allow_nan": True}) + "\n")

    assert minimize_episode_min_robot_distance(_evaluation(record_path)) is None
