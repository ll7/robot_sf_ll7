"""Tests for the frozen issue #5303 constraints-first search objective."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, Pose2D
from robot_sf.adversarial.objectives import constraints_first_lexicographic_v1, get_objective


def _evaluation(tmp_path: Path, name: str, record: dict[str, object]) -> CandidateEvaluation:
    episode_path = tmp_path / f"{name}.jsonl"
    episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return CandidateEvaluation(
        candidate=CandidateSpec(
            start=Pose2D(1.0, 2.0),
            goal=Pose2D(8.0, 2.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=7,
        ),
        certification_status=passed_status(),
        objective_value=None,
        failure_attribution=None,
        episode_record_path=episode_path,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
    )


def test_constraints_first_objective_is_registered_and_tiered(tmp_path: Path) -> None:
    """Safety, then liveness, then soft degradation occupy disjoint score bands."""
    collision = _evaluation(
        tmp_path,
        "collision",
        {"outcome": {"collision": True, "route_complete": False}, "metrics": {"snqi": 2.0}},
    )
    liveness = _evaluation(
        tmp_path,
        "liveness",
        {"outcome": {"collision": False, "route_complete": False}, "metrics": {"snqi": 0.1}},
    )
    soft = _evaluation(
        tmp_path,
        "soft",
        {
            "outcome": {"collision": False, "route_complete": True},
            "metrics": {"snqi": 10.0, "near_misses": 2},
        },
    )

    collision_score = constraints_first_lexicographic_v1(collision)
    liveness_score = constraints_first_lexicographic_v1(liveness)
    soft_score = constraints_first_lexicographic_v1(soft)

    assert get_objective("constraints_first_lexicographic_v1") is constraints_first_lexicographic_v1
    assert collision_score is not None and 4.0 <= collision_score < 5.0
    assert liveness_score is not None and 2.0 <= liveness_score < 3.0
    assert soft_score is not None and 0.0 <= soft_score < 1.0
    assert collision_score > liveness_score > soft_score


def test_constraints_first_objective_fails_closed_on_malformed_outcomes(tmp_path: Path) -> None:
    """Missing or non-boolean outcome fields cannot become liveness failures."""
    missing = _evaluation(tmp_path, "missing", {"status": "success"})
    missing_collision_evidence = _evaluation(
        tmp_path,
        "missing_collision_evidence",
        {
            "outcome": {"route_complete": True, "timeout": False},
            "metrics": {"snqi": 0.0},
        },
    )
    malformed = _evaluation(
        tmp_path,
        "malformed",
        {
            "outcome": {
                "collision": "false",
                "route_complete": True,
                "timeout": False,
            },
            "metrics": {"snqi": 0.0},
        },
    )

    assert constraints_first_lexicographic_v1(missing) is None
    assert constraints_first_lexicographic_v1(missing_collision_evidence) is None
    assert constraints_first_lexicographic_v1(malformed) is None


def test_constraints_first_objective_fails_closed_on_malformed_intrusion_metric(
    tmp_path: Path,
) -> None:
    """A non-boolean intrusion metric cannot be treated as a clean episode."""
    evaluation = _evaluation(
        tmp_path,
        "malformed_intrusion_metric",
        {
            "outcome": {"route_complete": True, "collision": False},
            "metrics": {"severe_intrusion": "false", "snqi": 0.0},
        },
    )

    assert constraints_first_lexicographic_v1(evaluation) is None


def test_constraints_first_objective_accepts_canonical_boolean_success_metric(
    tmp_path: Path,
) -> None:
    """The benchmark writer's boolean success metric remains valid evidence."""
    evaluation = _evaluation(
        tmp_path,
        "boolean_success",
        {
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "timeout_event": False,
            },
            "metrics": {"success": True, "collisions": 0, "near_misses": 0},
        },
    )

    score = constraints_first_lexicographic_v1(evaluation)

    assert score is not None
    assert 0.0 <= score < 1.0


def test_constraints_first_objective_rejects_success_metric_conflicting_with_outcome(
    tmp_path: Path,
) -> None:
    """A contradictory success metric cannot override an incomplete route outcome."""
    evaluation = _evaluation(
        tmp_path,
        "conflicting_success",
        {
            "outcome": {
                "route_complete": False,
                "collision_event": False,
                "timeout_event": False,
            },
            "metrics": {"success": True, "collisions": 0, "near_misses": 0},
        },
    )

    assert constraints_first_lexicographic_v1(evaluation) is None


def test_constraints_first_objective_rejects_out_of_domain_metrics(tmp_path: Path) -> None:
    """Negative counts and out-of-range efficiency cannot become clean outcomes."""
    base_record = {
        "outcome": {"route_complete": True, "collision": False},
        "metrics": {
            "collisions": 0,
            "near_misses": 0,
            "path_efficiency": 1.0,
            "snqi": 0.0,
        },
    }
    for field_name, invalid_value in (
        ("collisions", -1),
        ("near_misses", -1),
        ("path_efficiency", 1.1),
    ):
        record = {"outcome": dict(base_record["outcome"]), "metrics": dict(base_record["metrics"])}
        record["metrics"][field_name] = invalid_value
        evaluation = _evaluation(tmp_path, field_name, record)
        assert constraints_first_lexicographic_v1(evaluation) is None
