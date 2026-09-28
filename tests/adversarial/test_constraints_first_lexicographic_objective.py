"""Tests for the frozen issue #5303 constraints-first search objective."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, Pose2D
from robot_sf.adversarial.objectives import (
    constraints_first_lexicographic_v1,
    get_objective,
)
from robot_sf.adversarial.objectives_v2 import (
    constraints_first_lexicographic_v2,
    constraints_first_outcome_projection_v2,
)


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
                "severe_intrusion_event": False,
                "timeout_event": False,
            },
            "metrics": {"success": True, "collisions": 0, "near_misses": 0},
        },
    )

    score = constraints_first_lexicographic_v1(evaluation)

    assert score is not None
    assert 0.0 <= score < 1.0


def test_constraints_first_v2_keeps_partial_negative_safety_evidence_unknown(
    tmp_path: Path,
) -> None:
    """The #9645-shaped row cannot score as safe when intrusion evidence is absent."""
    evaluation = _evaluation(
        tmp_path,
        "partial_safety_evidence",
        {
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "timeout_event": False,
            },
            "metrics": {"success": True, "collisions": 0, "near_misses": 0},
        },
    )
    record = json.loads(evaluation.episode_record_path.read_text(encoding="utf-8"))

    assert get_objective("constraints_first_lexicographic_v2") is constraints_first_lexicographic_v2
    assert constraints_first_outcome_projection_v2(record)["status"] == "not_available"
    assert constraints_first_lexicographic_v2(evaluation) is None
    # V1 is retained unchanged for byte-bound historical contracts; new work uses v2.
    assert constraints_first_lexicographic_v1(evaluation) is not None


def test_constraints_first_v2_requires_both_negative_safety_components(
    tmp_path: Path,
) -> None:
    """Either missing component keeps a negative safety conclusion unavailable."""
    intrusion_only = _evaluation(
        tmp_path,
        "intrusion_only_negative",
        {
            "outcome": {
                "route_complete": True,
                "severe_intrusion_event": False,
                "timeout_event": False,
            },
            "metrics": {"success": True, "near_misses": 0},
        },
    )
    fully_observed_clear = _evaluation(
        tmp_path,
        "fully_observed_clear",
        {
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "severe_intrusion_event": False,
                "timeout_event": False,
            },
            "metrics": {"success": True, "collisions": 0, "near_misses": 0},
        },
    )

    assert constraints_first_lexicographic_v2(intrusion_only) is None
    clear_score = constraints_first_lexicographic_v2(fully_observed_clear)
    assert clear_score is not None
    assert 0.0 <= clear_score < 1.0


def test_constraints_first_v2_keeps_known_positive_safety_evidence_critical(
    tmp_path: Path,
) -> None:
    """A confirmed safety failure remains critical if the other component is unknown."""
    collision = _evaluation(
        tmp_path,
        "collision_intrusion_unknown",
        {
            "outcome": {
                "route_complete": False,
                "collision_event": True,
                "timeout_event": False,
            },
            "metrics": {"success": False, "collisions": 1, "near_misses": 0},
        },
    )
    intrusion = _evaluation(
        tmp_path,
        "intrusion_collision_unknown",
        {
            "outcome": {
                "route_complete": False,
                "severe_intrusion_event": True,
                "timeout_event": False,
            },
            "metrics": {"success": False, "severe_intrusion": True, "near_misses": 1},
        },
    )

    collision_score = constraints_first_lexicographic_v2(collision)
    intrusion_score = constraints_first_lexicographic_v2(intrusion)

    assert collision_score is not None and 4.0 <= collision_score < 5.0
    assert intrusion_score is not None and 4.0 <= intrusion_score < 5.0


def test_constraints_first_v2_positive_component_survives_other_component_conflict(
    tmp_path: Path,
) -> None:
    """A conflict local to one component cannot hide independent positive evidence."""
    confirmed_collision = _evaluation(
        tmp_path,
        "collision_with_malformed_intrusion",
        {
            "outcome": {
                "route_complete": False,
                "collision_event": True,
                "severe_intrusion": False,
                "severe_intrusion_event": True,
                "timeout_event": False,
            },
            "metrics": {
                "success": False,
                "collisions": 1,
                "severe_intrusion": True,
                "near_misses": 1,
            },
        },
    )
    confirmed_intrusion = _evaluation(
        tmp_path,
        "intrusion_with_conflicting_collision",
        {
            "outcome": {
                "route_complete": False,
                "collision": False,
                "collision_event": True,
                "severe_intrusion_event": True,
                "timeout_event": False,
            },
            "metrics": {"success": False, "severe_intrusion": True, "near_misses": 1},
        },
    )

    collision_score = constraints_first_lexicographic_v2(confirmed_collision)
    intrusion_score = constraints_first_lexicographic_v2(confirmed_intrusion)

    assert collision_score is not None and 4.0 <= collision_score < 5.0
    assert intrusion_score is not None and 4.0 <= intrusion_score < 5.0


def test_constraints_first_v2_same_component_source_conflict_stays_unknown(
    tmp_path: Path,
) -> None:
    """A collision flag that conflicts with its metric is not treated as confirmed."""
    evaluation = _evaluation(
        tmp_path,
        "collision_source_conflict_no_intrusion",
        {
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "severe_intrusion_event": False,
                "timeout_event": False,
            },
            "metrics": {"success": True, "collisions": 1, "severe_intrusion": False},
        },
    )

    assert constraints_first_lexicographic_v2(evaluation) is None


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


def test_constraints_first_objective_rejects_conflicting_collision_aliases(
    tmp_path: Path,
) -> None:
    """Contradictory canonical collision aliases cannot be combined with boolean OR."""
    evaluation = _evaluation(
        tmp_path,
        "conflicting_collision_aliases",
        {
            "outcome": {
                "route_complete": False,
                "collision": False,
                "collision_event": True,
            },
            "metrics": {"success": False, "near_misses": 0},
        },
    )

    assert constraints_first_lexicographic_v1(evaluation) is None


def test_constraints_first_objective_rejects_collision_metric_conflicting_with_outcome(
    tmp_path: Path,
) -> None:
    """Canonical collision flags and metrics must agree when both are present."""
    for name, route_complete, collision_event, collisions in (
        ("metric_reports_collision", True, False, 1),
        ("outcome_reports_collision", False, True, 0),
    ):
        evaluation = _evaluation(
            tmp_path,
            name,
            {
                "outcome": {
                    "route_complete": route_complete,
                    "collision_event": collision_event,
                    "timeout_event": False,
                },
                "metrics": {
                    "success": route_complete,
                    "collisions": collisions,
                    "near_misses": 0,
                },
            },
        )

        assert constraints_first_lexicographic_v1(evaluation) is None


def test_constraints_first_objective_combines_distinct_collision_and_intrusion_evidence(
    tmp_path: Path,
) -> None:
    """A severe intrusion remains a safety failure when collision evidence is clean."""
    evaluation = _evaluation(
        tmp_path,
        "severe_intrusion_without_collision",
        {
            "outcome": {
                "route_complete": False,
                "collision_event": False,
                "severe_intrusion": True,
                "timeout_event": False,
            },
            "metrics": {
                "success": False,
                "collisions": 0,
                "severe_intrusion": True,
                "near_misses": 1,
            },
        },
    )

    score = constraints_first_lexicographic_v1(evaluation)

    assert score is not None
    assert 4.0 <= score < 5.0


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
