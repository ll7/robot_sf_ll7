"""Tests for signed temporal-logic robustness objectives."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from robot_sf.adversarial.certification import failed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
)
from robot_sf.adversarial.objectives import get_objective, list_objectives
from robot_sf.adversarial.robustness import (
    RobustnessReport,
    _first_collision_time,
    compute_robustness_report,
    temporal_robustness_objective,
    write_robustness_report,
)
from robot_sf.benchmark.constants import NEAR_MISS_DIST
from robot_sf.benchmark.near_miss_ttc import DIAGNOSTIC_TTC_THRESHOLD_S

if TYPE_CHECKING:
    from pathlib import Path


def _make_episode_record(  # noqa: PLR0913
    *,
    min_clearance: float = 1.0,
    time_to_collision_min: float = 5.0,
    time_to_goal_norm: float = 0.3,
    failure_to_progress: float = 0.0,
    total_collision_count: float = 0.0,
    collision_event: bool = False,
    route_complete: bool = True,
    horizon: int = 200,
    collision_events: list[dict] | None = None,
    wall_time_sec: float | None = None,
    steps: int | None = None,
) -> dict:
    """Build a minimal synthetic episode JSONL record."""
    record: dict = {
        "version": "v1",
        "horizon": horizon,
        "metrics": {
            "min_clearance": min_clearance,
            "time_to_collision_min": time_to_collision_min,
            "time_to_goal_norm": time_to_goal_norm,
            "failure_to_progress": failure_to_progress,
            "total_collision_count": total_collision_count,
        },
        "outcome": {
            "collision_event": collision_event,
            "route_complete": route_complete,
        },
        "event_ledger": {
            "collision_events": collision_events or [],
        },
    }
    if wall_time_sec is not None:
        record["wall_time_sec"] = wall_time_sec
    if steps is not None:
        record["steps"] = steps
    return record


class TestClearanceRobustness:
    """Tests for the clearance (always d > d_safe) property."""

    def test_satisfied_when_clearance_above_threshold(self) -> None:
        record = _make_episode_record(min_clearance=1.0)
        report = compute_robustness_report(record)
        clearance = next(p for p in report.properties if p.property_name == "clearance")
        assert clearance.robustness == pytest.approx(1.0 - NEAR_MISS_DIST)
        assert not clearance.violated

    def test_violated_when_clearance_below_threshold(self) -> None:
        record = _make_episode_record(min_clearance=0.1)
        report = compute_robustness_report(record)
        clearance = next(p for p in report.properties if p.property_name == "clearance")
        assert clearance.robustness == pytest.approx(0.1 - NEAR_MISS_DIST)
        assert clearance.violated

    def test_boundary_when_clearance_equals_threshold(self) -> None:
        record = _make_episode_record(min_clearance=NEAR_MISS_DIST)
        report = compute_robustness_report(record)
        clearance = next(p for p in report.properties if p.property_name == "clearance")
        assert clearance.robustness == pytest.approx(0.0)
        assert not clearance.violated

    def test_collision_critical_time_populated(self) -> None:
        record = _make_episode_record(
            min_clearance=-0.1,
            collision_events=[{"collision_time": 1.5}],
        )
        report = compute_robustness_report(record)
        clearance = next(p for p in report.properties if p.property_name == "clearance")
        assert clearance.critical_time_s == pytest.approx(1.5)


class TestTtcRobustness:
    """Tests for the TTC (always TTC > tau) property."""

    def test_satisfied_when_ttc_above_tau(self) -> None:
        record = _make_episode_record(time_to_collision_min=5.0)
        report = compute_robustness_report(record)
        ttc = next(p for p in report.properties if p.property_name == "ttc")
        assert ttc.robustness == pytest.approx(5.0 - DIAGNOSTIC_TTC_THRESHOLD_S)
        assert not ttc.violated

    def test_violated_when_ttc_below_tau(self) -> None:
        record = _make_episode_record(time_to_collision_min=0.5)
        report = compute_robustness_report(record)
        ttc = next(p for p in report.properties if p.property_name == "ttc")
        assert ttc.robustness == pytest.approx(0.5 - DIAGNOSTIC_TTC_THRESHOLD_S)
        assert ttc.violated

    def test_custom_tau(self) -> None:
        record = _make_episode_record(time_to_collision_min=3.0)
        report = compute_robustness_report(record, tau=1.0)
        ttc = next(p for p in report.properties if p.property_name == "ttc")
        assert ttc.robustness == pytest.approx(2.0)
        assert not ttc.violated


class TestGoalRobustness:
    """Tests for the goal (eventually reach goal within T) property."""

    def test_satisfied_when_reached_early(self) -> None:
        record = _make_episode_record(
            time_to_goal_norm=0.5,
            route_complete=True,
            horizon=200,
        )
        report = compute_robustness_report(record, dt=0.1)
        goal = next(p for p in report.properties if p.property_name == "goal")
        t_total = 200 * 0.1
        t_actual = 0.5 * t_total
        assert goal.robustness == pytest.approx(t_total - t_actual)
        assert not goal.violated

    def test_violated_when_not_reached(self) -> None:
        record = _make_episode_record(
            time_to_goal_norm=1.0,
            route_complete=False,
            horizon=200,
        )
        report = compute_robustness_report(record, dt=0.1)
        goal = next(p for p in report.properties if p.property_name == "goal")
        assert goal.robustness == pytest.approx(-0.1)
        assert goal.violated

    def test_critical_time_is_actual_goal_time(self) -> None:
        record = _make_episode_record(
            time_to_goal_norm=0.25,
            route_complete=True,
            horizon=100,
        )
        report = compute_robustness_report(record, dt=0.5)
        goal = next(p for p in report.properties if p.property_name == "goal")
        t_total = 100 * 0.5
        assert goal.critical_time_s == pytest.approx(0.25 * t_total)


class TestProgressRobustness:
    """Tests for the progress (avoid sustained low-progress) property."""

    def test_satisfied_when_no_progress_failures(self) -> None:
        record = _make_episode_record(failure_to_progress=0.0)
        report = compute_robustness_report(record)
        progress = next(p for p in report.properties if p.property_name == "progress")
        assert progress.robustness == pytest.approx(0.0)
        assert not progress.violated

    def test_violated_when_progress_failures(self) -> None:
        record = _make_episode_record(failure_to_progress=3.0)
        report = compute_robustness_report(record)
        progress = next(p for p in report.properties if p.property_name == "progress")
        assert progress.robustness == pytest.approx(-3.0)
        assert progress.violated


class TestCollisionRobustness:
    """Tests for the collision (never collide) property."""

    def test_satisfied_when_no_collisions(self) -> None:
        record = _make_episode_record(total_collision_count=0.0, collision_event=False)
        report = compute_robustness_report(record)
        collision = next(p for p in report.properties if p.property_name == "collision")
        assert collision.robustness == pytest.approx(0.0)
        assert not collision.violated

    def test_violated_when_collisions(self) -> None:
        record = _make_episode_record(total_collision_count=2.0, collision_event=True)
        report = compute_robustness_report(record)
        collision = next(p for p in report.properties if p.property_name == "collision")
        assert collision.robustness == pytest.approx(-2.0)
        assert collision.violated

    def test_collision_event_flag_used_when_count_missing(self) -> None:
        record = _make_episode_record(total_collision_count=0.0, collision_event=True)
        metrics = record["metrics"]
        del metrics["total_collision_count"]
        report = compute_robustness_report(record)
        collision = next(p for p in report.properties if p.property_name == "collision")
        assert collision.robustness == pytest.approx(-1.0)
        assert collision.violated

    def test_critical_time_from_event_ledger(self) -> None:
        record = _make_episode_record(
            total_collision_count=1.0,
            collision_event=True,
            collision_events=[{"collision_time": 3.7}],
        )
        report = compute_robustness_report(record)
        collision = next(p for p in report.properties if p.property_name == "collision")
        assert collision.critical_time_s == pytest.approx(3.7)

    def test_non_violating_collision_has_no_critical_time(self) -> None:
        record = _make_episode_record(
            total_collision_count=0.0,
            collision_event=False,
            collision_events=[{"collision_time": 3.7}],
        )
        report = compute_robustness_report(record)
        collision = next(p for p in report.properties if p.property_name == "collision")
        assert collision.critical_time_s is None


class TestRobustnessReport:
    """Tests for the aggregated robustness report."""

    def test_overall_robustness_is_minimum(self) -> None:
        record = _make_episode_record(
            min_clearance=1.0,
            time_to_collision_min=5.0,
            time_to_goal_norm=0.3,
            failure_to_progress=0.0,
            total_collision_count=0.0,
        )
        report = compute_robustness_report(record)
        min_rho = min(p.robustness for p in report.properties)
        assert report.overall_robustness == pytest.approx(min_rho)

    def test_objective_is_negated_overall(self) -> None:
        record = _make_episode_record()
        report = compute_robustness_report(record)
        assert report.objective_value == pytest.approx(-report.overall_robustness)

    def test_worst_violation_dominates_objective(self) -> None:
        """The objective should be driven by the worst property."""
        record = _make_episode_record(
            min_clearance=1.0,
            time_to_collision_min=0.1,
            time_to_goal_norm=0.3,
            failure_to_progress=0.0,
            total_collision_count=0.0,
        )
        report = compute_robustness_report(record)
        ttc_rho = 0.1 - DIAGNOSTIC_TTC_THRESHOLD_S
        assert report.overall_robustness == pytest.approx(ttc_rho)
        assert report.objective_value == pytest.approx(-ttc_rho)

    def test_report_serialisation_roundtrip(self) -> None:
        record = _make_episode_record(
            min_clearance=0.3,
            total_collision_count=1.0,
            collision_event=True,
            collision_events=[{"collision_time": 2.0}],
        )
        report = compute_robustness_report(record)
        payload = report.to_json()
        restored = RobustnessReport.from_json(payload)
        assert restored.schema_version == report.schema_version
        assert len(restored.properties) == len(report.properties)
        for orig, rest in zip(report.properties, restored.properties, strict=True):
            assert orig.property_name == rest.property_name
            assert orig.robustness == pytest.approx(rest.robustness)
            assert orig.violated == rest.violated
            assert orig.critical_time_s == rest.critical_time_s
        collision = next(p for p in restored.properties if p.property_name == "collision")
        assert collision.critical_time_s == pytest.approx(2.0)


class TestWriteRobustnessReport:
    """Tests for writing robustness reports to disk."""

    def test_writes_json_file(self, tmp_path: Path) -> None:
        record = _make_episode_record()
        report = compute_robustness_report(record)
        out_path = tmp_path / "robustness_report.json"
        write_robustness_report(report, out_path)
        assert out_path.exists()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "robustness-report.v1"
        assert len(payload["properties"]) == 5

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        record = _make_episode_record()
        report = compute_robustness_report(record)
        out_path = tmp_path / "subdir" / "deep" / "robustness_report.json"
        write_robustness_report(report, out_path)
        assert out_path.exists()


class TestTemporalRobustnessObjective:
    """Tests for the registered objective function used by adversarial search."""

    def test_returns_none_when_no_record_path(self) -> None:
        candidate = CandidateSpec(
            start=Pose2D(0.0, 0.0),
            goal=Pose2D(1.0, 1.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=1,
        )
        evaluation = CandidateEvaluation(
            candidate=candidate,
            certification_status=failed_status("test"),
            objective_value=None,
            failure_attribution=None,
            episode_record_path=None,
            trajectory_csv_path=None,
            scenario_yaml_path=None,
        )
        assert temporal_robustness_objective(evaluation) is None

    def test_returns_none_when_record_missing(self, tmp_path: Path) -> None:
        candidate = CandidateSpec(
            start=Pose2D(0.0, 0.0),
            goal=Pose2D(1.0, 1.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=1,
        )
        evaluation = CandidateEvaluation(
            candidate=candidate,
            certification_status=failed_status("test"),
            objective_value=None,
            failure_attribution=None,
            episode_record_path=tmp_path / "nonexistent.jsonl",
            trajectory_csv_path=None,
            scenario_yaml_path=None,
        )
        assert temporal_robustness_objective(evaluation) is None

    def test_returns_scalar_and_writes_report(self, tmp_path: Path) -> None:
        record = _make_episode_record(
            min_clearance=0.1,
            total_collision_count=1.0,
            collision_event=True,
        )
        episode_path = tmp_path / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record), encoding="utf-8")
        bundle_path = tmp_path / "candidate_0001"
        bundle_path.mkdir()

        candidate = CandidateSpec(
            start=Pose2D(0.0, 0.0),
            goal=Pose2D(1.0, 1.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=1,
        )
        evaluation = CandidateEvaluation(
            candidate=candidate,
            certification_status=failed_status("test"),
            objective_value=None,
            failure_attribution=None,
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=None,
            bundle_path=bundle_path,
        )
        result = temporal_robustness_objective(evaluation)
        assert result is not None
        assert isinstance(result, float)
        assert math.isfinite(result)

        report_path = bundle_path / "robustness_report.json"
        assert report_path.exists()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "robustness-report.v1"
        assert len(payload["properties"]) == 5

    def test_objective_maximisation_finds_violations(self, tmp_path: Path) -> None:
        """Worse scenarios should produce larger objective values."""
        safe_record = _make_episode_record(
            min_clearance=2.0,
            time_to_collision_min=10.0,
            time_to_goal_norm=0.1,
            failure_to_progress=0.0,
            total_collision_count=0.0,
        )
        violation_record = _make_episode_record(
            min_clearance=0.1,
            time_to_collision_min=0.5,
            time_to_goal_norm=1.0,
            failure_to_progress=0.0,
            total_collision_count=0.0,
            route_complete=False,
        )

        safe_path = tmp_path / "safe.jsonl"
        safe_path.write_text(json.dumps(safe_record), encoding="utf-8")
        violation_path = tmp_path / "violation.jsonl"
        violation_path.write_text(json.dumps(violation_record), encoding="utf-8")

        candidate = CandidateSpec(
            start=Pose2D(0.0, 0.0),
            goal=Pose2D(1.0, 1.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=1,
        )

        safe_eval = CandidateEvaluation(
            candidate=candidate,
            certification_status=failed_status("test"),
            objective_value=None,
            failure_attribution=None,
            episode_record_path=safe_path,
            trajectory_csv_path=None,
            scenario_yaml_path=None,
        )
        violation_eval = CandidateEvaluation(
            candidate=candidate,
            certification_status=failed_status("test"),
            objective_value=None,
            failure_attribution=None,
            episode_record_path=violation_path,
            trajectory_csv_path=None,
            scenario_yaml_path=None,
        )
        safe_score = temporal_robustness_objective(safe_eval)
        violation_score = temporal_robustness_objective(violation_eval)
        assert safe_score is not None
        assert violation_score is not None
        assert violation_score > safe_score


class TestFirstCollisionTime:
    """Tests for the collision time extraction helper."""

    def test_returns_none_for_empty_ledger(self) -> None:
        assert _first_collision_time({}) is None
        assert _first_collision_time({"collision_events": []}) is None

    def test_returns_first_time(self) -> None:
        ledger = {"collision_events": [{"collision_time": 1.5}, {"collision_time": 3.0}]}
        assert _first_collision_time(ledger) == pytest.approx(1.5)

    def test_returns_none_for_invalid_time(self) -> None:
        ledger = {"collision_events": [{"collision_time": None}]}
        assert _first_collision_time(ledger) is None

    def test_skips_malformed_or_non_finite_events(self) -> None:
        ledger = {
            "collision_events": [
                {"collision_time": "not-a-time"},
                {"collision_time": float("nan")},
                {"collision_time": 3.0},
            ]
        }
        assert _first_collision_time(ledger) == pytest.approx(3.0)


class TestObjectiveRegistry:
    """Tests for the temporal_robustness objective registration."""

    def test_temporal_robustness_is_registered(self) -> None:
        objectives = list_objectives()
        assert "temporal_robustness" in objectives

    def test_get_objective_returns_function(self) -> None:
        obj = get_objective("temporal_robustness")
        assert callable(obj)

    def test_per_property_robustness_preserved_in_report(self, tmp_path: Path) -> None:
        """Per-property values AND critical timestamps must be preserved."""
        record = _make_episode_record(
            min_clearance=0.1,
            time_to_collision_min=0.5,
            time_to_goal_norm=1.0,
            failure_to_progress=2.0,
            total_collision_count=1.0,
            collision_event=True,
            route_complete=False,
            collision_events=[{"collision_time": 1.2}],
        )
        report = compute_robustness_report(record)
        assert len(report.properties) == 5

        names = {p.property_name for p in report.properties}
        assert names == {"clearance", "ttc", "goal", "progress", "collision"}

        collision = next(p for p in report.properties if p.property_name == "collision")
        assert collision.critical_time_s == pytest.approx(1.2)
        assert collision.violated

        progress = next(p for p in report.properties if p.property_name == "progress")
        assert progress.violated
        assert progress.robustness == pytest.approx(-2.0)


class TestDtDerivation:
    """Tests for automatic dt derivation from episode metadata."""

    def test_dt_from_wall_time_and_steps(self) -> None:
        record = _make_episode_record(wall_time_sec=20.0, steps=200)
        report = compute_robustness_report(record)
        goal = next(p for p in report.properties if p.property_name == "goal")
        expected_dt = 20.0 / 200.0
        t_total = 200 * expected_dt
        t_actual = 0.3 * t_total
        assert goal.robustness == pytest.approx(t_total - t_actual)

    def test_dt_defaults_to_0_1(self) -> None:
        record = _make_episode_record()
        report = compute_robustness_report(record)
        goal = next(p for p in report.properties if p.property_name == "goal")
        t_total = 200 * 0.1
        t_actual = 0.3 * t_total
        assert goal.robustness == pytest.approx(t_total - t_actual)

    @pytest.mark.parametrize(
        ("horizon", "wall_time_sec", "steps"),
        [(None, 20.0, 200), ("bad", "bad", "bad"), (0, float("nan"), 0)],
    )
    def test_malformed_metadata_falls_back_to_safe_defaults(
        self,
        horizon: object,
        wall_time_sec: object,
        steps: object,
    ) -> None:
        record = _make_episode_record()
        record["horizon"] = horizon
        record["wall_time_sec"] = wall_time_sec
        record["steps"] = steps
        report = compute_robustness_report(record)
        goal = next(p for p in report.properties if p.property_name == "goal")
        assert goal.robustness == pytest.approx(14.0)

    @pytest.mark.parametrize("name, value", [("dt", 0.0), ("dt", float("nan")), ("tau", -1.0)])
    def test_invalid_explicit_semantic_parameters_raise(self, name: str, value: float) -> None:
        kwargs = {name: value}
        with pytest.raises(ValueError, match=f"{name} must be a finite positive float"):
            compute_robustness_report(_make_episode_record(), **kwargs)
