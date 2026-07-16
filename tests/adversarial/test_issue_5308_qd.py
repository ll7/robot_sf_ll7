"""CPU tests for the MAP-Elites quality-diversity adversarial search (issue #5308)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.certification import failed_status, not_available_status, passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    RangeConfig,
    SearchSpaceConfig,
)
from robot_sf.adversarial.qd import (
    QD_ARCHIVE_SCHEMA_VERSION,
    GridSpec,
    QDArchive,
    QDComparisonReport,
    QDSearchConfig,
    QDSearchResult,
    compare_qd_vs_single_objective,
    default_behavior_descriptor,
    run_map_elites,
    write_qd_archive,
)


def _space() -> SearchSpaceConfig:
    """Build a 2D search space spanning start/goal x with fixed seeds."""
    return SearchSpaceConfig(
        start_x=RangeConfig(0.0, 4.0),
        start_y=RangeConfig(0.0, 0.0),
        goal_x=RangeConfig(0.0, 4.0),
        goal_y=RangeConfig(0.0, 0.0),
        spawn_time_s=RangeConfig(0.0, 0.0),
        pedestrian_speed_mps=RangeConfig(1.0, 1.0),
        pedestrian_delay_s=RangeConfig(0.0, 0.0),
        scenario_seed=RangeConfig(1.0, 1.0),
    )


def _record(
    min_distance: float, critical_time: float, failure: str = "collision"
) -> dict[str, Any]:
    """Build a minimal episode record carrying the QD behavior descriptors."""
    outcome: dict[str, Any] = {"route_complete": False}
    if failure == "collision":
        outcome["collision"] = True
        termination = "collision"
    elif failure == "timeout":
        outcome["timeout"] = True
        termination = "timeout"
    else:
        termination = "incomplete"
    return {
        "status": "completed",
        "termination_reason": termination,
        "outcome": outcome,
        "metrics": {
            "distance_to_human_min": min_distance,
            "time_to_collision_min": critical_time,
        },
    }


def _make_evaluation(
    *,
    candidate: CandidateSpec,
    min_distance: float,
    critical_time: float,
    objective: float,
    failure: str = "collision",
    cert_status: Any = None,
    bundle_index: int = 0,
    temp_root: Path,
) -> CandidateEvaluation:
    """Build a candidate evaluation with a measurable behavior descriptor."""
    record = _record(min_distance, critical_time, failure=failure)

    bundle = temp_root / f"qd_bundle_dummy_{bundle_index}"
    bundle.mkdir(parents=True, exist_ok=True)
    episode_path = bundle / "episode_records.jsonl"
    episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=cert_status or not_available_status("advisory"),
        objective_value=objective,
        failure_attribution=attribution_from_episode_record(record),
        episode_record_path=episode_path,
        trajectory_csv_path=None,
        scenario_yaml_path=temp_root / "qd_scenario_dummy.yaml",
        bundle_path=bundle,
    )


class _FakeEvaluator:
    """Deterministic injected evaluator that returns a scenario from a queue."""

    def __init__(self, evaluations: list[CandidateEvaluation]) -> None:
        self._evaluations = list(evaluations)

    def __call__(self, config: QDSearchConfig, candidate: CandidateSpec) -> CandidateEvaluation:
        if not self._evaluations:
            raise AssertionError("evaluator queue exhausted")
        return replace(self._evaluations.pop(0), candidate=candidate)


def _candidates(count: int) -> list[CandidateSpec]:
    """Build ``count`` distinct candidates across the search space."""
    return [
        CandidateSpec(
            start=Pose2D(float(i) * 0.4, 0.0),
            goal=Pose2D(4.0 - float(i) * 0.4, 0.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=1,
        )
        for i in range(count)
    ]


def _spanning_evaluations(count: int, *, temp_root: Path) -> list[CandidateEvaluation]:
    """Build evaluations whose descriptors span the full 2D grid."""
    evals: list[CandidateEvaluation] = []
    candidates = _candidates(count)
    for i in range(count):
        distance = 2.5 * ((i % 4) / 3.0)
        critical_time = 3.0 * ((i % 3) / 2.0)
        evals.append(
            _make_evaluation(
                candidate=candidates[i],
                min_distance=distance,
                critical_time=critical_time,
                objective=float(i),
                failure=["collision", "timeout", "incomplete"][i % 3],
                cert_status=passed_status("synthetic fixture"),
                bundle_index=i,
                temp_root=temp_root,
            )
        )
    return evals


def test_grid_cell_index_and_coverage() -> None:
    """GridSpec maps descriptors to cells and reports coverage correctly."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    assert grid.cell_count == 16
    assert grid.cell_index((0.0, 0.0)) == (0, 0)
    assert grid.cell_index((2.5, 3.0)) == (3, 3)
    assert grid.cell_index((-1.0, 1.0)) is None
    assert grid.cell_index((1.0, 5.0)) is None


def test_grid_spec_rejects_invalid_bounds() -> None:
    """GridSpec must reject non-finite bounds and zero bins."""
    with pytest.raises(ValueError):
        GridSpec(x_min=float("nan"), x_max=1.0, y_min=0.0, y_max=1.0)
    with pytest.raises(ValueError):
        GridSpec(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, bins=0)
    with pytest.raises(ValueError, match="x_max"):
        GridSpec(x_min=1.0, x_max=1.0, y_min=0.0, y_max=1.0)
    with pytest.raises(ValueError, match="y_max"):
        GridSpec(x_min=0.0, x_max=1.0, y_min=2.0, y_max=1.0)


def test_behavior_descriptor_reads_metrics(tmp_path: Path) -> None:
    """Default descriptor pulls (min distance, critical time) from the record."""
    evaluation = _make_evaluation(
        candidate=_candidates(1)[0],
        min_distance=1.25,
        critical_time=2.0,
        objective=3.0,
        temp_root=tmp_path,
    )
    assert default_behavior_descriptor(evaluation) == (1.25, 2.0)


def test_behavior_descriptor_none_when_unmeasurable(tmp_path: Path) -> None:
    """Descriptor returns None when the record/metrics are missing."""
    evaluation = _make_evaluation(
        candidate=_candidates(1)[0],
        min_distance=1.0,
        critical_time=1.0,
        objective=1.0,
        temp_root=tmp_path,
    )
    evaluation = evaluation.__class__(
        candidate=evaluation.candidate,
        certification_status=evaluation.certification_status,
        objective_value=evaluation.objective_value,
        failure_attribution=evaluation.failure_attribution,
        episode_record_path=None,
        trajectory_csv_path=None,
        scenario_yaml_path=evaluation.scenario_yaml_path,
        bundle_path=evaluation.bundle_path,
    )
    assert default_behavior_descriptor(evaluation) is None


def test_archive_admits_only_certified_finite(tmp_path: Path) -> None:
    """Archive rejects failed certification and non-finite objectives."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    archive = QDArchive(grid=grid, require_certification=True)

    valid = _make_evaluation(
        candidate=_candidates(1)[0],
        min_distance=1.0,
        critical_time=1.0,
        objective=5.0,
        cert_status=passed_status("ok"),
        temp_root=tmp_path,
    )
    assert archive.try_insert(
        descriptor=(1.0, 1.0), evaluation=valid, certification_status=valid.certification_status
    )

    failed = _make_evaluation(
        candidate=_candidates(2)[1],
        min_distance=2.0,
        critical_time=2.0,
        objective=9.0,
        cert_status=failed_status("excluded"),
        temp_root=tmp_path,
    )
    assert not archive.try_insert(
        descriptor=(2.0, 2.0), evaluation=failed, certification_status=failed.certification_status
    )

    nonfinite = _make_evaluation(
        candidate=_candidates(3)[2],
        min_distance=0.5,
        critical_time=0.5,
        objective=float("nan"),
        temp_root=tmp_path,
    )
    assert not archive.try_insert(
        descriptor=(0.5, 0.5),
        evaluation=nonfinite,
        certification_status=nonfinite.certification_status,
    )
    assert archive.filled_cell_count() == 1


def test_archive_keeps_higher_quality_incumbent(tmp_path: Path) -> None:
    """A new elite only replaces the incumbent at its cell when scoring higher."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    archive = QDArchive(grid=grid)

    low = _make_evaluation(
        candidate=_candidates(1)[0],
        min_distance=1.0,
        critical_time=1.0,
        objective=2.0,
        temp_root=tmp_path,
    )
    assert archive.try_insert(descriptor=(1.0, 1.0), evaluation=low, certification_status=None)
    high = _make_evaluation(
        candidate=_candidates(2)[1],
        min_distance=1.0,
        critical_time=1.0,
        objective=8.0,
        temp_root=tmp_path,
    )
    assert archive.try_insert(descriptor=(1.0, 1.0), evaluation=high, certification_status=None)
    worse = _make_evaluation(
        candidate=_candidates(3)[2],
        min_distance=1.0,
        critical_time=1.0,
        objective=1.0,
        temp_root=tmp_path,
    )
    assert not archive.try_insert(
        descriptor=(1.0, 1.0), evaluation=worse, certification_status=None
    )
    assert archive.cells[(1, 1)].objective_value == 8.0


def test_run_map_elites_populates_grid_artifact(tmp_path: Path) -> None:
    """run_map_elites fills distinct grid cells and writes the archive artifact."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    config = QDSearchConfig(
        search_space=_space(),
        objective="worst_case_snqi",
        grid=grid,
        budget=12,
        seed=7,
        require_certification=True,
    )
    evaluator = _FakeEvaluator(_spanning_evaluations(12, temp_root=tmp_path))
    result = run_map_elites(config, evaluator=evaluator)

    assert isinstance(result, QDSearchResult)
    assert result.num_evaluated == 12
    assert result.archive.filled_cell_count() >= 2
    assert result.archive.coverage_fraction() > 0.0
    assert len(result.archive.distinct_failure_modes()) >= 2

    artifact_path = write_qd_archive(result, tmp_path / "archive.json")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == QD_ARCHIVE_SCHEMA_VERSION
    assert payload["behavior_axes"] == ["distance_to_human_min", "time_to_collision_min"]
    assert payload["summary"]["filled_cell_count"] == result.archive.filled_cell_count()


def test_run_map_elites_is_reproducible(tmp_path: Path) -> None:
    """Same seed + config + evaluator ordering yields identical archive coverage."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    config = QDSearchConfig(
        search_space=_space(), objective="worst_case_snqi", grid=grid, budget=11, seed=7
    )
    a = run_map_elites(
        config, evaluator=_FakeEvaluator(_spanning_evaluations(11, temp_root=tmp_path))
    )
    b = run_map_elites(
        config, evaluator=_FakeEvaluator(_spanning_evaluations(11, temp_root=tmp_path))
    )
    assert a.to_json() == b.to_json()


def test_compare_qd_vs_single_objective(tmp_path: Path) -> None:
    """Equal-budget comparison reports QD coverage vs single-objective diversity."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    config = QDSearchConfig(
        search_space=_space(),
        objective="worst_case_snqi",
        grid=grid,
        budget=12,
        seed=7,
        require_certification=True,
    )
    qd_result = run_map_elites(
        config, evaluator=_FakeEvaluator(_spanning_evaluations(12, temp_root=tmp_path / "qd"))
    )

    single_objective = _spanning_evaluations(12, temp_root=tmp_path / "single")
    report = compare_qd_vs_single_objective(
        qd_result=qd_result,
        single_objective_evaluations=single_objective,
        budget=12,
        grid=grid,
        require_certification=True,
    )
    assert isinstance(report, QDComparisonReport)
    assert report.qd.filled_cells == qd_result.archive.filled_cell_count()
    assert report.qd.distinct_failure_modes >= 2
    assert report.single_objective.method == "single_objective"
    assert report.qd.distinct_failure_modes >= report.single_objective.distinct_failure_modes

    report_path = tmp_path / "comparison.json"
    report_path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["comparison_type"] == "equal_budget_qd_vs_single_objective"


def test_qd_finds_more_distinct_modes_than_single_objective_baseline(tmp_path: Path) -> None:
    """QD must expose more distinct synthetic failure mechanisms than the SO baseline.

    The single-objective baseline here converges on one failure mode at this budget
    while MAP-Elites spreads across the grid; this is the core QD claim at equal budget.
    """
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    config = QDSearchConfig(
        search_space=_space(),
        objective="worst_case_snqi",
        grid=grid,
        budget=12,
        seed=7,
        require_certification=True,
    )
    qd_result = run_map_elites(
        config, evaluator=_FakeEvaluator(_spanning_evaluations(12, temp_root=tmp_path / "qd"))
    )

    single_objective = [
        _make_evaluation(
            candidate=_candidates(1)[0],
            min_distance=1.0,
            critical_time=1.0,
            objective=9.0,
            failure="collision",
            cert_status=passed_status("synthetic fixture"),
            bundle_index=100 + j,
            temp_root=tmp_path / "single",
        )
        for j in range(12)
    ]
    report = compare_qd_vs_single_objective(
        qd_result=qd_result,
        single_objective_evaluations=single_objective,
        budget=12,
        grid=grid,
        require_certification=True,
    )
    assert report.qd.distinct_failure_modes > report.single_objective.distinct_failure_modes


def test_qd_config_and_emitters_fail_closed(tmp_path: Path) -> None:
    """Invalid budgets and empty explicit emitter sets must not look successful."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    with pytest.raises(ValueError, match="budget"):
        QDSearchConfig(search_space=_space(), objective="worst_case_snqi", grid=grid, budget=0)
    config = QDSearchConfig(search_space=_space(), objective="worst_case_snqi", grid=grid, budget=1)
    with pytest.raises(ValueError, match="emitters"):
        run_map_elites(
            config,
            evaluator=_FakeEvaluator(_spanning_evaluations(1, temp_root=tmp_path)),
            emitters=[],
        )


def test_single_objective_comparison_uses_archive_admission(tmp_path: Path) -> None:
    """The baseline row excludes failed, non-finite, and out-of-grid evaluations."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    candidates = _candidates(4)
    valid = _make_evaluation(
        candidate=candidates[0],
        min_distance=1.0,
        critical_time=1.0,
        objective=3.0,
        cert_status=passed_status("ok"),
        temp_root=tmp_path,
    )
    failed = _make_evaluation(
        candidate=candidates[1],
        min_distance=2.0,
        critical_time=2.0,
        objective=9.0,
        cert_status=failed_status("excluded"),
        bundle_index=1,
        temp_root=tmp_path,
    )
    nonfinite = _make_evaluation(
        candidate=candidates[2],
        min_distance=0.5,
        critical_time=0.5,
        objective=float("nan"),
        cert_status=passed_status("ok"),
        bundle_index=2,
        temp_root=tmp_path,
    )
    out_of_grid = _make_evaluation(
        candidate=candidates[3],
        min_distance=8.0,
        critical_time=8.0,
        objective=12.0,
        cert_status=passed_status("ok"),
        bundle_index=3,
        temp_root=tmp_path,
    )
    empty_qd = QDSearchResult(archive=QDArchive(grid=grid), num_evaluated=0, num_admitted=0)
    report = compare_qd_vs_single_objective(
        qd_result=empty_qd,
        single_objective_evaluations=[valid, failed, nonfinite, out_of_grid],
        budget=4,
        grid=grid,
        require_certification=True,
    )
    assert report.single_objective.filled_cells == 1
    assert report.single_objective.coverage_fraction == pytest.approx(1 / grid.cell_count)
    assert report.single_objective.qd_score == 3.0
