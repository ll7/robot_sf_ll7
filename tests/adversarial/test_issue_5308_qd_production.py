"""CPU tests for the production integration of MAP-Elites QD search (issue #5308).

These tests wire ``run_map_elites`` to the real adversarial validation /
materialization / certification / evaluation pipeline through
``production_qd_evaluator`` and ``QDSearchConfig.to_search_config``. The heavyweight
benchmark runner is replaced by an injected production evaluator so the integration
contract (bundle layout, certification gate, objective attachment, descriptor path)
is exercised on CPU without running a simulator. This is capability plumbing, not a
benchmark claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    RangeConfig,
    SearchConfig,
    SearchSpaceConfig,
)
from robot_sf.adversarial.qd import (
    QD_ARCHIVE_SCHEMA_VERSION,
    GridSpec,
    QDSearchConfig,
    production_qd_evaluator,
    run_map_elites,
    write_qd_archive,
)
from robot_sf.adversarial.search import production_candidate_evaluator


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


def _candidate(index: int) -> CandidateSpec:
    """Build one candidate spanning the search space by index."""
    return CandidateSpec(
        start=Pose2D(float(index) * 0.4, 0.0),
        goal=Pose2D(4.0 - float(index) * 0.4, 0.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=1,
    )


def _record(min_distance: float, critical_time: float, failure: str = "collision") -> dict:
    """Build a minimal episode record carrying the QD behavior descriptors."""
    if failure == "collision":
        termination = "collision"
        outcome = {"route_complete": False, "collision": True}
    elif failure == "timeout":
        termination = "timeout"
        outcome = {"route_complete": False, "timeout": True}
    else:
        termination = "incomplete"
        outcome = {"route_complete": False}
    return {
        "status": "completed",
        "termination_reason": termination,
        "outcome": outcome,
        "metrics": {
            "distance_to_human_min": min_distance,
            "time_to_collision_min": critical_time,
        },
    }


def _production_evaluator(tmp_root: Path) -> object:
    """Injected four-argument production evaluator that writes a real bundle + record.

    Mirrors the contract of ``search._default_evaluator``: write the scenario-derived
    episode record into ``candidate_dir/episode_records.jsonl`` and return a
    ``CandidateEvaluation`` carrying the path so the QD behavior descriptor can be read.
    """
    count = {"n": 0}

    def _evaluate(
        config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        index = count["n"]
        count["n"] += 1
        record = _record(
            min_distance=2.5 * ((index % 4) / 3.0),
            critical_time=3.0 * ((index % 3) / 2.0),
            failure=["collision", "timeout", "incomplete"][index % 3],
        )
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        attribution = attribution_from_episode_record(record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("production adapter fixture"),
            objective_value=None,
            failure_attribution=attribution,
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    return _evaluate


def _passing_certifier() -> object:
    """Injected certifier that always passes, so the integration test exercises the
    materialization / evaluation / archive path without a real certification package."""

    def _certify(candidate, scenario_yaml_path, require_certification):
        return passed_status("injected passing certifier")

    return _certify


def _qd_config(grid: GridSpec, *, budget: int = 12, seed: int = 7) -> QDSearchConfig:
    """Build a QD search config over the shared search space."""
    return QDSearchConfig(
        search_space=_space(),
        objective="worst_case_snqi",
        grid=grid,
        budget=budget,
        seed=seed,
        require_certification=False,
    )


def _write_template(path: Path) -> Path:
    """Write a minimal valid scenario template so materialization can run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "doorway",
                        "robot": {"start": [0.0, 0.0], "goal": [4.0, 0.0]},
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_production_evaluator_populates_archive_with_real_bundles(tmp_path: Path) -> None:
    """production_qd_evaluator runs the real pipeline and fills the QD grid."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    qd_config = _qd_config(grid)
    template = _write_template(tmp_path / "template.yaml")
    search_config = qd_config.to_search_config(
        policy="social_force",
        scenario_template=template,
        output_dir=tmp_path / "prod_out",
    )
    evaluator = production_qd_evaluator(
        search_config,
        candidate_evaluator=_production_evaluator(tmp_path),
        certifier=_passing_certifier(),
    )

    result = run_map_elites(qd_config, evaluator=evaluator)

    assert result.num_evaluated == 12
    assert result.archive.filled_cell_count() >= 2
    assert len(result.archive.distinct_failure_modes()) >= 2
    assert (search_config.output_dir / "candidate_0000").is_dir()


def test_production_evaluator_writes_archive_artifact(tmp_path: Path) -> None:
    """The production-wired QD run emits a schema-versioned coverage artifact."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    qd_config = _qd_config(grid)
    template = _write_template(tmp_path / "template.yaml")
    search_config = qd_config.to_search_config(
        policy="social_force",
        scenario_template=template,
        output_dir=tmp_path / "prod_out",
    )
    evaluator = production_qd_evaluator(
        search_config,
        candidate_evaluator=_production_evaluator(tmp_path),
        certifier=_passing_certifier(),
    )
    result = run_map_elites(qd_config, evaluator=evaluator)

    artifact = write_qd_archive(result, tmp_path / "prod_archive.json")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == QD_ARCHIVE_SCHEMA_VERSION
    assert payload["summary"]["filled_cell_count"] == result.archive.filled_cell_count()
    assert payload["summary"]["qd_score"] > 0.0


def test_production_evaluator_is_reproducible(tmp_path: Path) -> None:
    """Same config + injected production evaluator yields an identical archive."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    qd_config = _qd_config(grid, budget=11)

    def _run() -> dict:
        template = _write_template(tmp_path / "template.yaml")
        search_config = qd_config.to_search_config(
            policy="social_force",
            scenario_template=template,
            output_dir=tmp_path / f"out_{id(_run)}",
        )
        evaluator = production_qd_evaluator(
            search_config,
            candidate_evaluator=_production_evaluator(tmp_path),
            certifier=_passing_certifier(),
        )
        return run_map_elites(qd_config, evaluator=evaluator).to_json()

    assert _run() == _run()


def test_to_search_config_writes_search_space(tmp_path: Path) -> None:
    """QDSearchConfig.to_search_config materializes a loadable search-space YAML."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    qd_config = _qd_config(grid)
    template = _write_template(tmp_path / "template.yaml")
    search_config = qd_config.to_search_config(
        policy="social_force",
        scenario_template=template,
        output_dir=tmp_path / "out",
    )
    assert search_config.budget == 12
    assert search_config.objective == "worst_case_snqi"
    assert search_config.search_space_path.exists()
    reloaded = SearchSpaceConfig.from_file(search_config.search_space_path)
    assert reloaded.start_x.min == 0.0
    assert reloaded.start_x.max == 4.0


def test_production_evaluator_handles_invalid_candidate(tmp_path: Path) -> None:
    """Candidates failing search-space validation are skipped, not evaluated."""
    grid = GridSpec(x_min=0.0, x_max=2.5, y_min=0.0, y_max=3.0, bins=4)
    space = _space()
    qd_config = QDSearchConfig(
        search_space=space,
        objective="worst_case_snqi",
        grid=grid,
        budget=4,
        seed=7,
        require_certification=False,
    )
    template = _write_template(tmp_path / "template.yaml")
    search_config = qd_config.to_search_config(
        policy="social_force",
        scenario_template=template,
        output_dir=tmp_path / "out",
    )
    step = production_candidate_evaluator(
        evaluator=_production_evaluator(tmp_path),
        certifier=_passing_certifier(),
    )
    out_of_range = CandidateSpec(
        start=Pose2D(99.0, 0.0),
        goal=Pose2D(4.0, 0.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=1,
    )
    evaluation = step(search_config, out_of_range, 0)
    assert evaluation.error is not None
    assert evaluation.certification_status.status == "failed"
    assert evaluation.episode_record_path is None
