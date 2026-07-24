"""CPU tests for the bounded doorway-family MAP-Elites QD campaign (issue #5308).

Validates campaign wiring (config loading, grid/emitter building, archive schema)
with injected synthetic evaluators.  The production benchmark path requires a
CARLA-capable environment; these tests prove the campaign orchestration contract
on CPU without running the simulator.  This is capability plumbing
(design #1433), not a camera-ready benchmark claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.certification import not_available_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    SearchSpaceConfig,
)
from robot_sf.adversarial.qd import (
    QD_ARCHIVE_SCHEMA_VERSION,
    GridSpec,
    QDArchive,
    QDSearchConfig,
    run_map_elites,
    write_qd_archive,
)
from scripts.adversarial.run_doorway_qd_campaign import (
    _load_campaign_config,
    build_emitters,
    build_grid,
    build_search_space,
    run_campaign,
)

_CAMPAIGN_CONFIG = "configs/adversarial/doorway_qd_campaign_v1.yaml"


def _synthetic_evaluator():
    """Return a synthetic QDEvaluator that maps candidates to deterministic cells."""
    count = {"n": 0}

    def _evaluate(
        qd_config: QDSearchConfig, candidate: CandidateSpec
    ) -> CandidateEvaluation:
        index = count["n"]
        count["n"] += 1
        from robot_sf.adversarial.attribution import attribution_from_episode_record

        record = _synthetic_record(index)
        candidate_dir = (
            qd_config.to_search_config(
                policy="social_force",
                scenario_template=Path("/tmp/fake_template.yaml"),
                output_dir=Path("/tmp/qd_campaign_test"),
            ).output_dir
            / f"candidate_{index:04d}"
        )
        candidate_dir.mkdir(parents=True, exist_ok=True)
        episode_file = candidate_dir / "episode_records.jsonl"
        episode_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=not_available_status("synthetic"),
            objective_value=float(index % 5),
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_file,
            trajectory_csv_path=None,
            scenario_yaml_path=candidate_dir / "scenario.yaml",
            bundle_path=candidate_dir,
        )

    return _evaluate


def _synthetic_record(index: int) -> dict:
    """Build a minimal episode record with deterministic behavior descriptors."""
    distance = 2.0 * ((index % 4) / 3.0)
    critical = 4.0 * ((index % 3) / 2.0)
    failures = ["collision", "timeout", "incomplete"]
    failure = failures[index % 3]
    outcome = {"route_complete": False}
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
            "distance_to_human_min": distance,
            "time_to_collision_min": critical,
        },
    }


def _synth_certifier() -> object:
    """Return a synthetic certifier that always passes."""

    def _certify(_candidate, _scenario_yaml_path, _require_certification):
        from robot_sf.adversarial.certification import passed_status

        return passed_status("synthetic")

    return _certify


@pytest.fixture
def campaign_config() -> dict:
    """Load the campaign config once per session."""
    return _load_campaign_config(Path(_CAMPAIGN_CONFIG))


def test_campaign_config_loads(campaign_config: dict) -> None:
    """Campaign YAML must load with all required sections."""
    assert "campaign" in campaign_config
    assert "search_space" in campaign_config
    assert "production" in campaign_config
    assert campaign_config["campaign"]["objective"] == "temporal_robustness"
    assert campaign_config["campaign"]["budget"] == 256


def test_build_grid(campaign_config: dict) -> None:
    """Grid spec matches the config behavior_grid section."""
    grid = build_grid(campaign_config)
    assert isinstance(grid, GridSpec)
    assert grid.x_min == 0.0
    assert grid.x_max == 4.0
    assert grid.y_min == 0.0
    assert grid.y_max == 6.0
    assert grid.bins == 6
    assert grid.cell_count == 36


def test_build_search_space(campaign_config: dict) -> None:
    """Search space loads and validates from config mapping."""
    space = build_search_space(campaign_config)
    assert isinstance(space, SearchSpaceConfig)
    assert space.start_x.min == 0.5
    assert space.start_x.max == 1.5
    assert space.goal_x.min == 8.0
    assert space.goal_x.max == 10.0
    assert space.spawn_time_s.min == 0.0
    assert space.spawn_time_s.max == 3.0
    assert space.pedestrian_speed_mps.min == 0.6
    assert space.pedestrian_speed_mps.max == 1.6


def test_build_emitters(campaign_config: dict) -> None:
    """Emitters are constructed from the configured emitter list."""
    space = build_search_space(campaign_config)
    grid = build_grid(campaign_config)
    from robot_sf.adversarial.qd import QDArchive

    archive = QDArchive(grid=grid)
    emitters = build_emitters(space, archive, cfg=campaign_config, seed=42)
    assert len(emitters) == 3
    names = [type(e).__name__ for e in emitters]
    assert "RandomCandidateSampler" in names
    assert "CoordinateRefinementSampler" in names
    assert "CMaMeEmitter" in names


def test_build_emitters_with_warm_starts(campaign_config: dict) -> None:
    """Emitters accept warm-start candidates without error."""
    space = build_search_space(campaign_config)
    grid = build_grid(campaign_config)
    from robot_sf.adversarial.config import WarmStartCandidate
    from robot_sf.adversarial.qd import QDArchive

    archive = QDArchive(grid=grid)
    warm = (
        WarmStartCandidate(
            candidate=CandidateSpec(
                start=Pose2D(1.0, 5.0),
                goal=Pose2D(9.0, 5.0),
                spawn_time_s=0.5,
                pedestrian_speed_mps=1.2,
                pedestrian_delay_s=0.2,
                scenario_seed=500,
            ),
            scenario="doorway",
            planner="social_force",
            outcome_margin=0.1,
        ),
    )
    emitters = build_emitters(
        space, archive, cfg=campaign_config, seed=42, warm_starts=warm
    )
    assert len(emitters) == 3


def test_campaign_synthetic_wiring_produces_valid_archive(
    campaign_config: dict, tmp_path: Path
) -> None:
    """Campaign with synthetic evaluator produces a schema-valid archive."""

    from robot_sf.adversarial.qd import QDArchive, run_map_elites

    space = build_search_space(campaign_config)
    grid = build_grid(campaign_config)
    archive = QDArchive(grid=grid)

    emitters = build_emitters(space, archive, cfg=campaign_config, seed=42)

    qd_config = QDSearchConfig(
        search_space=space,
        objective="worst_case_snqi",
        grid=grid,
        budget=8,
        seed=42,
    )

    result = run_map_elites(
        qd_config,
        evaluator=_synthetic_evaluator(),
        emitters=emitters,
        archive=archive,
    )

    archive_file = tmp_path / "archive.json"
    write_qd_archive(result, archive_file)
    payload = json.loads(archive_file.read_text(encoding="utf-8"))

    assert payload["schema_version"] == QD_ARCHIVE_SCHEMA_VERSION
    assert payload["behavior_axes"] == ["distance_to_human_min", "time_to_collision_min"]
    assert payload["summary"]["filled_cell_count"] == result.archive.filled_cell_count()
    assert payload["summary"]["filled_cell_count"] > 0
    assert payload["summary"]["qd_score"] > 0.0


def test_campaign_synthetic_full_pipeline(tmp_path: Path) -> None:
    """Full campaign pipeline with synthetic evaluator produces all artifacts."""
    config_path = Path(_CAMPAIGN_CONFIG)
    output_dir = tmp_path / "full_campaign"

    from robot_sf.adversarial.qd import run_map_elites, write_qd_archive

    cfg = _load_campaign_config(config_path)
    space = build_search_space(cfg)
    grid = build_grid(cfg)
    archive = QDArchive(
        grid=grid,
        require_certification=cfg["campaign"].get("require_certification", False),
    )

    emitters = build_emitters(space, archive, cfg=cfg, seed=42)

    qd_config = QDSearchConfig(
        search_space=space,
        objective="worst_case_snqi",
        grid=grid,
        budget=12,
        seed=42,
    )

    qd_result = run_map_elites(
        qd_config,
        evaluator=_synthetic_evaluator(),
        emitters=emitters,
        archive=archive,
    )

    archive_path = output_dir / "qd_archive.json"
    write_qd_archive(qd_result, archive_path)

    assert archive_path.exists()
    assert qd_result.archive.filled_cell_count() >= 2
    assert len(qd_result.archive.distinct_failure_modes()) >= 2
    assert qd_result.num_evaluated == 12
    assert qd_result.num_admitted >= qd_result.archive.filled_cell_count()


def test_campaign_synthetic_comparison(tmp_path: Path) -> None:
    """QD vs single-objective comparison produces schema-valid report."""
    cfg = _load_campaign_config(Path(_CAMPAIGN_CONFIG))
    space = build_search_space(cfg)
    grid = build_grid(cfg)
    archive = QDArchive(grid=grid)
    emitters = build_emitters(space, archive, cfg=cfg, seed=42)

    from robot_sf.adversarial.qd import compare_qd_vs_single_objective

    qd_config = QDSearchConfig(
        search_space=space,
        objective="worst_case_snqi",
        grid=grid,
        budget=8,
        seed=42,
    )

    qd_result = run_map_elites(
        qd_config,
        evaluator=_synthetic_evaluator(),
        emitters=emitters,
        archive=archive,
    )

    from robot_sf.adversarial.attribution import attribution_from_episode_record
    from robot_sf.adversarial.certification import passed_status

    so_evaluations = []
    for i in range(8):
        record = _synthetic_record(i)
        episode_path = tmp_path / f"so_episode_{i}.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        bundle = tmp_path / f"so_bundle_{i}"
        bundle.mkdir(exist_ok=True)
        so_evaluations.append(
            CandidateEvaluation(
                candidate=CandidateSpec(
                    start=Pose2D(1.0, 5.0),
                    goal=Pose2D(9.0, 5.0),
                    spawn_time_s=0.0,
                    pedestrian_speed_mps=1.0,
                    pedestrian_delay_s=0.0,
                    scenario_seed=100 + i,
                ),
                certification_status=passed_status("synthetic"),
                objective_value=float(i),
                failure_attribution=attribution_from_episode_record(record),
                episode_record_path=episode_path,
                trajectory_csv_path=None,
                scenario_yaml_path=bundle / "scenario.yaml",
                bundle_path=bundle,
            )
        )

    comparison = compare_qd_vs_single_objective(
        qd_result=qd_result,
        single_objective_evaluations=so_evaluations,
        budget=8,
        grid=grid,
    )

    assert comparison.qd.method == "map_elites"
    assert comparison.single_objective.method == "single_objective"
    assert comparison.qd.filled_cells > 0
    assert comparison.single_objective.filled_cells >= 0
    assert comparison.qd.distinct_failure_modes >= comparison.single_objective.distinct_failure_modes

    payload = comparison.to_json()
    assert payload["schema_version"] == QD_ARCHIVE_SCHEMA_VERSION
    assert payload["comparison_type"] == "equal_budget_qd_vs_single_objective"


def test_campaign_emitter_rejects_unknown_name(campaign_config: dict) -> None:
    """Unknown emitter names must fail closed."""
    from robot_sf.adversarial.qd import QDArchive

    bad_cfg = dict(campaign_config)
    bad_cfg["campaign"] = dict(bad_cfg["campaign"])
    bad_cfg["campaign"]["emitters"] = ["bogus_emitter"]
    with pytest.raises(ValueError, match="Unknown emitter"):
        build_emitters(
            build_search_space(bad_cfg),
            QDArchive(grid=build_grid(bad_cfg)),
            cfg=bad_cfg,
            seed=0,
        )


def test_campaign_config_missing_sections(tmp_path: Path) -> None:
    """Campaign config validation rejects missing required sections."""
    bad = {"schema_version": "adversarial-qd-campaign.v1", "campaign": {}}
    bad_path = tmp_path / "bad_config.yaml"
    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        _load_campaign_config(bad_path)


def test_campaign_config_rejects_wrong_schema(tmp_path: Path) -> None:
    """Campaign config validation rejects unsupported schema versions."""
    bad = {"schema_version": "wrong-schema", "campaign": {}, "search_space": {}, "production": {}}
    bad_path = tmp_path / "bad_schema.yaml"
    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported campaign schema"):
        _load_campaign_config(bad_path)


def test_run_campaign_returns_summary_structure(tmp_path: Path) -> None:
    """run_campaign returns a dict with all required summary keys."""
    config_path = Path(_CAMPAIGN_CONFIG)
    output_dir = tmp_path / "run_summary_test"

    from scripts.adversarial.run_doorway_qd_campaign import _resolve_artifact_path

    out = _resolve_artifact_path(output_dir)
    summary = run_campaign(
        config_path=config_path,
        output_dir=out,
        budget=4,
        seed=7,
    )

    assert isinstance(summary, dict)
    assert summary["schema_version"] == "adversarial_qd_campaign_result.v1"
    assert "archive_path" in summary
    assert "filled_cell_count" in summary
    assert "distinct_failure_modes" in summary
    assert "qd_score" in summary
    assert "coverage_fraction" in summary
    assert "num_evaluated" in summary
    assert "num_admitted" in summary
    assert "budget" in summary
    assert summary["budget"] == 4
