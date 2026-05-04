from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import certification, objectives, search
from robot_sf.adversarial.attribution import attribution_from_episode_record, attribution_from_error
from robot_sf.adversarial.bundle import write_trajectory_csv
from robot_sf.adversarial.certification import failed_status, not_available_status, passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    MultiPedAdversarialConfig,
    MultiPedCandidateSpec,
    Pose2D,
    SearchConfig,
    SearchSpaceConfig,
)
from robot_sf.adversarial.io import read_first_jsonl_record
from robot_sf.adversarial.materialize import (
    materialize_multi_ped_scenario_payload,
    materialize_multi_ped_single_pedestrian_overrides,
)
from robot_sf.adversarial.samplers import CoordinateRefinementSampler, RandomCandidateSampler


def _write_template(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "template",
                        "map_id": "classic_cross_trap",
                        "simulation_config": {"max_episode_steps": 30, "ped_density": 0.0},
                        "robot_config": {},
                        "metadata": {"archetype": "test"},
                        "seeds": [1],
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_space(path: Path, *, min_distance: float = 0.5) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 1.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": min_distance},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _config(
    tmp_path: Path,
    *,
    require_certification: bool = False,
    workers: int = 1,
) -> SearchConfig:
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    _write_space(search_space)
    return SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        budget=2,
        seed=123,
        workers=workers,
        require_certification=require_certification,
    )


class _SequenceSampler:
    def __init__(self, candidates: list[CandidateSpec]) -> None:
        self._candidates = list(candidates)

    def sample(self) -> CandidateSpec:
        return self._candidates.pop(0)


def _candidate(seed: int, *, goal_x: float = 5.0) -> CandidateSpec:
    return CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(goal_x, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=seed,
    )


def test_search_config_from_files_validates_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path)

    candidate = config.search_space.sample_candidate(__import__("random").Random(1))

    assert config.search_space.validate_candidate(candidate) == []
    assert candidate.scenario_seed == 7


def test_search_space_validates_all_configured_candidate_ranges(tmp_path: Path) -> None:
    config = _config(tmp_path)

    errors = config.search_space.validate_candidate(
        CandidateSpec(
            start=Pose2D(1.0, 2.0),
            goal=Pose2D(5.0, 2.0),
            spawn_time_s=1.0,
            pedestrian_speed_mps=2.0,
            pedestrian_delay_s=1.0,
            scenario_seed=8,
        )
    )

    assert "spawn_time_s outside search space" in errors
    assert "pedestrian_speed_mps outside search space" in errors
    assert "pedestrian_delay_s outside search space" in errors
    assert "scenario_seed outside search space" in errors


def test_search_space_rejects_invalid_min_start_goal_distance() -> None:
    payload = {
        "variables": {
            "start_x": {"min": 0, "max": 1},
            "start_y": {"min": 0, "max": 1},
            "goal_x": {"min": 2, "max": 3},
            "goal_y": {"min": 2, "max": 3},
        },
        "constraints": {"min_start_goal_distance_m": -1.0},
    }

    with pytest.raises(ValueError, match="min_start_goal_distance_m"):
        SearchSpaceConfig.from_mapping(payload)


def test_search_space_rejects_non_integral_seed_bounds() -> None:
    """Scenario seed ranges must describe a discrete integer sampling space."""
    payload = {
        "variables": {
            "start_x": {"min": 0, "max": 1},
            "start_y": {"min": 0, "max": 1},
            "goal_x": {"min": 2, "max": 3},
            "goal_y": {"min": 2, "max": 3},
            "scenario_seed": {"min": 1.5, "max": 2.5},
        }
    }

    with pytest.raises(ValueError, match="scenario_seed bounds must be integers"):
        SearchSpaceConfig.from_mapping(payload)


def test_multi_ped_adversarial_config_parses_and_serializes_yaml(tmp_path: Path) -> None:
    """Multi-ped adversarial candidates should have a deterministic schema contract."""
    path = tmp_path / "multi_ped.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "adversarial-multi-ped.v1",
                "family": "group_squeeze",
                "scenario_seed": 41,
                "constraints": {"min_start_goal_distance_m": 1.0},
                "pedestrians": [
                    {
                        "id": "left_blocker",
                        "start": {"x": 1.0, "y": 2.0},
                        "goal": {"x": 5.0, "y": 2.0},
                        "spawn_time_s": 0.5,
                        "speed_mps": 1.1,
                        "delay_s": 0.25,
                    },
                    {
                        "id": "right_blocker",
                        "start": {"x": 1.0, "y": 3.0},
                        "goal": {"x": 5.0, "y": 3.0},
                        "spawn_time_s": 0.75,
                        "speed_mps": 1.2,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config = MultiPedAdversarialConfig.from_file(path)

    assert config.schema_version == "adversarial-multi-ped.v1"
    assert config.family == "group_squeeze"
    assert config.scenario_seed == 41
    assert config.validate() == []
    assert [ped.id for ped in config.pedestrians] == ["left_blocker", "right_blocker"]
    assert config.to_json() == {
        "schema_version": "adversarial-multi-ped.v1",
        "family": "group_squeeze",
        "scenario_seed": 41,
        "constraints": {"min_start_goal_distance_m": 1.0},
        "pedestrians": [
            {
                "id": "left_blocker",
                "start": {"x": 1.0, "y": 2.0, "theta": 0.0},
                "goal": {"x": 5.0, "y": 2.0, "theta": 0.0},
                "spawn_time_s": 0.5,
                "speed_mps": 1.1,
                "delay_s": 0.25,
                "metadata": {},
            },
            {
                "id": "right_blocker",
                "start": {"x": 1.0, "y": 3.0, "theta": 0.0},
                "goal": {"x": 5.0, "y": 3.0, "theta": 0.0},
                "spawn_time_s": 0.75,
                "speed_mps": 1.2,
                "delay_s": 0.0,
                "metadata": {},
            },
        ],
    }


def test_multi_ped_adversarial_config_reports_validation_errors() -> None:
    """Invalid multi-ped contracts should fail before runtime integration."""
    config = MultiPedAdversarialConfig(
        family="group_squeeze",
        scenario_seed=-1,
        min_start_goal_distance_m=2.0,
        pedestrians=[
            MultiPedCandidateSpec(
                id="blocker",
                start=Pose2D(1.0, 2.0),
                goal=Pose2D(1.5, 2.0),
                spawn_time_s=-0.1,
                speed_mps=0.0,
            ),
            MultiPedCandidateSpec(
                id="blocker",
                start=Pose2D(float("nan"), 3.0),
                goal=Pose2D(5.0, 3.0),
                spawn_time_s=0.0,
                speed_mps=1.0,
                delay_s=-0.5,
            ),
        ],
    )

    errors = config.validate()

    assert "scenario_seed must be non-negative" in errors
    assert "pedestrians ids must be unique" in errors
    assert "pedestrians[0].spawn_time_s must be non-negative" in errors
    assert "pedestrians[0].speed_mps must be positive" in errors
    assert "pedestrians[0] start and goal are closer than min_start_goal_distance_m" in errors
    assert "pedestrians[1].start.x must be finite" in errors
    assert "pedestrians[1].delay_s must be non-negative" in errors


def test_multi_ped_adversarial_example_config_is_valid() -> None:
    """The checked-in example should stay aligned with the schema parser."""
    config = MultiPedAdversarialConfig.from_file(
        Path("configs/adversarial/group_squeeze_multi_ped_example.yaml")
    )

    assert config.family == "group_squeeze"
    assert len(config.pedestrians) == 2
    assert config.validate() == []


def test_multi_ped_config_materializes_single_pedestrian_overrides() -> None:
    """Multi-ped adversarial configs should bridge to scenario-loader override entries."""
    config = MultiPedAdversarialConfig(
        family="group_squeeze",
        scenario_seed=41,
        pedestrians=[
            MultiPedCandidateSpec(
                id="left_blocker",
                start=Pose2D(1.0, 2.0, 0.1),
                goal=Pose2D(5.0, 2.0),
                spawn_time_s=0.5,
                speed_mps=1.1,
                delay_s=0.25,
                metadata={"role": "left"},
            ),
            MultiPedCandidateSpec(
                id="right_blocker",
                start=Pose2D(1.0, 3.0),
                goal=Pose2D(5.0, 3.0),
                spawn_time_s=0.75,
                speed_mps=1.2,
            ),
        ],
    )

    overrides = materialize_multi_ped_single_pedestrian_overrides(config)

    assert overrides == [
        {
            "id": "left_blocker",
            "start": [1.0, 2.0],
            "goal": [5.0, 2.0],
            "speed_m_s": 1.1,
            "start_delay_s": 0.75,
            "note": "adversarial-multi-ped.v1 group_squeeze seed=41 ped=left_blocker",
            "metadata": {
                "adversarial_family": "group_squeeze",
                "adversarial_schema_version": "adversarial-multi-ped.v1",
                "adversarial_scenario_seed": 41,
                "pedestrian_metadata": {"role": "left"},
                "spawn_time_s": 0.5,
                "delay_s": 0.25,
            },
        },
        {
            "id": "right_blocker",
            "start": [1.0, 3.0],
            "goal": [5.0, 3.0],
            "speed_m_s": 1.2,
            "start_delay_s": 0.75,
            "note": "adversarial-multi-ped.v1 group_squeeze seed=41 ped=right_blocker",
            "metadata": {
                "adversarial_family": "group_squeeze",
                "adversarial_schema_version": "adversarial-multi-ped.v1",
                "adversarial_scenario_seed": 41,
                "pedestrian_metadata": {},
                "spawn_time_s": 0.75,
                "delay_s": 0.0,
            },
        },
    ]


def test_multi_ped_materialized_overrides_are_yaml_safe() -> None:
    """Materialized overrides should serialize without custom YAML representers."""
    config = MultiPedAdversarialConfig(
        family="late_stop",
        scenario_seed=7,
        pedestrians=[
            MultiPedCandidateSpec(
                id="stopper",
                start=Pose2D(0.0, 0.0),
                goal=Pose2D(1.0, 0.0),
                speed_mps=0.8,
            )
        ],
    )

    dumped = yaml.safe_dump(
        {"single_pedestrians": materialize_multi_ped_single_pedestrian_overrides(config)},
        sort_keys=False,
    )

    loaded = yaml.safe_load(dumped)
    assert loaded["single_pedestrians"][0]["id"] == "stopper"
    assert loaded["single_pedestrians"][0]["start_delay_s"] == 0.0


def test_multi_ped_config_materializes_scenario_payload_with_template_merge() -> None:
    """Multi-ped configs should produce scenario-loader-ready manifest payloads."""
    config = MultiPedAdversarialConfig(
        family="group_squeeze",
        scenario_seed=41,
        pedestrians=[
            MultiPedCandidateSpec(
                id="left_blocker",
                start=Pose2D(1.0, 2.0),
                goal=Pose2D(5.0, 2.0),
                spawn_time_s=0.5,
                speed_mps=1.1,
            ),
            MultiPedCandidateSpec(
                id="right_blocker",
                start=Pose2D(1.0, 3.0),
                goal=Pose2D(5.0, 3.0),
                spawn_time_s=0.75,
                speed_mps=1.2,
            ),
        ],
    )
    template = {
        "scenarios": [
            {
                "name": "template",
                "map_id": "classic_cross_trap",
                "simulation_config": {"max_episode_steps": 30, "ped_density": 0.0},
                "metadata": {"archetype": "test"},
                "single_pedestrians": [
                    {"id": "left_blocker", "role": "wait", "note": "template note"}
                ],
            }
        ]
    }

    payload = materialize_multi_ped_scenario_payload(config, template)

    scenario = payload["scenarios"][0]
    assert scenario["name"] == "template_multi_ped_adversarial_0041"
    assert scenario["seeds"] == [41]
    assert scenario["simulation_config"]["route_spawn_seed"] == 41
    assert scenario["metadata"]["archetype"] == "test"
    assert scenario["metadata"]["adversarial_multi_ped"]["family"] == "group_squeeze"
    assert scenario["single_pedestrians"][0]["id"] == "left_blocker"
    assert scenario["single_pedestrians"][0]["role"] == "wait"
    assert scenario["single_pedestrians"][0]["start"] == [1.0, 2.0]
    assert scenario["single_pedestrians"][0]["note"].startswith("adversarial-multi-ped.v1")
    assert scenario["single_pedestrians"][1]["id"] == "right_blocker"


def test_multi_ped_materialized_scenario_payload_is_yaml_safe() -> None:
    """Scenario payload helper should emit plain YAML-safe primitives."""
    config = MultiPedAdversarialConfig(
        family="late_stop",
        scenario_seed=7,
        pedestrians=[
            MultiPedCandidateSpec(
                id="stopper",
                start=Pose2D(0.0, 0.0),
                goal=Pose2D(1.0, 0.0),
                speed_mps=0.8,
            )
        ],
    )

    dumped = yaml.safe_dump(
        materialize_multi_ped_scenario_payload(
            config,
            {"scenarios": [{"name": "template", "map_id": "classic_cross_trap"}]},
        ),
        sort_keys=False,
    )

    loaded = yaml.safe_load(dumped)
    assert loaded["scenarios"][0]["single_pedestrians"][0]["id"] == "stopper"
    assert loaded["scenarios"][0]["metadata"]["adversarial_multi_ped"]["schema_version"] == (
        "adversarial-multi-ped.v1"
    )


def test_programmatic_search_scores_candidates_without_subprocess(tmp_path: Path) -> None:
    config = _config(tmp_path)
    scores = [0.8, 0.2]

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        snqi = scores.pop(0)
        record: dict[str, Any] = {
            "episode_id": f"episode-{candidate.scenario_seed}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": snqi, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=trajectory_path,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=_SequenceSampler([_candidate(7), _candidate(7)]),
    )

    assert result.num_candidates == 2
    assert result.num_invalid_candidates == 0
    assert result.best_objective_value == -0.2
    assert result.best_bundle_path == config.output_dir / "candidate_0001"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["best_bundle_path"].endswith("candidate_0001")
    assert (config.output_dir / "candidate_0001" / "scenario.yaml").exists()
    assert (config.output_dir / "candidate_0001" / "route_overrides.yaml").exists()


def test_coordinate_refinement_sampler_improves_synthetic_objective(tmp_path: Path) -> None:
    """Feedback-capable optimizer samplers should run through the existing search API."""
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    search_space.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 0.0, "max": 2.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 5.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": 0.5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        budget=3,
        seed=123,
    )
    sampler = CoordinateRefinementSampler(config.search_space, seed=123)
    sampled: list[CandidateSpec] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        sampled.append(candidate)
        record: dict[str, Any] = {
            "episode_id": f"candidate-{len(sampled)}",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 2.0 - candidate.start.x, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=trajectory_path,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=sampler,
    )

    assert [candidate.start.x for candidate in sampled[:2]] == [1.0, 2.0]
    assert result.best_bundle_path == config.output_dir / "candidate_0001"
    assert result.best_objective_value == pytest.approx(-0.0)


def test_invalid_optimizer_proposals_are_rejected_before_evaluation(tmp_path: Path) -> None:
    """Search-space validation must fail closed before benchmark evaluation."""
    config = _config(tmp_path)

    class InvalidThenValidSampler:
        def __init__(self) -> None:
            self._candidates = [
                CandidateSpec(
                    start=Pose2D(-10.0, 2.0),
                    goal=Pose2D(5.0, 2.0),
                    spawn_time_s=0.0,
                    pedestrian_speed_mps=1.0,
                    pedestrian_delay_s=0.0,
                    scenario_seed=7,
                ),
                _candidate(7),
            ]
            self.observed: list[CandidateEvaluation] = []

        def sample(self) -> CandidateSpec:
            return self._candidates.pop(0)

        def observe(self, evaluation: CandidateEvaluation) -> None:
            self.observed.append(evaluation)

    sampler = InvalidThenValidSampler()
    evaluated: list[CandidateSpec] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        evaluated.append(candidate)
        record: dict[str, Any] = {
            "episode_id": "valid",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 0.5, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=sampler,
    )

    assert evaluated == [_candidate(7)]
    assert result.num_invalid_candidates == 1
    assert len(sampler.observed) == 2
    assert sampler.observed[0].objective_value is None
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["error"] == "start.x outside search space"


def test_default_search_keeps_candidate_evaluation_sequential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Candidate search order is deterministic; workers are scoped to each benchmark call."""
    config = _config(tmp_path, workers=4)
    call_order: list[tuple[Path, int]] = []

    def fake_run_batch(
        _scenario_yaml_path: Path,
        *,
        out_path: Path,
        workers: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        call_order.append((out_path.parent, workers))
        record: dict[str, Any] = {
            "episode_id": out_path.parent.name,
            "seed": len(call_order),
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 0.5, "success": 1.0},
        }
        out_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return {"failures": []}

    monkeypatch.setattr(search, "run_batch", fake_run_batch)

    result = search.run_adversarial_search(
        config,
        certifier=lambda _candidate, _path, _required: passed_status("test certifier"),
        sampler=_SequenceSampler([_candidate(7), _candidate(7)]),
    )

    assert result.num_candidates == 2
    assert call_order == [
        (config.output_dir / "candidate_0000", 4),
        (config.output_dir / "candidate_0001", 4),
    ]


def test_required_certification_fails_closed_when_adapter_missing(tmp_path: Path) -> None:
    config = _config(tmp_path, require_certification=True)
    config = SearchConfig.from_files(
        policy=config.policy,
        scenario_template=config.scenario_template,
        search_space=config.search_space_path,
        objective=config.objective,
        output_dir=config.output_dir,
        budget=1,
        seed=config.seed,
        require_certification=True,
    )

    def evaluator(*_args: object, **_kwargs: object) -> CandidateEvaluation:
        raise AssertionError("strict certification should reject before evaluation")

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        certifier=lambda _candidate, _path, _required: not_available_status(
            "scenario_cert.v1 adapter is not available"
        ),
        sampler=_SequenceSampler([_candidate(7)]),
    )

    assert result.best_candidate is None
    assert result.num_invalid_candidates == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["certification_status"]["status"] == "not_available"
    assert manifest["candidates"][0]["error"] == "scenario_cert.v1 adapter is not available"


def test_required_certification_uses_real_scenario_certification_api(tmp_path: Path) -> None:
    """Strict adversarial search should exclude invalid candidates and evaluate certified ones."""
    template = tmp_path / "template.yaml"
    search_space = tmp_path / "space.yaml"
    _write_template(template)
    search_space.write_text(
        yaml.safe_dump(
            {
                "variables": {
                    "start_x": {"min": 1.0, "max": 2.0},
                    "start_y": {"min": 2.0, "max": 2.0},
                    "goal_x": {"min": 4.0, "max": 5.0},
                    "goal_y": {"min": 2.0, "max": 2.0},
                    "spawn_time_s": {"min": 0.0, "max": 0.0},
                    "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                    "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                    "scenario_seed": {"min": 7, "max": 7},
                },
                "constraints": {"min_start_goal_distance_m": 0.5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    config = SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        budget=2,
        seed=123,
        require_certification=True,
    )
    invalid_candidate = _candidate(7)
    valid_candidate = CandidateSpec(
        start=Pose2D(2.0, 2.0),
        goal=Pose2D(4.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )
    evaluated: list[CandidateSpec] = []

    def evaluator(
        _config: SearchConfig,
        candidate: CandidateSpec,
        scenario_yaml_path: Path,
        candidate_dir: Path,
    ) -> CandidateEvaluation:
        evaluated.append(candidate)
        record: dict[str, Any] = {
            "episode_id": "strict-cert-valid",
            "seed": candidate.scenario_seed,
            "status": "success",
            "steps": 3,
            "termination_reason": "success",
            "outcome": {"route_complete": True, "collision": False, "timeout": False},
            "metrics": {"snqi": 0.5, "success": 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        trajectory_path = write_trajectory_csv(candidate_dir / "trajectory.csv", record)
        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status(),
            objective_value=None,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=trajectory_path,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    result = search.run_adversarial_search(
        config,
        evaluator=evaluator,
        sampler=_SequenceSampler([invalid_candidate, valid_candidate]),
    )

    assert evaluated == [valid_candidate]
    assert result.num_candidates == 2
    assert result.num_invalid_candidates == 1
    assert result.num_valid_candidates == 1
    assert result.best_bundle_path == config.output_dir / "candidate_0001"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    invalid_status = manifest["candidates"][0]["certification_status"]
    valid_status = manifest["candidates"][1]["certification_status"]
    assert invalid_status["status"] == "failed"
    assert "start_inside_static_obstacle" in invalid_status["reason"]
    assert manifest["candidates"][0]["error"] == "start_inside_static_obstacle"
    assert valid_status["status"] == "passed"
    assert valid_status["details"]["certificates"][0]["benchmark_eligibility"] != "excluded"
    assert manifest["candidates"][1]["trajectory_csv_path"].endswith(
        "candidate_0001/trajectory.csv"
    )


def test_default_evaluator_treats_failures_as_failed_jobs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Benchmark failure summaries must fail closed before objective scoring."""
    config = _config(tmp_path)

    def fake_run_batch(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"failures": [{"scenario_id": "candidate", "error": "boom"}]}

    monkeypatch.setattr(search, "run_batch", fake_run_batch)

    with pytest.raises(RuntimeError, match="candidate evaluation failed"):
        search._default_evaluator(
            config,
            _candidate(1),
            tmp_path / "scenario.yaml",
            tmp_path / "candidate",
        )


def test_failure_attribution_covers_primary_outcomes() -> None:
    """Failure attribution must distinguish collision, timeout, incomplete, and errors."""
    collision = attribution_from_episode_record(
        {"status": "done", "outcome": {"collision": True, "route_complete": False}}
    )
    legacy_collision = attribution_from_episode_record(
        {"status": "done", "outcome": {"collision_event": True, "route_complete": False}}
    )
    timeout = attribution_from_episode_record(
        {"status": "done", "outcome": {"timeout_event": True, "route_complete": False}}
    )
    incomplete = attribution_from_episode_record(
        {"status": "done", "outcome": {"route_complete": False}}
    )
    error = attribution_from_error("boom")

    assert collision.primary_failure == "collision"
    assert legacy_collision.primary_failure == "collision"
    assert timeout.primary_failure == "timeout"
    assert incomplete.primary_failure == "incomplete"
    assert error.to_json()["status"] == "evaluation_failed"


def test_read_first_jsonl_record_skips_malformed_lines(tmp_path: Path) -> None:
    """JSONL helper should fail soft for malformed records and keep scanning."""
    path = tmp_path / "episode.jsonl"
    path.write_text("{bad json}\n\n" + json.dumps({"episode_id": "ok"}) + "\n", encoding="utf-8")

    assert read_first_jsonl_record(path) == {"episode_id": "ok"}


def test_write_trajectory_csv_escapes_fields(tmp_path: Path) -> None:
    """Trajectory index rows should remain valid CSV when fields contain punctuation."""
    path = write_trajectory_csv(
        tmp_path / "trajectory.csv",
        {
            "episode_id": "episode,1",
            "seed": 7,
            "status": "done",
            "steps": 3,
            "termination_reason": 'quote "and" comma, here',
        },
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[1] == ["episode,1", "7", "done", "3", 'quote "and" comma, here']


def test_write_trajectory_csv_exports_dense_trajectory_data(tmp_path: Path) -> None:
    """Trajectory data in episode records should become per-entity CSV rows."""
    path = write_trajectory_csv(
        tmp_path / "trajectory.csv",
        {
            "episode_id": "episode-1",
            "seed": 7,
            "trajectory_data": [
                {
                    "step": 0,
                    "time_s": 0.0,
                    "robot": {"x": 1.0, "y": 2.0, "theta": 0.1},
                    "pedestrians": {"ped-1": [3.0, 4.0, 0.2]},
                },
                {
                    "step": 1,
                    "time_s": 0.1,
                    "robot_position": [1.5, 2.5, 0.15],
                    "pedestrian_positions": [[3.5, 4.5, 0.25]],
                },
            ],
        },
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "episode_id": "episode-1",
            "seed": "7",
            "step": "0",
            "entity_type": "robot",
            "entity_id": "robot",
            "time_s": "0.0",
            "x": "1.0",
            "y": "2.0",
            "theta": "0.1",
        },
        {
            "episode_id": "episode-1",
            "seed": "7",
            "step": "0",
            "entity_type": "pedestrian",
            "entity_id": "ped-1",
            "time_s": "0.0",
            "x": "3.0",
            "y": "4.0",
            "theta": "0.2",
        },
        {
            "episode_id": "episode-1",
            "seed": "7",
            "step": "1",
            "entity_type": "robot",
            "entity_id": "robot",
            "time_s": "0.1",
            "x": "1.5",
            "y": "2.5",
            "theta": "0.15",
        },
        {
            "episode_id": "episode-1",
            "seed": "7",
            "step": "1",
            "entity_type": "pedestrian",
            "entity_id": "0",
            "time_s": "0.1",
            "x": "3.5",
            "y": "4.5",
            "theta": "0.25",
        },
    ]


def test_write_trajectory_csv_supports_legacy_coordinate_lists(tmp_path: Path) -> None:
    """Existing visualization-style coordinate lists should produce dense robot rows."""
    path = write_trajectory_csv(
        tmp_path / "trajectory.csv",
        {"episode_id": "episode-2", "seed": 11, "trajectory_data": [[0.0, 1.0], [0.5, 1.5]]},
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["step"] for row in rows] == ["0", "1"]
    assert [row["entity_type"] for row in rows] == ["robot", "robot"]
    assert [(row["x"], row["y"]) for row in rows] == [("0.0", "1.0"), ("0.5", "1.5")]


def test_certification_adapter_handles_missing_and_mocked_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Certification must fail closed when required and normalize adapter payloads."""
    monkeypatch.setitem(sys.modules, "robot_sf.scenario_certification", None)

    advisory = certification.certify_candidate(
        _candidate(1), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=False
    )
    assert advisory.passed
    assert failed_status("bad", details={"why": "test"}).to_json()["details"] == {"why": "test"}

    fake_module = types.ModuleType("robot_sf.scenario_certification")
    responses: list[object] = [
        {"status": "valid", "reason": "ok", "details": ["raw"]},
        {"status": "unavailable", "reason": "backend absent"},
        "not a mapping",
        {"status": "invalid", "reason": "outside map", "details": {"field": "start"}},
    ]

    def fake_certify_scenario(*_args: object, **_kwargs: object) -> object:
        return responses.pop(0)

    fake_module.certify_scenario = fake_certify_scenario
    monkeypatch.setitem(sys.modules, "robot_sf.scenario_certification", fake_module)

    passed = certification.certify_candidate(
        _candidate(2), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )
    unavailable = certification.certify_candidate(
        _candidate(3), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )
    non_mapping = certification.certify_candidate(
        _candidate(4), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )
    failed = certification.certify_candidate(
        _candidate(5), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )

    assert passed.passed
    assert passed.details == {"raw_details": ["raw"]}
    assert unavailable.status == "not_available"
    assert non_mapping.status == "failed"
    assert failed.reason == "outside map"


def test_certification_adapter_uses_current_scenario_certification_file_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Adversarial certification should call the in-repo scenario-cert file API."""
    fake_module = types.ModuleType("robot_sf.scenario_certification")

    def fake_certify_scenario_file(*_args: object, **_kwargs: object) -> list[object]:
        return [object()]

    def fake_certificate_to_dict(_certificate: object) -> dict[str, object]:
        return {
            "classification": "valid",
            "benchmark_eligibility": "eligible",
            "reasons": ["ok"],
        }

    fake_module.certify_scenario_file = fake_certify_scenario_file
    fake_module.certificate_to_dict = fake_certificate_to_dict
    monkeypatch.setitem(sys.modules, "robot_sf.scenario_certification", fake_module)

    status = certification.certify_candidate(
        _candidate(6), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )

    assert status.passed
    assert status.reason == "ok"
    assert status.details["certificates"][0]["classification"] == "valid"


def test_certification_adapter_preserves_worst_file_api_eligibility(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """File-API normalization should keep stress-only/excluded evidence visible."""
    fake_module = types.ModuleType("robot_sf.scenario_certification")

    def fake_certify_scenario_file(*_args: object, **_kwargs: object) -> list[object]:
        return [object(), object(), object()]

    payloads: list[dict[str, object]] = [
        {
            "classification": "valid",
            "benchmark_eligibility": None,
            "reasons": ["valid fallback"],
        },
        {
            "classification": "knife_edge",
            "benchmark_eligibility": "stress_only",
            "reasons": ["knife-edge clearance"],
        },
        {
            "classification": "valid",
            "benchmark_eligibility": "eligible",
            "reasons": ["eligible route"],
        },
    ]

    def fake_certificate_to_dict(_certificate: object) -> dict[str, object]:
        return payloads.pop(0)

    fake_module.certify_scenario_file = fake_certify_scenario_file
    fake_module.certificate_to_dict = fake_certificate_to_dict
    monkeypatch.setitem(sys.modules, "robot_sf.scenario_certification", fake_module)

    status = certification.certify_candidate(
        _candidate(7), scenario_yaml_path=tmp_path / "scenario.yaml", require_certification=True
    )

    assert status.passed
    assert status.reason == "knife-edge clearance"


def test_objective_registry_and_fallback_scoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Objectives should score SNQI records, fallback failures, and registry errors."""
    monkeypatch.setattr(objectives, "_OBJECTIVES", dict(objectives._OBJECTIVES))
    episode_path = tmp_path / "episode.jsonl"
    episode_path.write_text(
        json.dumps({"metrics": {"snqi": "nan"}, "outcome": {"route_complete": True}}) + "\n",
        encoding="utf-8",
    )
    empty_eval = CandidateEvaluation(
        candidate=_candidate(1),
        certification_status=passed_status(),
        objective_value=None,
        failure_attribution=None,
        episode_record_path=None,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
    )
    scored_eval = CandidateEvaluation(
        candidate=_candidate(2),
        certification_status=passed_status(),
        objective_value=None,
        failure_attribution=None,
        episode_record_path=episode_path,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
    )

    assert objectives.worst_case_snqi(empty_eval) is None
    assert objectives.worst_case_snqi(scored_eval) == -0.0

    episode_path.write_text(
        json.dumps(
            {
                "metrics": {"success": "bad", "near_misses": 2},
                "outcome": {"collision": True, "timeout": True, "route_complete": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert objectives.worst_case_snqi(scored_eval) == 15.0

    objectives.register_objective("unit_test_constant", lambda _evaluation: 42.0)
    try:
        assert objectives.get_objective("unit_test_constant")(scored_eval) == 42.0
        assert "unit_test_constant" in objectives.list_objectives()
    finally:
        objectives.unregister_objective("unit_test_constant")
    with pytest.raises(ValueError, match="objective name"):
        objectives.register_objective("", lambda _evaluation: 0.0)
    with pytest.raises(ValueError, match="Unknown adversarial objective"):
        objectives.get_objective("missing")


def test_random_sampler_is_deterministic(tmp_path: Path) -> None:
    """Random search must be repeatable for the same seed and search space."""
    template_path = tmp_path / "template.yaml"
    space_path = tmp_path / "space.yaml"
    _write_template(template_path)
    _write_space(space_path)
    payload = yaml.safe_load(space_path.read_text(encoding="utf-8"))
    payload["variables"]["scenario_seed"] = {"min": 7, "max": 9}
    space_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    space = SearchConfig.from_files(
        policy="goal",
        scenario_template=template_path,
        search_space=space_path,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
    ).search_space
    left = RandomCandidateSampler(space, seed=7).sample()
    right = RandomCandidateSampler(space, seed=7).sample()

    assert left == right
    assert isinstance(left.scenario_seed, int)
    assert 7 <= left.scenario_seed <= 9
