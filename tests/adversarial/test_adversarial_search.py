from __future__ import annotations

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
from robot_sf.adversarial.certification import failed_status, passed_status
from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, Pose2D, SearchConfig
from robot_sf.adversarial.samplers import RandomCandidateSampler


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


def _config(tmp_path: Path, *, require_certification: bool = False) -> SearchConfig:
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
        sampler=_SequenceSampler([_candidate(1), _candidate(2)]),
    )

    assert result.num_candidates == 2
    assert result.num_invalid_candidates == 0
    assert result.best_objective_value == -0.2
    assert result.best_bundle_path == config.output_dir / "candidate_0001"
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["best_bundle_path"].endswith("candidate_0001")
    assert (config.output_dir / "candidate_0001" / "scenario.yaml").exists()
    assert (config.output_dir / "candidate_0001" / "route_overrides.yaml").exists()


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
        sampler=_SequenceSampler([_candidate(1)]),
    )

    assert result.best_candidate is None
    assert result.num_invalid_candidates == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["certification_status"]["status"] == "not_available"
    assert manifest["candidates"][0]["error"] == "scenario_cert.v1 adapter is not available"


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
    timeout = attribution_from_episode_record(
        {"status": "done", "outcome": {"timeout": True, "route_complete": False}}
    )
    incomplete = attribution_from_episode_record(
        {"status": "done", "outcome": {"route_complete": False}}
    )
    error = attribution_from_error("boom")

    assert collision.primary_failure == "collision"
    assert timeout.primary_failure == "timeout"
    assert incomplete.primary_failure == "incomplete"
    assert error.to_json()["status"] == "evaluation_failed"


def test_certification_adapter_handles_missing_and_mocked_backends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Certification must fail closed when required and normalize adapter payloads."""
    monkeypatch.delitem(sys.modules, "robot_sf.scenario_certification", raising=False)

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


def test_objective_registry_and_fallback_scoring(tmp_path: Path) -> None:
    """Objectives should score SNQI records, fallback failures, and registry errors."""
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
    assert objectives.get_objective("unit_test_constant")(scored_eval) == 42.0
    assert "unit_test_constant" in objectives.list_objectives()
    with pytest.raises(ValueError, match="objective name"):
        objectives.register_objective("", lambda _evaluation: 0.0)
    with pytest.raises(ValueError, match="Unknown adversarial objective"):
        objectives.get_objective("missing")


def test_random_sampler_is_deterministic(tmp_path: Path) -> None:
    """Random search must be repeatable for the same seed and search space."""
    space_path = tmp_path / "space.yaml"
    _write_space(space_path)
    space = SearchConfig.from_files(
        policy="goal",
        scenario_template=tmp_path / "template.yaml",
        search_space=space_path,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
    ).search_space
    left = RandomCandidateSampler(space, seed=7).sample()
    right = RandomCandidateSampler(space, seed=7).sample()

    assert left == right
