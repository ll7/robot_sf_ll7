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
    Pose2D,
    SearchConfig,
    SearchSpaceConfig,
)
from robot_sf.adversarial.io import read_first_jsonl_record
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
