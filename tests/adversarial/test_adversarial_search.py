from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.attribution import attribution_from_episode_record
from robot_sf.adversarial.bundle import write_trajectory_csv
from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import CandidateEvaluation, CandidateSpec, Pose2D, SearchConfig
from robot_sf.adversarial.search import run_adversarial_search


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

    result = run_adversarial_search(
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

    result = run_adversarial_search(
        config,
        evaluator=evaluator,
        sampler=_SequenceSampler([_candidate(1)]),
    )

    assert result.best_candidate is None
    assert result.num_invalid_candidates == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["certification_status"]["status"] == "not_available"
    assert manifest["candidates"][0]["error"] == "scenario_cert.v1 adapter is not available"
