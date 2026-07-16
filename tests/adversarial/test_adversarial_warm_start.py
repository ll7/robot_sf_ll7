"""Tests for knife-edge warm-start wiring into the adversarial samplers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial import warm_start as warm_start_module
from robot_sf.adversarial.bundle import write_json
from robot_sf.adversarial.certification import passed_status
from robot_sf.adversarial.config import (
    CandidateEvaluation,
    CandidateSpec,
    Pose2D,
    SearchConfig,
    SearchSpaceConfig,
    WarmStartCandidate,
)
from robot_sf.adversarial.samplers import (
    CmaEsCandidateSampler,
    CoordinateRefinementSampler,
    OptunaCandidateSampler,
    RandomCandidateSampler,
)
from robot_sf.adversarial.search import run_adversarial_search
from robot_sf.adversarial.warm_start import (
    extract_warm_starts,
    load_flip_report,
    warm_vs_cold_pilot,
)
from robot_sf.common.artifact_paths import get_repository_root


def _space() -> SearchSpaceConfig:
    """Build a fixed-range search space fixture."""
    return SearchSpaceConfig.from_mapping(
        {
            "variables": {
                "start_x": {"min": 1.0, "max": 1.0},
                "start_y": {"min": 2.0, "max": 2.0},
                "goal_x": {"min": 5.0, "max": 5.0},
                "goal_y": {"min": 2.0, "max": 2.0},
                "spawn_time_s": {"min": 0.0, "max": 0.0},
                "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                "scenario_seed": {"min": 7, "max": 9},
            },
            "constraints": {"min_start_goal_distance_m": 0.25},
        }
    )


def _warm(seed: int, *, planner: str = "goal", margin: float = 0.1) -> WarmStartCandidate:
    """Build one knife-edge warm-start fixture."""
    return WarmStartCandidate(
        candidate=CandidateSpec(
            start=Pose2D(1.0, 2.0),
            goal=Pose2D(5.0, 2.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=seed,
        ),
        scenario="doorway",
        planner=planner,
        outcome_margin=margin,
    )


def _evaluate(candidate: CandidateSpec, *, collapse_seed: int = 8) -> CandidateEvaluation:
    """Build a deterministic evaluation for one candidate."""
    return CandidateEvaluation(
        candidate=candidate,
        certification_status=passed_status(),
        objective_value=1.0 if candidate.scenario_seed == collapse_seed else 0.0,
        failure_attribution=None,
        episode_record_path=None,
        trajectory_csv_path=None,
        scenario_yaml_path=None,
    )


def test_random_sampler_yields_warm_starts_first() -> None:
    """Random sampler should return warm starts before cold random sampling."""
    space = _space()
    warm = _warm(7)
    sampler = RandomCandidateSampler(space, seed=3, warm_start=[warm])
    assert sampler.sample() == warm.candidate


def test_coordinate_sampler_yields_warm_starts_first() -> None:
    """Coordinate sampler should yield warm starts before the midpoint."""
    space = _space()
    warm = _warm(7)
    sampler = CoordinateRefinementSampler(space, seed=3, warm_start=[warm])
    assert sampler.sample() == warm.candidate


def test_optuna_sampler_enqueues_warm_starts() -> None:
    """Optuna sampler should evaluate the enqueued warm start first."""
    space = _space()
    warm = _warm(7)
    sampler = OptunaCandidateSampler(space, seed=11, warm_start=[warm])
    first = sampler.sample()
    assert first == warm.candidate
    sampler.observe(_evaluate(first))
    assert len(sampler._study.trials) == 1
    assert sampler._study.trials[0].value == pytest.approx(0.0)
    second = sampler.sample()
    assert space.validate_candidate(second) == []
    assert len(sampler._study.trials) == 2


def test_cmaes_sampler_returns_warm_starts_first() -> None:
    """CMA-ES sampler should drain warm starts before exploring."""
    space = _space()
    warm = _warm(7)
    sampler = CmaEsCandidateSampler(space, seed=7, popsize=4, warm_start=[warm])
    assert sampler.sample() == warm.candidate


def test_cmaes_sampler_uses_warm_start_as_x0() -> None:
    """CMA-ES sampler should center exploration on the first warm start."""
    space = SearchSpaceConfig.from_mapping(
        {
            "variables": {
                "start_x": {"min": 1.0, "max": 3.0},
                "start_y": {"min": 2.0, "max": 2.0},
                "goal_x": {"min": 5.0, "max": 5.0},
                "goal_y": {"min": 2.0, "max": 2.0},
                "spawn_time_s": {"min": 0.0, "max": 0.0},
                "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                "scenario_seed": {"min": 7, "max": 9},
            },
            "constraints": {"min_start_goal_distance_m": 0.25},
        }
    )
    warm = WarmStartCandidate(
        candidate=CandidateSpec(
            start=Pose2D(2.5, 2.0),
            goal=Pose2D(5.0, 2.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=7,
        ),
        scenario="doorway",
        planner="goal",
        outcome_margin=0.1,
    )
    sampler = CmaEsCandidateSampler(space, seed=7, popsize=4, warm_start=[warm])
    first = sampler.sample()
    assert first.start.x == pytest.approx(2.5)
    sampler.observe(_evaluate(first))
    assert sampler._observed == []
    assert space.validate_candidate(sampler.sample()) == []


def test_search_config_validates_warm_start_inside_space(tmp_path: Path) -> None:
    """SearchConfig should reject warm starts outside the search space."""
    space = _space()
    template = tmp_path / "template.yaml"
    template.write_text(
        "scenarios:\n  - name: template\n    map_id: classic_cross_trap\n"
        "    simulation_config:\n      max_episode_steps: 30\n      ped_density: 0.0\n"
        "    robot_config: {}\n    metadata:\n      archetype: test\n    seeds: [1]\n",
        encoding="utf-8",
    )
    space_path = tmp_path / "space.yaml"
    space_path.write_text(
        "variables:\n"
        "  start_x: {min: 1.0, max: 1.0}\n"
        "  start_y: {min: 2.0, max: 2.0}\n"
        "  goal_x: {min: 5.0, max: 5.0}\n"
        "  goal_y: {min: 2.0, max: 2.0}\n"
        "  spawn_time_s: {min: 0.0, max: 0.0}\n"
        "  pedestrian_speed_mps: {min: 1.0, max: 1.0}\n"
        "  pedestrian_delay_s: {min: 0.0, max: 0.0}\n"
        "  scenario_seed: {min: 7.0, max: 9.0}\n"
        "constraints:\n  min_start_goal_distance_m: 0.25\n",
        encoding="utf-8",
    )
    out_of_space = WarmStartCandidate(
        candidate=CandidateSpec(
            start=Pose2D(1.0, 2.0),
            goal=Pose2D(9.0, 2.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=7,
        ),
        scenario="doorway",
        planner="goal",
    )
    with pytest.raises(ValueError, match="outside search space"):
        config = SearchConfig(
            policy="goal",
            scenario_template=template,
            search_space_path=space_path,
            search_space=space,
            objective="worst_case_snqi",
            output_dir=tmp_path / "out",
            warm_start=(out_of_space,),
        )
        config.validate()


def test_extract_warm_starts_from_flip_report() -> None:
    """Extractor should select only near-boundary entries inside the space."""
    space = _space()
    report = {
        "entries": [
            {
                "scenario": "doorway",
                "planner": "goal",
                "outcome_margin": 0.05,
                "candidate": {
                    "start": {"x": 1.0, "y": 2.0},
                    "goal": {"x": 5.0, "y": 2.0},
                    "scenario_seed": 7,
                },
            },
            {
                "scenario": "doorway",
                "planner": "goal",
                "outcome_margin": 0.9,
                "candidate": {
                    "start": {"x": 1.0, "y": 2.0},
                    "goal": {"x": 5.0, "y": 2.0},
                    "scenario_seed": 8,
                },
            },
            {
                "scenario": "doorway",
                "planner": "goal",
                "outcome_margin": 0.1,
                "candidate": {
                    "start": {"x": 1.0, "y": 2.0},
                    "goal": {"x": 5.0, "y": 2.0},
                    "scenario_seed": 9,
                },
            },
        ]
    }
    extraction = extract_warm_starts(report, search_space=space, margin_threshold=0.5)
    assert extraction.num_near_boundary == 2
    assert extraction.num_selected == 2
    seeds = {warm.candidate.scenario_seed for warm in extraction.warm_starts}
    assert seeds == {7, 9}
    assert extraction.schema_version == warm_start_module.WARM_START_SCHEMA_VERSION


def test_extract_warm_starts_bare_list_and_file(tmp_path: Path) -> None:
    """Extractor should accept a bare list and a JSON file path via load_flip_report."""
    space = _space()
    entries = [
        {
            "scenario": "doorway",
            "planner": "goal",
            "outcome_margin": 0.2,
            "candidate": {
                "start": {"x": 1.0, "y": 2.0},
                "goal": {"x": 5.0, "y": 2.0},
                "scenario_seed": 7,
            },
        },
    ]
    extraction = extract_warm_starts(entries, search_space=space)
    assert extraction.num_selected == 1

    report_path = tmp_path / "flip_report.json"
    report_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    loaded = load_flip_report(report_path)
    extraction2 = extract_warm_starts(loaded, search_space=space, source=str(report_path))
    assert extraction2.source == str(report_path)
    assert extraction2.num_selected == 1


def test_extract_warm_starts_rejects_each_malformed_entry() -> None:
    """Malformed report entries should be rejected individually without aborting extraction."""
    valid_candidate = {
        "start": {"x": 1.0, "y": 2.0},
        "goal": {"x": 5.0, "y": 2.0},
        "scenario_seed": 7,
    }
    entries = [
        "not-a-mapping",
        {
            "scenario": "doorway",
            "planner": "goal",
            "outcome_margin": 0.1,
            "candidate": {**valid_candidate, "scenario_seed": "not-an-integer"},
        },
        {
            "scenario": "doorway",
            "planner": "goal",
            "outcome_margin": 0.1,
            "candidate": {**valid_candidate, "start": {"x": "bad", "y": 2.0}},
        },
        {
            "scenario": "doorway",
            "planner": "goal",
            "outcome_margin": 0.1,
            "candidate": {**valid_candidate, "pedestrian_speed_mps": "bad"},
        },
    ]
    extraction = extract_warm_starts(entries, search_space=_space())
    assert extraction.num_selected == 0
    assert len(extraction.rejected) == len(entries)
    assert extraction.rejected[0]["reason"] == "entry must be a mapping"


def test_warm_started_search_finds_failure_faster_than_cold() -> None:
    """Warm-started Optuna should find the collapse seed at least as early as cold."""
    space = _space()
    warm = _warm(8)

    def make_sampler(*, warm_start: tuple[WarmStartCandidate, ...]) -> OptunaCandidateSampler:
        return OptunaCandidateSampler(space, seed=5, warm_start=warm_start)

    cold = make_sampler(warm_start=())
    warm_sampler = make_sampler(warm_start=(warm,))

    budget = 8
    cold_first_failure = None
    for index in range(budget):
        candidate = cold.sample()
        cold.observe(_evaluate(candidate, collapse_seed=8))
        if candidate.scenario_seed == 8 and cold_first_failure is None:
            cold_first_failure = index
            break

    warm_first_failure = None
    for index in range(budget):
        candidate = warm_sampler.sample()
        warm_sampler.observe(_evaluate(candidate, collapse_seed=8))
        if candidate.scenario_seed == 8 and warm_first_failure is None:
            warm_first_failure = index
            break

    assert warm_first_failure is not None
    assert cold_first_failure is not None
    assert warm_first_failure <= cold_first_failure


def test_search_config_round_trips_warm_start_json(tmp_path: Path) -> None:
    """SearchConfig warm starts should serialize and re-validate via to_json."""
    space = _space()
    config = SearchConfig(
        policy="goal",
        scenario_template=tmp_path / "template.yaml",
        search_space_path=tmp_path / "space.yaml",
        search_space=space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        warm_start=(_warm(7),),
    )
    payload = config.to_json()
    assert payload["warm_start"][0]["candidate"]["scenario_seed"] == 7
    assert payload["warm_start"][0]["scenario"] == "doorway"


def _synthetic_collapse_evaluator(seed: int):
    """Return an evaluator that flags one seed as a stable collision."""

    def evaluator(_config, candidate, scenario_yaml_path, candidate_dir):
        collapse = candidate.scenario_seed == seed
        record = {
            "episode_id": f"synthetic-{candidate.scenario_seed}",
            "seed": int(candidate.scenario_seed),
            "status": "collision" if collapse else "success",
            "steps": 1,
            "termination_reason": "collision" if collapse else "success",
            "outcome": {
                "route_complete": not collapse,
                "collision": collapse,
                "timeout": False,
            },
            "metrics": {"snqi": 0.0 if collapse else 1.0, "success": 0.0 if collapse else 1.0},
        }
        episode_path = candidate_dir / "episode_records.jsonl"
        episode_path.write_text(
            __import__("json").dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        from robot_sf.adversarial.attribution import attribution_from_episode_record

        return CandidateEvaluation(
            candidate=candidate,
            certification_status=passed_status("synthetic"),
            objective_value=1.0 if collapse else 0.0,
            failure_attribution=attribution_from_episode_record(record),
            episode_record_path=episode_path,
            trajectory_csv_path=None,
            scenario_yaml_path=scenario_yaml_path,
            bundle_path=candidate_dir,
        )

    return evaluator


def _pilot_config(
    tmp_path: Path,
    budget: int = 6,
    *,
    warm_start: tuple[WarmStartCandidate, ...] = (),
) -> SearchConfig:
    """Build an Optuna-friendly search config with a real seed range."""
    template = tmp_path / "template.yaml"
    template.write_text(
        "scenarios:\n  - name: template\n    map_id: classic_cross_trap\n"
        "    simulation_config:\n      max_episode_steps: 30\n      ped_density: 0.0\n"
        "    robot_config: {}\n    metadata:\n      archetype: test\n    seeds: [1]\n",
        encoding="utf-8",
    )
    space = SearchSpaceConfig.from_mapping(
        {
            "variables": {
                "start_x": {"min": 1.0, "max": 1.0},
                "start_y": {"min": 2.0, "max": 2.0},
                "goal_x": {"min": 5.0, "max": 5.0},
                "goal_y": {"min": 2.0, "max": 2.0},
                "spawn_time_s": {"min": 0.0, "max": 0.0},
                "pedestrian_speed_mps": {"min": 1.0, "max": 1.0},
                "pedestrian_delay_s": {"min": 0.0, "max": 0.0},
                "scenario_seed": {"min": 9, "max": 14},
            },
            "constraints": {"min_start_goal_distance_m": 0.25},
        }
    )
    space_path = tmp_path / "space.yaml"
    write_json(space_path, space.to_json())
    return SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=space_path,
        objective="worst_case_snqi",
        output_dir=tmp_path / "out",
        budget=budget,
        seed=3,
        workers=1,
        warm_start=warm_start,
    )


def test_warm_vs_cold_pilot_reports_faster_warm_start(tmp_path: Path) -> None:
    """Warm-started Optuna should surface the collapse seed no later than cold."""
    config = _pilot_config(tmp_path, budget=6)
    collapse_seed = 9
    evaluator = _synthetic_collapse_evaluator(collapse_seed)
    certifier = lambda _c, _p, _r: passed_status("synthetic")  # noqa: E731
    warm = WarmStartCandidate(
        candidate=CandidateSpec(
            start=Pose2D(1.0, 2.0),
            goal=Pose2D(5.0, 2.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=1.0,
            pedestrian_delay_s=0.0,
            scenario_seed=collapse_seed,
        ),
        scenario="doorway",
        planner="goal",
        outcome_margin=0.1,
    )
    report = warm_vs_cold_pilot(
        config,
        warm_start=(warm,),
        objective="worst_case_snqi",
        evaluator=evaluator,
        certifier=certifier,
        output_dir=tmp_path / "pilot",
        sampler_name="optuna",
        collapse_predicate=lambda c: c.scenario_seed == collapse_seed,
    )
    assert report.num_warm_starts == 1
    assert report.warm_first_failure_iteration == 0
    # Warm starts the collapse seed (enqueued) at iteration 0; the pilot must
    # report the warm arm as at least as fast as cold, regardless of whether the
    # cold arm independently samples that seed within the fixed budget.
    assert report.faster
    assert report.cold_first_failure_iteration is None or report.cold_first_failure_iteration >= 0
    assert (tmp_path / "pilot" / "warm_vs_cold_pilot.json").exists()


def test_warm_vs_cold_pilot_rejects_release_evidence_path_before_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pilot execution must fail before either arm writes into release evidence."""
    config = _pilot_config(tmp_path)

    def unexpected_search(*_args, **_kwargs):
        raise AssertionError("search must not start before output-boundary validation")

    monkeypatch.setattr(warm_start_module, "run_adversarial_search", unexpected_search)
    forbidden = get_repository_root() / "output" / "release_evidence" / "issue_5833"
    with pytest.raises(ValueError, match="under output/adversarial"):
        warm_vs_cold_pilot(
            config,
            warm_start=(_warm(9),),
            objective="worst_case_snqi",
            evaluator=_synthetic_collapse_evaluator(9),
            output_dir=forbidden,
        )


def test_warm_started_search_runs_end_to_end(tmp_path: Path) -> None:
    """A full adversarial search run should accept warm starts without error."""
    warm = _warm(12)
    config = _pilot_config(tmp_path, budget=4, warm_start=(warm,))
    result = run_adversarial_search(
        config,
        evaluator=_synthetic_collapse_evaluator(13),
        certifier=lambda _c, _p, _r: passed_status("synthetic"),
    )
    assert result.num_candidates == 4
    assert config.output_dir.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["candidates"][0]["candidate"]["scenario_seed"] == 12
