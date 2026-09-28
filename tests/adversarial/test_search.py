"""Fixture-only coverage for search's scenario-admissibility consumer."""

# fast-lane: fast-contract

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import search
from robot_sf.adversarial.certification import failed_status, passed_status
from robot_sf.adversarial.config import CandidateSpec, Pose2D, SearchConfig
from robot_sf.adversarial.scenario_admissibility import ScenarioAdmissibilityVerdict


class _SequenceSampler:
    """Return one prepared candidate without invoking a production sampler."""

    def __init__(self, candidate: CandidateSpec) -> None:
        self.candidate = candidate

    def sample(self) -> CandidateSpec:
        """Return the fixture candidate."""
        return self.candidate


def _candidate(*, start_x: float = 1.0) -> CandidateSpec:
    """Build one static candidate with no runtime or simulator dependencies."""
    return CandidateSpec(
        start=Pose2D(start_x, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.0,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.0,
        scenario_seed=7,
    )


def _config(
    tmp_path: Path,
    *,
    require_certification: bool = True,
    apply_admissibility_filter: bool = False,
) -> SearchConfig:
    """Write the smallest config accepted by the public search runner."""
    template = tmp_path / "template.yaml"
    template.write_text(
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
    search_space = tmp_path / "space.yaml"
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
                    "scenario_seed": {"min": 7.0, "max": 7.0},
                },
                "constraints": {"min_start_goal_distance_m": 0.5},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return SearchConfig.from_files(
        policy="goal",
        scenario_template=template,
        search_space=search_space,
        objective="worst_case_snqi",
        output_dir=tmp_path / "output",
        budget=1,
        seed=17,
        require_certification=require_certification,
        apply_admissibility_filter=apply_admissibility_filter,
    )


def _verdict(
    case_id: str,
    scenario_id: str | None,
    *,
    disposition: str = "retain",
    verdict: str = "admissible_feasibility_unknown",
) -> ScenarioAdmissibilityVerdict:
    """Build one typed, identity-bound result for an injected classifier."""
    return ScenarioAdmissibilityVerdict(
        case_id=case_id,
        scenario_id=scenario_id,
        verdict=verdict,
        target_planner_outcome="not_evaluated",
        search_disposition=disposition,
        reason_codes=("fixture",),
        assumptions={},
        evidence={"source": "fixture"},
    )


def _evaluator_must_not_run(*_args: Any, **_kwargs: Any) -> None:
    """Fail if a rejected case crosses the evaluator boundary."""
    raise AssertionError("rejected candidate reached the evaluator")


def test_bound_rejection_is_recorded_and_skips_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a schema-valid rejection bound to the materialized case is filtered."""
    config = _config(tmp_path, apply_admissibility_filter=True)

    def reject_bound_case(case_id: str, *, scenario_id: str | None, **_kwargs: Any):
        return _verdict(
            case_id,
            scenario_id,
            disposition="reject",
            verdict="structurally_invalid",
        )

    monkeypatch.setattr(search, "classify_scenario_admissibility", reject_bound_case)
    result = search.run_adversarial_search(
        config,
        evaluator=_evaluator_must_not_run,
        certifier=lambda *_args: passed_status("fixture certification"),
        sampler=_SequenceSampler(_candidate()),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    row = manifest["candidates"][0]
    assert result.num_invalid_candidates == 1
    assert row["evaluation_disposition"] == "rejected_by_admissibility"
    assert (
        row["certification_status"]["details"]["scenario_admissibility"]["search_disposition"]
        == "reject"
    )
    assert (config.output_dir / "candidate_0000" / "failure_attribution.json").is_file()


def test_classifier_error_stays_unknown_and_candidate_remains_evaluable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable classification is persisted as unknown and does not filter a candidate."""
    config = _config(tmp_path, apply_admissibility_filter=True)

    def unavailable_classifier(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("fixture classifier unavailable")

    def evaluator_raises(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("fixture evaluator failure")

    monkeypatch.setattr(search, "classify_scenario_admissibility", unavailable_classifier)
    result = search.run_adversarial_search(
        config,
        evaluator=evaluator_raises,
        certifier=lambda *_args: passed_status("fixture certification"),
        sampler=_SequenceSampler(_candidate()),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    row = manifest["candidates"][0]
    verdict = row["certification_status"]["details"]["scenario_admissibility"]
    assert result.num_invalid_candidates == 0
    assert result.num_failed_evaluations == 1
    assert verdict["verdict"] == "admissible_feasibility_unknown"
    assert verdict["search_disposition"] == "retain"
    assert verdict["evidence"]["classification_adapter"]["status"] == "unavailable"
    assert row["evaluation_disposition"] == "evaluator_invoked"


def test_invalid_search_space_candidate_is_recorded_without_materialization(
    tmp_path: Path,
) -> None:
    """An out-of-space candidate stays in the ledger as a rejected row."""
    config = _config(tmp_path)
    result = search.run_adversarial_search(
        config,
        evaluator=_evaluator_must_not_run,
        certifier=lambda *_args: pytest.fail("invalid candidate reached certification"),
        sampler=_SequenceSampler(_candidate(start_x=3.0)),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    row = manifest["candidates"][0]
    assert result.num_invalid_candidates == 1
    assert row["evaluation_disposition"] == "rejected_by_search_space"
    assert row["certification_status"]["details"]["scenario_admissibility"]["verdict"] == (
        "admissible_feasibility_unknown"
    )
    assert not (config.output_dir / "candidate_0000" / "scenario.yaml").exists()


def test_failed_certification_remains_distinct_from_admissibility(
    tmp_path: Path,
) -> None:
    """A target certification failure is retained as its own search disposition."""
    config = _config(tmp_path)
    result = search.run_adversarial_search(
        config,
        evaluator=_evaluator_must_not_run,
        certifier=lambda *_args: failed_status("fixture certification failure"),
        sampler=_SequenceSampler(_candidate()),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    row = manifest["candidates"][0]
    assert result.num_invalid_candidates == 1
    assert row["evaluation_disposition"] == "rejected_by_certification"
    assert row["certification_status"]["details"]["scenario_admissibility"]["verdict"] == (
        "admissible_feasibility_unknown"
    )


def test_matching_certificate_is_passed_to_admissibility_classifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected scenario row binds exactly one matching certificate."""
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(
        yaml.safe_dump({"scenarios": [{"name": "bound-case"}]}), encoding="utf-8"
    )
    certificate = {"scenario_id": "bound-case", "certificate_digest": "fixture-digest"}
    status = replace(passed_status("fixture"), details={"certificates": [certificate]})
    observed: list[Any] = []

    def capture_certificate(
        case_id: str,
        *,
        scenario_id: str | None,
        scenario_certificate: dict[str, Any] | None,
        **_kwargs: Any,
    ) -> ScenarioAdmissibilityVerdict:
        observed.append((scenario_id, scenario_certificate))
        return _verdict(case_id, scenario_id)

    monkeypatch.setattr(search, "classify_scenario_admissibility", capture_certificate)
    status_with_verdict, _case_id = search._attach_scenario_admissibility(
        status,
        candidate=_candidate(),
        scenario_yaml_path=scenario_path,
    )

    assert observed == [("bound-case", certificate)]
    assert status_with_verdict.details["scenario_admissibility"]["scenario_id"] == "bound-case"


def test_malformed_admissibility_record_cannot_reject_candidate() -> None:
    """Malformed persisted evidence cannot be treated as an exclusion."""
    status = replace(
        passed_status("fixture"),
        details={"scenario_admissibility": {"schema_version": "unsupported"}},
    )

    assert not search._scenario_admissibility_rejects(
        status,
        expected_case_id="candidate-case",
    )
