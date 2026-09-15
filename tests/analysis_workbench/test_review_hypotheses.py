"""Focused tests for the SREV-21 review-hypotheses component (issue #9292)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    component_request_from_dict,
    experiment_recipe_from_dict,
)
from robot_sf.analysis_workbench.review_hypotheses import (
    COMPONENT_ID,
    COMPONENT_VERSION,
    SUPPORTED_TEMPLATES,
    descriptor,
    run,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "scenario_review" / "review_hypotheses"


def _hypothesis(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "template": "single-pedestrian-speed",
        "pedestrian_id": "ped-0000",
        "factor_value": 1.2,
        "expected_direction": "increase",
        "priority": 0,
        "terminal_condition": "episode_end_or_timeout",
    }
    doc.update(overrides)
    return doc


def _request_doc(hypothesis: dict[str, Any] | None = None, **overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": "component-request.v1",
        "request_id": "srev21-test",
        "component_id": COMPONENT_ID,
        "sources": [
            {
                "artifact_id": "finding-0000",
                "uri": "finding-0000.json",
                "format": "trace_failure_predicates.v1",
            }
        ],
        "config": {"hypothesis": hypothesis if hypothesis is not None else _hypothesis()},
        "output_directory": "out",
    }
    doc.update(overrides)
    return doc


def test_fixture_smoke_produces_valid_recipe_and_descriptor(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    payload = {**payload, "config": {**payload["config"], **config}}
    request = component_request_from_dict(payload)
    result = run(request, base=tmp_path)
    assert result.status == "complete"
    assert len(result.artifacts) == 2
    recipe_path = tmp_path / request.output_directory / "experiment-recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    experiment_recipe_from_dict(recipe)  # schema + duplicate-ID contract
    assert recipe["recipe_id"] == "srev21-smoke-recipe"
    assert len(recipe["interventions"]) == 3
    assert [item["intervention_id"] for item in recipe["interventions"]] == sorted(
        item["intervention_id"] for item in recipe["interventions"]
    )
    assert recipe["budget"]["max_simulator_executions"] == 6
    assert recipe["budget"]["max_elapsed_seconds"] == 600
    assert recipe["budget"]["max_concurrent_local_cpu_processes"] == 1
    assert recipe["control_conditions"]["mode"] == "unchanged-control"
    assert recipe["preservation_destination"] == request.output_directory
    assert any("activation" in name for name in [m["name"] for m in recipe["measurements"]])
    descriptor_doc = json.loads(
        (tmp_path / request.output_directory / "component-descriptor.json").read_text(
            encoding="utf-8"
        )
    )
    assert descriptor_doc["component_id"] == COMPONENT_ID
    assert descriptor_doc["component_version"] == COMPONENT_VERSION
    assert "experiment-recipe.v1" in descriptor_doc["output_types"]


def test_start_delay_template_orders_candidates_deterministically(tmp_path: Path) -> None:
    request = component_request_from_dict(
        _request_doc(_hypothesis(template="single-pedestrian-start-delay", factor_value=2.0))
    )
    result = run(request, base=tmp_path)
    assert result.status == "complete"
    recipe = json.loads(
        (tmp_path / request.output_directory / "experiment-recipe.json").read_text(encoding="utf-8")
    )
    ordered = [(item["priority"], item["intervention_id"]) for item in recipe["interventions"]]
    assert ordered == sorted(ordered)
    assert len({item["intervention_id"] for item in recipe["interventions"]}) == 3


def test_unsupported_template_returns_actionable_unavailable(tmp_path: Path) -> None:
    request = component_request_from_dict(_request_doc(_hypothesis(template="learned-policy")))
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported-hypothesis-template" in result.reason
    assert "single-pedestrian-speed" in result.reason
    assert result.artifacts == ()


def test_corrupt_inputs_fail_closed_without_artifacts(tmp_path: Path) -> None:
    cases = [
        ({}, "missing-hypothesis"),
        (_hypothesis(factor_value=float("inf")), "corrupt-hypothesis"),
        (_hypothesis(expected_direction="sideways"), "corrupt-hypothesis"),
        (_hypothesis(priority=-1), "corrupt-hypothesis"),
        (_hypothesis(pedestrian_id=""), "corrupt-hypothesis"),
    ]
    for index, (hypothesis, reason) in enumerate(cases):
        doc = _request_doc(hypothesis, output_directory=f"out-{index}")
        result = run(component_request_from_dict(doc), base=tmp_path)
        assert result.status == "failed", hypothesis
        assert reason in result.reason, hypothesis
        assert result.artifacts == (), hypothesis


def test_missing_capability_and_output_collision(tmp_path: Path) -> None:
    capped = component_request_from_dict(
        _request_doc(required_capabilities=["video-frames"], output_directory="capped")
    )
    result = run(capped, base=tmp_path)
    assert result.status == "unavailable"
    assert "video-frames" in result.reason
    plain = component_request_from_dict(_request_doc(output_directory="plain"))
    assert run(plain, base=tmp_path).status == "complete"
    again = run(component_request_from_dict(_request_doc(output_directory="plain")), base=tmp_path)
    assert again.status == "failed"
    assert "output-collision" in again.reason
    assert again.artifacts == ()


def test_incompatible_required_version_is_unavailable(tmp_path: Path) -> None:
    doc = _request_doc(_hypothesis(), output_directory="versioned")
    doc["config"] = {**doc["config"], "required_component_version": "2.0.0"}
    result = run(component_request_from_dict(doc), base=tmp_path)
    assert result.status == "unavailable"
    assert "incompatible-required-version" in result.reason


def test_unknown_component_is_unavailable(tmp_path: Path) -> None:
    doc = _request_doc(output_directory="unknown")
    doc["component_id"] = "no-such-component"
    result = run(component_request_from_dict(doc), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported component" in result.reason


def test_deterministic_rerun_matches_digests_and_preserves_source_hash(tmp_path: Path) -> None:
    def once(base: Path) -> tuple[str, dict[str, Any]]:
        request = component_request_from_dict(_request_doc(output_directory="run"))
        result = run(request, base=base)
        assert result.status == "complete"
        recipe = json.loads((base / "run" / "experiment-recipe.json").read_text(encoding="utf-8"))
        return str(result.artifacts[0]["sha256"]), recipe

    first_base = tmp_path / "first"
    second_base = tmp_path / "second"
    first_base.mkdir()
    second_base.mkdir()
    first_digest, first_recipe = once(first_base)
    second_digest, second_recipe = once(second_base)
    assert first_digest == second_digest
    assert first_recipe == second_recipe
    assert (
        first_recipe["control_conditions"]["source_config_sha256"]
        == first_recipe["source_identity"]["hypothesis_sha256"]
    )


def test_descriptor_lists_supported_templates_contract() -> None:
    assert set(SUPPORTED_TEMPLATES) == {
        "single-pedestrian-speed",
        "single-pedestrian-start-delay",
    }
    assert descriptor()["component_id"] == COMPONENT_ID
