"""Focused tests for the SREV-22 review-execute component (issue #9293)."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.analysis_workbench.review_contracts import (
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)
from robot_sf.analysis_workbench.review_execute import (
    COMPONENT_ID,
    COMPONENT_VERSION,
    _run_owned_child,
    descriptor,
    run,
    validate_execute_config,
)

FIXTURES = "tests/fixtures/scenario_review/review_execute"


def _fixture_json(name: str) -> dict[str, Any]:
    with open(f"{FIXTURES}/{name}", encoding="utf-8") as handle:
        return json.load(handle)


def _fixture_request(**config_overrides: Any) -> Any:
    payload = _fixture_json("request.json")
    config = _fixture_json("config.json")
    for key, value in config_overrides.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    payload["config"] = config
    request = component_request_from_dict(payload)
    assert request.component_id == COMPONENT_ID
    return request


def _logical_ledger(path: Path) -> dict[str, Any]:
    ledger = json.loads((path / "attempt-ledger.json").read_text(encoding="utf-8"))
    for entry in ledger["attempts"]:
        entry.pop("elapsed_s", None)
    ledger.pop("wall_elapsed_s", None)
    return ledger


def test_descriptor_validates_against_contract_schema() -> None:
    described = component_descriptor_from_dict(descriptor())
    assert described.component_id == COMPONENT_ID
    assert described.component_version == COMPONENT_VERSION
    assert described.required_capabilities == ("bounded-execution",)


def test_validate_execute_config_rejects_unknown_keys() -> None:
    from robot_sf.analysis_workbench.review_execute import ReviewExecuteError

    with pytest.raises(ReviewExecuteError, match="unknown config keys"):
        validate_execute_config({"recipe": {}, "arbitrary_code": "rm -rf /"})
    with pytest.raises(ReviewExecuteError, match="config.recipe"):
        validate_execute_config({})


def test_fixture_run_completes_with_measured_verdicts(tmp_path: Path) -> None:
    request = _fixture_request()
    result = run(request, base=tmp_path)
    assert result.status == "complete"
    envelope = component_result_from_dict(
        {
            "schema_version": "component-result.v1",
            "request_id": result.request_id,
            "component_id": result.component_id,
            "status": result.status,
            "artifacts": [dict(entry) for entry in result.artifacts],
            "diagnostics": [dict(entry) for entry in result.diagnostics],
            "provenance": dict(result.provenance),
            "reason": result.reason,
        }
    )
    assert envelope.status == "complete"
    output_dir = tmp_path / "srev-22-smoke"
    by_id = {entry["artifact_id"]: entry for entry in result.artifacts}
    assert set(by_id) == {
        "execute-report.json",
        "activation-traces.json",
        "attempt-ledger.json",
        "preservation-manifest.json",
    }
    for entry in by_id.values():
        payload_path = tmp_path / entry["uri"]
        assert payload_path.is_file()
        content = payload_path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    report = json.loads((output_dir / "execute-report.json").read_text(encoding="utf-8"))
    verdicts = {item["intervention_id"]: item for item in report["candidates"]}
    assert verdicts["ped-speed-up"]["status"] == "complete"
    assert verdicts["ped-speed-up"]["verdict"] == "survived"
    assert verdicts["ped-speed-down"]["status"] == "complete"
    assert verdicts["ped-speed-down"]["verdict"] == "falsified"
    assert verdicts["ped-start-delay"]["status"] == "unavailable"
    assert "intervention_not_executable" in verdicts["ped-start-delay"]["reason"]
    # Activation is measured from executed trajectories, never from requested config.
    assert verdicts["ped-speed-up"]["control_activated"] is True
    assert verdicts["ped-speed-up"]["treatment_activated"] is True
    assert verdicts["ped-speed-up"]["nonintervened_config_match"] is True
    traces = json.loads((output_dir / "activation-traces.json").read_text(encoding="utf-8"))
    assert len(traces["traces"]) == 2
    ledger = json.loads((output_dir / "attempt-ledger.json").read_text(encoding="utf-8"))
    assert ledger["executions_consumed"] == 4
    assert len(ledger["attempts"]) == 4
    manifest = json.loads((output_dir / "preservation-manifest.json").read_text(encoding="utf-8"))
    assert manifest["retrieval_destination"] == "external:post-execution-preservation"
    assert manifest["artifacts"]["execute-report.json"] == by_id["execute-report.json"]["sha256"]


def test_repeated_runs_agree_on_logical_artifacts(tmp_path: Path) -> None:
    request = _fixture_request(horizon_steps=20, max_candidates=1)
    first = run(request, base=tmp_path)
    assert first.status == "complete"
    first_dir = tmp_path / "srev-22-smoke"
    first_report = json.loads((first_dir / "execute-report.json").read_text(encoding="utf-8"))
    first_traces = (first_dir / "activation-traces.json").read_bytes()
    first_ledger = _logical_ledger(first_dir)

    second_payload = _fixture_json("request.json")
    second_payload["config"] = _fixture_json("config.json")
    second_payload["config"]["horizon_steps"] = 20
    second_payload["config"]["max_candidates"] = 1
    second_payload["output_directory"] = "srev-22-rerun"
    second = run(component_request_from_dict(second_payload), base=tmp_path)
    assert second.status == "complete"
    second_dir = tmp_path / "srev-22-rerun"
    second_report = json.loads((second_dir / "execute-report.json").read_text(encoding="utf-8"))
    assert second_report["candidates"] == first_report["candidates"]
    assert (second_dir / "activation-traces.json").read_bytes() == first_traces
    assert _logical_ledger(second_dir) == first_ledger


def test_corrupt_recipe_fails_without_artifacts(tmp_path: Path) -> None:
    config = _fixture_json("config.json")
    config["recipe"] = {"schema_version": "experiment-recipe.v1", "recipe_id": "broken"}
    payload = _fixture_json("request.json")
    payload["config"] = config
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "failed"
    assert "corrupt_recipe" in result.reason
    assert result.artifacts == ()


def test_missing_capability_is_unavailable(tmp_path: Path) -> None:
    payload = _fixture_json("request.json")
    payload["config"] = _fixture_json("config.json")
    payload["required_capabilities"] = ["video-frames"]
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "unavailable"
    assert "video-frames" in result.reason


def test_incompatible_required_version_is_unavailable(tmp_path: Path) -> None:
    request = _fixture_request(required_component_version="99.0.0")
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "incompatible_version" in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    payload = _fixture_json("request.json")
    payload["config"] = _fixture_json("config.json")
    payload["component_id"] = "no-such-component"
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported component" in result.reason


def test_output_collision_fails_and_resume_needs_a_ledger(tmp_path: Path) -> None:
    payload = _fixture_json("request.json")
    payload["config"] = _fixture_json("config.json")
    (tmp_path / "srev-22-smoke").mkdir(parents=True)
    result = run(component_request_from_dict(copy.deepcopy(payload)), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason
    resume_result = run(
        component_request_from_dict(copy.deepcopy(payload)), base=tmp_path, resume=True
    )
    assert resume_result.status == "failed"
    assert "ledger" in resume_result.reason


def test_exhausted_execution_budget_reports_partial(tmp_path: Path) -> None:
    request = _fixture_request(max_executions=2)
    result = run(request, base=tmp_path)
    assert result.status == "partial"
    assert "execution_budget_exhausted" in result.reason
    assert result.artifacts == ()
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0]["status"] == "complete"


def test_owned_child_timeout_terminates() -> None:
    outcome = _run_owned_child({"sleep_s": 30.0}, 0.5, target="sleep")
    assert outcome["outcome"] == "timeout"
    assert "terminated" in str(outcome.get("error", ""))
