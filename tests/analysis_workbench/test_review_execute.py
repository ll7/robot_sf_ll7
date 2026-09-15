"""Focused tests for the SREV-22 review-execute component (issue #9293)."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    ComponentResult,
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


def _fixture_episode_identity() -> dict[str, Any]:
    source_identity = _fixture_json("config.json")["recipe"]["source_identity"]
    return {
        "scenario_id": source_identity["scenario_id"],
        "source_ref": dict(source_identity["source_ref"]),
    }


def _patch_fake_execution(
    monkeypatch: pytest.MonkeyPatch, *, fail_treatment_speed: float | None = None
) -> list[dict[str, Any]]:
    """Supply deterministic telemetry without starting the simulator child."""
    import robot_sf.analysis_workbench.review_execute as review_execute_module

    calls: list[dict[str, Any]] = []

    def _fake_child(job: dict[str, Any], _timeout_s: float, **_kwargs: Any) -> dict[str, Any]:
        calls.append(dict(job))
        speed = float(job["ped_speed_m_s"])
        if fail_treatment_speed is not None and speed == fail_treatment_speed:
            return {"outcome": "error", "error": "synthetic child failure"}
        horizon = int(job["horizon_steps"])
        ped_traj = [[10.0, step * speed * 0.1] for step in range(horizon + 1)]
        robot_traj = [[step * 0.1, 0.0] for step in range(horizon + 1)]
        return {
            "outcome": "ok",
            "payload": {
                "status": "ok",
                "steps_completed": horizon,
                "ped_traj": ped_traj,
                "robot_traj": robot_traj,
            },
        }

    monkeypatch.setattr(review_execute_module, "_run_owned_child", _fake_child)
    return calls


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
    assert manifest["evidence_boundary"] == "diagnostic_only"
    assert manifest["scientific_claim_allowed"] is False
    assert manifest["dependent_family_status"] == "standalone_fixture_only"


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


def test_output_symlink_and_preservation_escape_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "link").symlink_to(target, target_is_directory=True)
    linked_request = replace(_fixture_request(), output_directory="link")
    linked_result = run(linked_request, base=tmp_path)
    assert linked_result.status == "failed"
    assert "symlink" in linked_result.reason

    (tmp_path / "nested-link").symlink_to(target, target_is_directory=True)
    nested_request = replace(_fixture_request(), output_directory="nested-link/output")
    nested_result = run(nested_request, base=tmp_path)
    assert nested_result.status == "failed"
    assert "symlink" in nested_result.reason

    unsafe_destination = _recipe_payload(preservation_destination="/tmp/not-owned")
    unsafe_result = run(component_request_from_dict(unsafe_destination), base=tmp_path)
    assert unsafe_result.status == "failed"
    assert "invalid_preservation_destination" in unsafe_result.reason


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


def _recipe_payload(**overrides: Any) -> dict[str, Any]:
    config = _fixture_json("config.json")
    recipe = config["recipe"]
    recipe.update(overrides)
    payload = _fixture_json("request.json")
    payload["config"] = config
    return payload


def test_config_field_validation_branches() -> None:
    from robot_sf.analysis_workbench.review_execute import ReviewExecuteError

    base = {"recipe": {}}
    with pytest.raises(ReviewExecuteError, match="unsupported planner"):
        validate_execute_config({**base, "planner": "learned"})
    with pytest.raises(ReviewExecuteError, match="seed"):
        validate_execute_config({**base, "seed": -1})
    with pytest.raises(ReviewExecuteError, match="horizon_steps"):
        validate_execute_config({**base, "horizon_steps": 0})
    with pytest.raises(ReviewExecuteError, match="robot_speed_m_s"):
        validate_execute_config({**base, "robot_speed_m_s": 99.0})
    with pytest.raises(ReviewExecuteError, match="max_candidates"):
        validate_execute_config({**base, "max_candidates": 0})
    with pytest.raises(ReviewExecuteError, match="wall_timeout_s"):
        validate_execute_config({**base, "wall_timeout_s": 0.0})
    with pytest.raises(ReviewExecuteError, match="per_execution_timeout_s"):
        validate_execute_config({**base, "per_execution_timeout_s": -2.0})
    with pytest.raises(ReviewExecuteError, match="activation_speed_tolerance"):
        validate_execute_config({**base, "activation_speed_tolerance_m_s": float("nan")})
    with pytest.raises(ReviewExecuteError, match="motion_epsilon_m"):
        validate_execute_config({**base, "motion_epsilon_m": "far"})
    with pytest.raises(ReviewExecuteError, match="required_component_version"):
        validate_execute_config({**base, "required_component_version": 3})
    with pytest.raises(ReviewExecuteError, match="intervention_parameters"):
        validate_execute_config({**base, "intervention_parameters": []})
    with pytest.raises(ReviewExecuteError, match="config must be a mapping"):
        validate_execute_config([])


def test_config_budget_ceilings_and_nested_fields_are_closed() -> None:
    from robot_sf.analysis_workbench.review_execute import ReviewExecuteError

    base = {"recipe": {}}
    with pytest.raises(ReviewExecuteError, match="max_candidates"):
        validate_execute_config({**base, "max_candidates": 4})
    with pytest.raises(ReviewExecuteError, match="max_executions"):
        validate_execute_config({**base, "max_executions": 7})
    with pytest.raises(ReviewExecuteError, match="wall_timeout_s"):
        validate_execute_config({**base, "wall_timeout_s": 601.0})
    with pytest.raises(ReviewExecuteError, match="per_execution_timeout_s"):
        validate_execute_config({**base, "per_execution_timeout_s": 121.0})
    with pytest.raises(ReviewExecuteError, match="unknown keys"):
        validate_execute_config(
            {**base, "intervention_parameters": {"candidate": {"python": "code"}}}
        )


def test_recipe_budget_and_stop_rules_are_enforced(tmp_path: Path) -> None:
    budget_payload = _recipe_payload()
    budget_payload["config"]["recipe"]["budget"]["max_executions"] = 2
    budget_result = run(component_request_from_dict(budget_payload), base=tmp_path)
    assert budget_result.status == "failed"
    assert "invalid_budget" in budget_result.reason

    stop_payload = _recipe_payload()
    stop_payload["config"]["recipe"]["stop_rules"] = ["exhausted_candidates"]
    stop_result = run(component_request_from_dict(stop_payload), base=tmp_path)
    assert stop_result.status == "failed"
    assert "invalid_stop_rules" in stop_result.reason


def test_diagnostic_boundary_rejects_benchmark_source(tmp_path: Path) -> None:
    payload = _recipe_payload()
    payload["config"]["recipe"]["source_identity"]["kind"] = "benchmark"
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "failed"
    assert "invalid_evidence_boundary" in result.reason

    missing_boundary = _recipe_payload()
    missing_boundary["config"]["recipe"]["source_identity"].pop("evidence_boundary")
    missing_result = run(component_request_from_dict(missing_boundary), base=tmp_path)
    assert missing_result.status == "failed"
    assert "invalid_evidence_boundary" in missing_result.reason


def test_fixture_source_binding_rejects_mismatches_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_fake_execution(monkeypatch)

    wrong_scenario = _recipe_payload()
    wrong_scenario["output_directory"] = "wrong-scenario"
    wrong_scenario["config"]["recipe"]["source_identity"]["scenario_id"] = "other-fixture"
    scenario_result = run(component_request_from_dict(wrong_scenario), base=tmp_path)
    assert scenario_result.status == "failed"
    assert "supported fixture" in scenario_result.reason

    missing_source_ref = _recipe_payload()
    missing_source_ref["output_directory"] = "missing-source-ref"
    missing_source_ref["config"]["recipe"]["source_identity"].pop("source_ref")
    source_ref_result = run(component_request_from_dict(missing_source_ref), base=tmp_path)
    assert source_ref_result.status == "failed"
    assert "source_ref" in source_ref_result.reason

    wrong_request_source = _recipe_payload()
    wrong_request_source["output_directory"] = "wrong-request-source"
    wrong_request_source["sources"][0]["uri"] = "other-recipe.json"
    request_source_result = run(component_request_from_dict(wrong_request_source), base=tmp_path)
    assert request_source_result.status == "failed"
    assert "request source" in request_source_result.reason
    assert calls == []


def test_run_rejects_direct_request_output_escape(tmp_path: Path) -> None:
    request = replace(_fixture_request(), output_directory="../srev22-escape")
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "invalid_output_path" in result.reason
    assert not (tmp_path.parent / "srev22-escape").exists()


def test_budget_reserves_a_complete_pair_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_fake_execution(monkeypatch)
    request = _fixture_request(max_candidates=1, max_executions=1)
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "execution_budget_exhausted" in result.reason
    assert calls == []


def test_wall_budget_reserves_complete_pair_before_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_fake_execution(monkeypatch)
    request = _fixture_request(
        max_candidates=1,
        max_executions=2,
        wall_timeout_s=0.19,
        per_execution_timeout_s=0.1,
    )
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "reserving 2 execution(s)" in result.reason
    assert calls == []


def test_wall_budget_reserves_remaining_treatment_after_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import robot_sf.analysis_workbench.review_execute as review_execute_module

    calls = _patch_fake_execution(monkeypatch)
    wall_remaining = iter((0.2, 0.2, 0.09))
    monkeypatch.setattr(
        review_execute_module._Executor,
        "_wall_remaining",
        lambda _executor: next(wall_remaining),
    )
    request = _fixture_request(
        max_candidates=1,
        max_executions=2,
        wall_timeout_s=0.25,
        per_execution_timeout_s=0.1,
    )
    result = run(request, base=tmp_path)
    assert result.status == "partial"
    assert "reserving 1 execution(s)" in result.reason
    assert [job["ped_speed_m_s"] for job in calls] == [1.0]


def test_failed_candidate_is_partial_without_complete_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fake_execution(monkeypatch, fail_treatment_speed=0.5)
    request = _fixture_request(max_candidates=2, max_executions=4)
    result = run(request, base=tmp_path)
    assert result.status == "partial"
    assert "candidate_execution_failed" in result.reason
    assert result.artifacts == ()
    ledger = json.loads((tmp_path / "srev-22-smoke" / "attempt-ledger.json").read_text())
    reports = {report["intervention_id"]: report for report in ledger["candidate_reports"]}
    assert reports["ped-speed-up"]["status"] == "complete"
    assert reports["ped-speed-down"]["status"] == "failed"
    assert ledger["evidence_boundary"] == "diagnostic_only"
    assert ledger["scientific_claim_allowed"] is False
    assert ledger["dependent_family_status"] == "standalone_fixture_only"
    assert not (tmp_path / "srev-22-smoke" / "execute-report.json").exists()


def test_resume_restores_reports_and_does_not_rerun_terminal_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _patch_fake_execution(monkeypatch)
    partial = run(_fixture_request(max_executions=2), base=tmp_path)
    assert partial.status == "partial"
    assert [job["ped_speed_m_s"] for job in calls] == [1.0, 1.5]

    resumed = run(_fixture_request(max_executions=6), base=tmp_path, resume=True)
    assert resumed.status == "complete"
    assert [job["ped_speed_m_s"] for job in calls] == [1.0, 1.5, 1.0, 0.5]
    ledger = json.loads((tmp_path / "srev-22-smoke" / "attempt-ledger.json").read_text())
    assert {report["intervention_id"] for report in ledger["candidate_reports"]} == {
        "ped-speed-up",
        "ped-speed-down",
        "ped-start-delay",
    }
    assert len(ledger["traces"]) == 2


def test_resume_rejects_tampered_ledger_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fake_execution(monkeypatch)
    partial = run(_fixture_request(max_executions=2), base=tmp_path)
    assert partial.status == "partial"
    ledger_path = tmp_path / "srev-22-smoke" / "attempt-ledger.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["recipe_digest"] = "tampered"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    result = run(_fixture_request(max_executions=6), base=tmp_path, resume=True)
    assert result.status == "failed"
    assert "recipe identity mismatch" in result.reason


def test_intervention_update_branches() -> None:
    from robot_sf.analysis_workbench.review_execute import _intervention_update

    update, reason = _intervention_update("nope", {}, control_speed=1.0, control_delay=0.0)
    assert update is None and "unsupported intervention factor" in str(reason)
    update, reason = _intervention_update(
        "single_pedestrian_speed_offset", None, control_speed=1.0, control_delay=0.0
    )
    assert update is None and "missing intervention_parameters" in str(reason)
    update, reason = _intervention_update(
        "single_pedestrian_speed_offset",
        {"speed_delta_m_s": 0.0},
        control_speed=1.0,
        control_delay=0.0,
    )
    assert update is None and "non-zero" in str(reason)
    update, reason = _intervention_update(
        "single_pedestrian_speed_offset",
        {"speed_delta_m_s": 2.0},
        control_speed=1.0,
        control_delay=0.0,
    )
    assert update is None and "fixture bound" in str(reason)
    update, reason = _intervention_update(
        "single_pedestrian_speed_offset",
        {"speed_delta_m_s": -1.0},
        control_speed=1.0,
        control_delay=0.0,
    )
    assert update is None and "validity range" in str(reason)
    update, reason = _intervention_update(
        "single_pedestrian_speed_offset",
        {"speed_delta_m_s": 0.5},
        control_speed=1.0,
        control_delay=0.0,
    )
    assert update == {"ped_speed_m_s": 1.5, "ped_start_delay_s": 0.0} and reason is None
    update, reason = _intervention_update(
        "single_pedestrian_start_delay_offset", {}, control_speed=1.0, control_delay=0.0
    )
    assert update is None and "requires a finite non-zero dt_s" in str(reason)
    update, reason = _intervention_update(
        "single_pedestrian_start_delay_offset",
        {"dt_s": 9.0},
        control_speed=1.0,
        control_delay=0.0,
    )
    assert update is None and "fixture bound" in str(reason)
    update, reason = _intervention_update(
        "single_pedestrian_start_delay_offset",
        {"dt_s": -1.0},
        control_speed=1.0,
        control_delay=0.5,
    )
    assert update is None and "negative" in str(reason)


def test_measurement_selection_branches() -> None:
    from robot_sf.analysis_workbench.review_execute import _measurement_for_recipe

    recipe = _fixture_json("config.json")["recipe"]
    selected, reason = _measurement_for_recipe(recipe)
    assert selected is not None and selected["name"] == "min_robot_ped_distance_m"
    assert reason is None
    selected, reason = _measurement_for_recipe({**recipe, "measurements": []})
    assert selected is None and "no measurements" in str(reason)
    bad_name = {"measurements": [{"name": "vibes", "units": "u", "expected_direction": "increase"}]}
    selected, reason = _measurement_for_recipe({**recipe, **bad_name})
    assert selected is None and "unsupported measurement" in str(reason)
    bad_dir = {
        "measurements": [
            {"name": "ped_mean_speed_m_s", "units": "u", "expected_direction": "sideways"}
        ]
    }
    selected, reason = _measurement_for_recipe({**recipe, **bad_dir})
    assert selected is None and "expected_direction" in str(reason)


def test_telemetry_metrics_rejects_malformed_payloads() -> None:
    from robot_sf.analysis_workbench.review_execute import _telemetry_metrics

    assert _telemetry_metrics({"status": "error"}, horizon=4, motion_epsilon_m=0.05) is None
    assert (
        _telemetry_metrics({"status": "ok", "steps_completed": 3}, horizon=4, motion_epsilon_m=0.05)
        is None
    )
    assert (
        _telemetry_metrics(
            {"status": "ok", "steps_completed": 4, "ped_traj": [[0.0]], "robot_traj": [[0.0]]},
            horizon=4,
            motion_epsilon_m=0.05,
        )
        is None
    )
    assert (
        _telemetry_metrics(
            {
                "status": "ok",
                "steps_completed": 4,
                "ped_traj": [[float("inf")] * 2] * 5,
                "robot_traj": [[0.0, 0.0]] * 5,
            },
            horizon=4,
            motion_epsilon_m=0.05,
        )
        is None
    )
    assert _telemetry_metrics({"status": "ok"}, horizon=4, motion_epsilon_m=0.05) is None
    assert _telemetry_metrics(None, horizon=4, motion_epsilon_m=0.05) is None  # type: ignore[arg-type]


def test_episode_job_reports_errors_without_raising() -> None:
    from robot_sf.analysis_workbench.review_execute import _execute_episode_job

    failed = _execute_episode_job({})
    assert failed["status"] == "error" and "error" in failed
    tiny = _execute_episode_job(
        {
            **_fixture_episode_identity(),
            "seed": 7,
            "horizon_steps": 4,
            "robot_speed_m_s": 1.0,
            "ped_speed_m_s": 1.0,
            "ped_start_delay_s": 0.0,
        }
    )
    assert tiny["status"] == "ok" and tiny["steps_completed"] == 4


def test_episode_job_rejects_unbound_fixture_identity() -> None:
    from robot_sf.analysis_workbench.review_execute import _execute_episode_job

    job = {
        **_fixture_episode_identity(),
        "seed": 7,
        "horizon_steps": 4,
        "robot_speed_m_s": 1.0,
        "ped_speed_m_s": 1.0,
        "ped_start_delay_s": 0.0,
    }
    wrong_scenario = _execute_episode_job({**job, "scenario_id": "other-fixture"})
    assert wrong_scenario["status"] == "error"
    assert "scenario_id" in wrong_scenario["error"]

    wrong_source = _execute_episode_job({**job, "source_ref": {"uri": "other-recipe.json"}})
    assert wrong_source["status"] == "error"
    assert "source_ref" in wrong_source["error"]


def test_child_main_reports_transport_failure() -> None:
    from robot_sf.analysis_workbench.review_execute import _child_main

    class _ExplodingConn:
        def send(self, _payload: Any) -> None:
            raise OSError("pipe gone")

        def close(self) -> None:
            pass

    # Must not raise: the transport failure path is silent by design.
    _child_main("sleep", {"sleep_s": 0.0}, _ExplodingConn())


def test_sleep_job_returns_ok() -> None:
    from robot_sf.analysis_workbench.review_execute import _sleep_job

    assert _sleep_job({"sleep_s": 0.0}) == {"status": "ok"}


def test_owned_child_spawn_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import multiprocessing as multiprocessing_module

    real_context = multiprocessing_module.get_context()

    class _FailingProcess:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise OSError("no fork today")

    monkeypatch.setattr(multiprocessing_module, "get_context", lambda _method=None: real_context)
    monkeypatch.setattr(real_context, "Process", _FailingProcess)
    outcome = _run_owned_child({"sleep_s": 0.0}, 1.0, target="sleep")
    assert outcome["outcome"] == "error" and "spawn failed" in str(outcome.get("error"))


def test_owned_child_interrupt_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    import multiprocessing as multiprocessing_module

    real_context = multiprocessing_module.get_context()
    real_pipe = real_context.Pipe

    class _InterruptConn:
        def poll(self, _timeout: float) -> bool:
            raise KeyboardInterrupt

        def close(self) -> None:
            pass

    def _fake_pipe(*args: Any, **kwargs: Any) -> Any:
        parent, child = real_pipe(*args, **kwargs)
        child.close()
        return _InterruptConn(), parent

    class _NoopProcess:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def join(self, _timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(real_context, "Pipe", _fake_pipe)
    monkeypatch.setattr(real_context, "Process", _NoopProcess)
    monkeypatch.setattr(multiprocessing_module, "get_context", lambda _method=None: real_context)
    outcome = _run_owned_child({"sleep_s": 0.0}, 1.0, target="sleep")
    assert outcome["outcome"] == "interrupted" and "terminated" in str(outcome.get("error"))


def test_repo_commit_unknown_on_tooling_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as subprocess_module

    from robot_sf.analysis_workbench.review_execute import _repo_commit

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no git today")

    monkeypatch.setattr(subprocess_module, "run", _raise)
    assert _repo_commit() == "unknown"


def test_admission_identity_and_control_branches(tmp_path: Path) -> None:
    no_identity = _recipe_payload(source_identity={})
    result = run(component_request_from_dict(no_identity), base=tmp_path)
    assert result.status == "failed" and "invalid_source_identity" in result.reason
    bad_control = _recipe_payload(control_conditions={"ped_speed_m_s": 0.0})
    result = run(component_request_from_dict(bad_control), base=tmp_path)
    assert result.status == "failed" and "invalid_control_conditions" in result.reason
    not_mapping = _recipe_payload(control_conditions=[])
    result = run(component_request_from_dict(not_mapping), base=tmp_path)
    assert result.status == "failed" and "corrupt_recipe" in result.reason
    bad_measurement = _recipe_payload(
        measurements=[{"name": "vibes", "units": "u", "expected_direction": "increase"}]
    )
    result = run(component_request_from_dict(bad_measurement), base=tmp_path)
    assert result.status == "unavailable" and "unsupported_measurement" in result.reason


def test_unknown_factor_candidate_is_unavailable(tmp_path: Path) -> None:
    payload = _recipe_payload(
        interventions=[{"intervention_id": "mystery", "factor": "teleport", "priority": 1}]
    )
    payload["config"]["intervention_parameters"] = {}
    payload["config"]["max_candidates"] = 1
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported intervention factor" in result.reason


def test_wall_budget_exhaustion_before_first_candidate(tmp_path: Path) -> None:
    request = _fixture_request(wall_timeout_s=1e-9)
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "wall_timeout" in result.reason


def test_per_execution_timeout_is_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import robot_sf.analysis_workbench.review_execute as review_execute_module

    def _timed_out(_job: Any, _timeout_s: float, **_kwargs: Any) -> dict[str, Any]:
        return {"outcome": "timeout", "error": "child exceeded 120s and was terminated"}

    monkeypatch.setattr(review_execute_module, "_run_owned_child", _timed_out)
    request = _fixture_request(max_candidates=1, max_executions=2)
    result = run(request, base=tmp_path)
    assert result.status == "partial"
    assert "per_execution_timeout" in result.reason
    assert result.artifacts == ()


def test_resume_ledger_error_branches(tmp_path: Path) -> None:
    out = tmp_path / "srev-22-smoke"
    out.mkdir()
    (out / "attempt-ledger.json").write_text("not json", encoding="utf-8")
    payload = _fixture_json("request.json")
    payload["config"] = _fixture_json("config.json")
    result = run(component_request_from_dict(copy.deepcopy(payload)), base=tmp_path, resume=True)
    assert result.status == "failed" and "ledger" in result.reason
    (out / "attempt-ledger.json").write_text(
        json.dumps(
            {
                "schema_version": "attempt-ledger.v1",
                "request_id": "other",
                "attempts": [],
                "executions_consumed": 0,
                "wall_elapsed_s": 0.0,
            }
        ),
        encoding="utf-8",
    )
    result = run(component_request_from_dict(copy.deepcopy(payload)), base=tmp_path, resume=True)
    assert result.status == "failed" and "mismatch" in result.reason


def test_resume_parser_limit_fails_closed(tmp_path: Path) -> None:
    out = tmp_path / "srev-22-smoke"
    out.mkdir()
    (out / "attempt-ledger.json").write_text('{"padding":' + "9" * 5001 + "}", encoding="utf-8")
    result = run(_fixture_request(), base=tmp_path, resume=True)
    assert result.status == "failed"
    assert "unreadable attempt ledger" in result.reason


def test_resume_continues_after_partial(tmp_path: Path) -> None:
    first = _fixture_request(max_executions=2)
    partial = run(first, base=tmp_path)
    assert partial.status == "partial"
    resumed = _fixture_request(max_executions=6)
    result = run(resumed, base=tmp_path, resume=True)
    assert result.status == "complete"
    ledger = json.loads(
        (tmp_path / "srev-22-smoke" / "attempt-ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["executions_consumed"] == 4


def test_control_fidelity_predicate() -> None:
    from robot_sf.analysis_workbench.review_execute import _Executor

    payload = _fixture_json("request.json")
    payload["config"] = _fixture_json("config.json")
    request = component_request_from_dict(payload)
    from robot_sf.analysis_workbench.review_execute import validate_execute_config

    config = validate_execute_config(request.config)
    executor = _Executor(request=request, config=config, recipe={}, output_dir=Path("."))
    ok, _ = executor._control_fidelity_ok({"ped_displacement_m": 1.0, "robot_displacement_m": 1.0})
    assert ok is True
    ok, reason = executor._control_fidelity_ok(
        {"ped_displacement_m": 0.0, "robot_displacement_m": 1.0}
    )
    assert ok is False and "pedestrian" in reason


def test_specs_match_except_predicate() -> None:
    from robot_sf.analysis_workbench.review_execute import _specs_match_except

    assert _specs_match_except({"a": 1, "b": 2}, {"a": 1, "b": 3}, "b") is True
    assert _specs_match_except({"a": 1}, {"a": 1, "b": 3}, "b") is False
    assert _specs_match_except({"a": 1, "b": 2}, {"a": 9, "b": 3}, "b") is False


def test_cli_rejects_invalid_config_without_execution(tmp_path: Path, capsys: Any) -> None:
    from robot_sf.analysis_workbench.review_execute import main

    request_path = tmp_path / "request.json"
    config_path = tmp_path / "config.json"
    request_path.write_text(json.dumps(_fixture_json("request.json")), encoding="utf-8")
    bad_config = _fixture_json("config.json")
    bad_config["unknown_key"] = True
    config_path.write_text(json.dumps(bad_config), encoding="utf-8")
    code = main(
        [
            "--input",
            str(request_path),
            "--config",
            str(config_path),
            "--output",
            "out",
            "--base",
            str(tmp_path),
        ]
    )
    assert code == 1
    invalid_config = json.loads(capsys.readouterr().out)
    assert invalid_config["schema_version"] == "component-result.v1"
    assert invalid_config["status"] == "failed"
    missing_code = main(
        ["--input", str(tmp_path / "missing.json"), "--output", "out", "--base", str(tmp_path)]
    )
    missing = json.loads(capsys.readouterr().out)
    assert missing_code == 1
    assert missing["schema_version"] == "component-result.v1"
    assert missing["status"] == "failed"


@pytest.mark.parametrize("parser_input", ["request", "config"])
def test_cli_parser_limit_emits_stable_failed_result(
    tmp_path: Path, capsys: Any, parser_input: str
) -> None:
    """Request and override config parser limits never leak a traceback."""
    from robot_sf.analysis_workbench.review_execute import main

    request_path = tmp_path / "request.json"
    config_path = tmp_path / "config.json"
    huge_integer = "9" * 5001
    request_text = json.dumps(_fixture_json("request.json"))
    if parser_input == "request":
        request_path.write_text(
            request_text.replace('"config": {}', f'"config": {{"seed": {huge_integer}}}'),
            encoding="utf-8",
        )
    else:
        request_path.write_text(request_text, encoding="utf-8")
        config_path.write_text(f'{{"seed": {huge_integer}}}', encoding="utf-8")

    arguments = [
        "--input",
        str(request_path),
        "--output",
        "parser-limit-output",
        "--base",
        str(tmp_path),
    ]
    if parser_input == "config":
        arguments.extend(["--config", str(config_path)])

    exit_code = main(arguments)
    captured = capsys.readouterr()
    printed = json.loads(captured.out)

    assert exit_code == 1
    assert captured.err == ""
    assert printed["schema_version"] == "component-result.v1"
    result = component_result_from_dict(printed)
    assert result.status == "failed"
    assert result.reason == f"invalid_input: {parser_input} JSON cannot be parsed safely"
    assert not (tmp_path / "parser-limit-output").exists()


def test_cli_stdout_is_a_component_result_v1_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    """The raw standalone CLI payload round-trips through the result contract."""
    import robot_sf.analysis_workbench.review_execute as review_execute_module

    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_fixture_json("request.json")), encoding="utf-8")

    def _fake_run(
        request: Any, *, base: Path | None = None, resume: bool = False
    ) -> ComponentResult:
        assert base == tmp_path
        assert resume is False
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="complete",
            diagnostics=({"status": "complete", "source": "test"},),
            provenance={"evidence_boundary": "diagnostic_only"},
        )

    monkeypatch.setattr(review_execute_module, "run", _fake_run)
    code = review_execute_module.main(
        [
            "--input",
            str(request_path),
            "--output",
            "cli-output",
            "--base",
            str(tmp_path),
        ]
    )

    raw_stdout = capsys.readouterr().out
    printed = json.loads(raw_stdout)
    parsed = component_result_from_dict(printed)

    assert code == 0
    assert printed["schema_version"] == "component-result.v1"
    assert parsed.status == "complete"
    assert parsed.request_id == "srev22-smoke"
    assert parsed.diagnostics == ({"status": "complete", "source": "test"},)
