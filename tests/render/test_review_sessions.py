"""Focused SREV-28 preview, lifecycle, provenance, and offline-browser tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import threading
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench import review_experiment_loop as loop
from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    component_descriptor_from_dict,
    component_request_from_dict,
)
from robot_sf.render import review_sessions

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "scenario_review" / "review_sessions"
)


def _request(
    tmp_path: Path,
    *,
    output: str = "session",
    required_capabilities: tuple[str, ...] = (),
    **config_overrides: Any,
) -> ComponentRequest:
    request = json.loads((FIXTURE_ROOT / "request.json").read_text(encoding="utf-8"))
    config = json.loads((FIXTURE_ROOT / "config.json").read_text(encoding="utf-8"))
    config.update(config_overrides)
    shutil.copyfile(FIXTURE_ROOT / "recipe.json", tmp_path / "recipe.json")
    request["config"] = config
    request["output_directory"] = output
    request["required_capabilities"] = list(required_capabilities)
    return component_request_from_dict(request, source="fixture-request.json")


def _proof(tmp_path: Path, request: ComponentRequest) -> dict[str, Any]:
    source_path = tmp_path / "recipe.json"
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    recipe = request.config["recipe"]
    source = request.sources[0]
    return {
        "status": "admitted",
        "receipt_id": "srev28-test-receipt",
        "source_root": str(tmp_path),
        "source": {
            "artifact_id": source.artifact_id,
            "uri": source.uri,
            "format": source.format,
            "sha256": digest,
        },
        "request_digest": loop._canonical_digest(loop._request_identity(request)),
        "recipe_digest": loop.experiment_recipe_canonical_digest(recipe),
        "evidence_boundary": "diagnostic_only",
        "scientific_claim_allowed": False,
        "dependent_family_status": "standalone_fixture_only",
    }


def _metrics(kind: str) -> dict[str, Any]:
    treatment = kind == "treatment"
    return {
        "ped_displacement_m": 1.0,
        "robot_displacement_m": 1.0,
        "ped_mean_speed_m_s": 1.2 if treatment else 1.0,
        "min_robot_ped_distance_m": 1.2 if treatment else 1.0,
        "robot_goal_reached": 1,
        "ped_motion_onset_step": 1,
    }


class FakeExecutor:
    """Idempotent local executor used only to exercise the adapter seam."""

    def __init__(self) -> None:  # noqa: D107
        self.calls: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {}

    def execute(
        self,
        operation_id: str,
        candidate: dict[str, Any],
        kind: str,
        spec: dict[str, Any],
        attempt: int,
    ) -> dict[str, Any]:
        del spec
        self.calls.append(
            {
                "operation_id": operation_id,
                "candidate": candidate["intervention_id"],
                "kind": kind,
                "attempt": attempt,
            }
        )
        result = {"status": "ok", "metrics": _metrics(kind), "mechanism_activated": True}
        self.results[operation_id] = result
        return result

    def result_for(self, operation_id: str) -> dict[str, Any] | None:
        return self.results.get(operation_id)


class SlowExecutor(FakeExecutor):
    """Executor that makes stop/journal settlement observable."""

    def __init__(self, delay_s: float = 0.15) -> None:  # noqa: D107
        super().__init__()
        self.delay_s = delay_s
        self.started = threading.Event()

    def execute(
        self,
        operation_id: str,
        candidate: dict[str, Any],
        kind: str,
        spec: dict[str, Any],
        attempt: int,
    ) -> dict[str, Any]:
        self.started.set()
        time.sleep(self.delay_s)
        return super().execute(operation_id, candidate, kind, spec, attempt)


def test_descriptor_is_contract_valid_and_explicitly_diagnostic() -> None:
    document = review_sessions.descriptor()
    descriptor = component_descriptor_from_dict(document)
    assert descriptor.component_id == review_sessions.COMPONENT_ID
    assert "review-session-preview.v1" in descriptor.output_types
    assert "bounded-experiment-session" in descriptor.required_capabilities
    assert review_sessions.EVIDENCE_BOUNDARY == "diagnostic_only"


def test_small_helpers_reject_unsafe_json_paths_and_versions(tmp_path: Path) -> None:
    control = review_sessions.SessionControl(
        "progress", "http://localhost:8000", "token", {"index": 1}
    )
    assert control.to_dict()["payload"] == {"index": 1}
    assert review_sessions._strict_loads('{"ok": true}') == {"ok": True}
    with pytest.raises(review_sessions.ReviewSessionError, match="non-finite"):
        review_sessions._strict_loads("NaN")
    with pytest.raises(review_sessions.ReviewSessionError, match="JSON"):
        review_sessions._strict_loads("{")
    with pytest.raises(review_sessions.ReviewSessionError, match="strict JSON"):
        review_sessions._json_bytes({"value": float("nan")})
    with pytest.raises(review_sessions.ReviewSessionError, match="strict JSON"):
        review_sessions._canonical_digest({"value": float("nan")})
    with pytest.raises(review_sessions.ReviewSessionError, match="relative"):
        review_sessions._resolve_output(tmp_path, "../escape", create=False)
    missing_base = tmp_path / "missing-base"
    with pytest.raises(review_sessions.ReviewSessionError, match="base"):
        review_sessions._resolve_output(missing_base, "out", create=False)
    assert review_sessions._version_compatible({"required_component_version": "1.4.0"})[0]
    assert not review_sessions._version_compatible({"required_component_version": 1})[0]
    assert not review_sessions._version_compatible({"required_component_version": "2.0.0"})[0]


def test_persisted_native_admission_is_only_a_revalidation_candidate() -> None:
    keys = (
        "schema_version",
        "source_root",
        "receipt_reference",
        "receipt_sha256",
        "preservation_destination",
        "preservation_receipt_reference",
        "preservation_receipt_sha256",
        "config_identity",
    )
    admission = {key: f"value-{key}" for key in keys}
    assert review_sessions._admission_config_from_journal(admission) == admission
    assert review_sessions._admission_config_from_journal({"status": "admitted"}) is None
    assert review_sessions._admission_config_from_journal(None) is None


def test_atomic_output_and_symlink_guards_are_fail_closed(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    digest = review_sessions._atomic_new_file(destination, b"one")
    assert digest == hashlib.sha256(b"one").hexdigest()
    with pytest.raises(review_sessions.ReviewSessionError, match="collision"):
        review_sessions._atomic_new_file(destination, b"two")
    assert review_sessions._atomic_new_file(destination, b"two", overwrite=True)
    temporary = tmp_path / ".other.json.tmp"
    temporary.write_bytes(b"leftover")
    with pytest.raises(review_sessions.ReviewSessionError, match="collision"):
        review_sessions._atomic_new_file(tmp_path / "other.json", b"two")
    link_target = tmp_path / "target.json"
    link_target.write_bytes(b"target")
    link = tmp_path / "link.json"
    link.symlink_to(link_target)
    with pytest.raises(review_sessions.ReviewSessionError, match="collision"):
        review_sessions._atomic_new_file(link, b"no")
    component = tmp_path / "component"
    component.symlink_to(link_target)
    with pytest.raises(review_sessions.ReviewSessionError, match="symlink"):
        review_sessions._resolve_output(tmp_path, "component/out", create=False)


def test_preview_reports_preservation_and_injected_admission_state(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        preservation={"destination": "external:test", "receipt_reference": "receipt.json"},
    )
    proof = _proof(tmp_path, request)
    document = review_sessions.preview(
        request,
        base=tmp_path,
        executor=FakeExecutor(),
        source_admission=proof,
    )
    assert document["source_admission"]["status"] == "provided"
    assert document["preservation"] == {
        "status": "configured",
        "destination": "external:test",
        "receipt_reference": "receipt.json",
    }
    forged = review_sessions.preview(
        _request(tmp_path, output="forged-preview"),
        base=tmp_path,
        executor=FakeExecutor(),
        source_admission={"status": "forged"},
    )
    assert forged["source_admission"]["status"] == "unavailable"
    invalid = review_sessions.preview(
        _request(tmp_path, output="invalid-preservation", preservation="unsafe"),
        base=tmp_path,
    )
    assert invalid["preservation"]["status"] == "invalid"
    with pytest.raises(review_sessions.ReviewSessionError, match="unknown loop"):
        review_sessions._loop_config({"recipe": {}, "unexpected": True})
    with pytest.raises(review_sessions.ReviewSessionError, match="loop"):
        review_sessions._loop_config({"loop": "not-a-mapping"})


def test_input_proof_and_native_admission_preview_fail_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)
    child = review_sessions._loop_request(request)
    recipe = request.config["recipe"]
    assert (
        review_sessions._source_proof_for_child(
            request, child, recipe, {"status": "admitted"}, base=tmp_path
        )
        is None
    )
    native_preview = review_sessions.preview(
        request,
        base=tmp_path,
        admission_config={},
    )
    assert native_preview["source_admission"]["status"] == "unavailable"
    assert "admission" in native_preview["source_admission"]["reason"]


def test_progress_and_navigation_handle_not_started_corrupt_and_missing_state(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    initial = review_sessions.progress(request, base=tmp_path)
    assert initial["status"] == "not_started"
    output = tmp_path / request.output_directory
    output.mkdir()
    (output / review_sessions.JOURNAL_FILENAME).write_text("{", encoding="utf-8")
    corrupt = review_sessions.progress(request, base=tmp_path)
    assert corrupt["status"] == "failed"
    missing = review_sessions.result_navigation(request, base=tmp_path)
    assert missing["status"] == "unavailable"
    assert (
        review_sessions.navigate_result(request, base=tmp_path, direction=1)["status"]
        == "unavailable"
    )


def test_control_envelope_and_origin_parser_reject_invalid_values(tmp_path: Path) -> None:
    request = _request(tmp_path)
    session_token = "token"
    with pytest.raises(review_sessions.ControlAuthorizationError, match="schema"):
        review_sessions.validate_control(
            {"schema_version": "v0", "action": "progress"},
            expected_origin="http://localhost:8000",
            expected_session_token=session_token,
        )
    with pytest.raises(review_sessions.ControlAuthorizationError, match="action"):
        review_sessions.validate_control(
            {"action": "shell", "origin": "http://localhost:8000", "session_token": "token"},
            expected_origin="http://localhost:8000",
            expected_session_token=session_token,
        )
    with pytest.raises(review_sessions.ControlAuthorizationError, match="required"):
        review_sessions.validate_control(
            {"action": "progress", "origin": "http://localhost:8000", "session_token": ""},
            expected_origin="http://localhost:8000",
            expected_session_token=session_token,
        )
    with pytest.raises(review_sessions.ControlAuthorizationError, match="mapping"):
        review_sessions.validate_control(
            object(), expected_origin="http://localhost:8000", expected_session_token=session_token
        )
    assert not review_sessions.is_loopback_origin("not a URL")
    assert not review_sessions.is_loopback_origin("http://localhost:8000/path")
    assert not review_sessions.is_loopback_origin("http://user@localhost:8000")
    assert request.request_id


def test_cli_read_only_and_malformed_inputs_are_truthful(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = FIXTURE_ROOT / "request.json"
    assert (
        review_sessions.main(
            [
                "--input",
                str(input_path),
                "--config",
                str(FIXTURE_ROOT / "config.json"),
                "--output",
                "read-only",
                "--base",
                str(tmp_path),
                "--read-only",
            ]
        )
        == 1
    )
    assert "read_only_never_executes" in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text("NaN", encoding="utf-8")
    assert (
        review_sessions.main(["--input", str(bad), "--output", "bad", "--base", str(tmp_path)]) == 1
    )
    assert "invalid_input" in capsys.readouterr().out
    bad_config = tmp_path / "bad-config.json"
    bad_config.write_text("[]", encoding="utf-8")
    assert (
        review_sessions.main(
            [
                "--input",
                str(input_path),
                "--config",
                str(bad_config),
                "--output",
                "bad-config",
                "--base",
                str(tmp_path),
            ]
        )
        == 1
    )
    assert "config must be an object" in capsys.readouterr().out


def test_cli_stop_requires_request_bound_origin_and_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.loads((FIXTURE_ROOT / "request.json").read_text(encoding="utf-8"))
    payload["config"] = {
        **json.loads((FIXTURE_ROOT / "config.json").read_text(encoding="utf-8")),
        "origin": "http://localhost:8000",
        "session_token": "expected-token",
    }
    input_path = tmp_path / "request.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    exit_code = review_sessions.main(
        [
            "--input",
            str(input_path),
            "--output",
            "session",
            "--base",
            str(tmp_path),
            "--stop",
            "--origin",
            "http://localhost:8000",
            "--session-token",
            "wrong-token",
        ]
    )
    assert exit_code == 1
    assert "session token mismatch" in capsys.readouterr().out


def test_cli_stop_requires_native_admission_after_authentication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    request = _request(
        tmp_path,
        origin="http://localhost:8000",
        session_token="expected-token",  # noqa: S106
    )
    completed = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, request),
    )
    assert completed.status == "complete"
    payload = json.loads((FIXTURE_ROOT / "request.json").read_text(encoding="utf-8"))
    payload["config"] = dict(request.config)
    input_path = tmp_path / "request.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    exit_code = review_sessions.main(
        [
            "--input",
            str(input_path),
            "--output",
            request.output_directory,
            "--base",
            str(tmp_path),
            "--stop",
            "--origin",
            "http://localhost:8000",
            "--session-token",
            "expected-token",
        ]
    )
    assert exit_code == 1
    assert "source_admission_required" in capsys.readouterr().out


def test_fixture_cli_preview_writes_budget_admission_and_browser_free_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = FIXTURE_ROOT / "request.json"
    config_path = FIXTURE_ROOT / "config.json"
    exit_code = review_sessions.main(
        [
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            "preview",
            "--base",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    document = json.loads((tmp_path / "preview" / review_sessions.PREVIEW_FILENAME).read_text())
    assert document["authorization"]["explicit_start_required"] is True
    assert document["budget"]["max_executions"] == 2
    assert document["budget"]["remaining_executions"] == 2
    assert document["source_admission"]["status"] == "required"
    assert document["preservation"]["status"] == "required"


def test_run_requires_explicit_start_and_read_only_never_dispatches(tmp_path: Path) -> None:
    request = _request(tmp_path)
    executor = FakeExecutor()
    proof = _proof(tmp_path, request)
    not_started = review_sessions.run(
        request,
        base=tmp_path,
        executor=executor,
        source_admission=proof,
    )
    read_only = review_sessions.run(
        request,
        base=tmp_path,
        read_only=True,
        executor=executor,
        source_admission=proof,
    )
    assert not_started.status == read_only.status == "unavailable"
    assert "never_executes" in read_only.reason
    assert "authorization_required" in not_started.reason
    assert executor.calls == []


def test_fake_executor_completion_delegates_journal_and_exposes_navigation(tmp_path: Path) -> None:
    request = _request(tmp_path)
    executor = FakeExecutor()
    result = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=executor,
        source_admission=_proof(tmp_path, request),
    )
    assert result.status == "complete", result.reason
    assert [item["kind"] for item in executor.calls] == ["control", "treatment"]
    assert (tmp_path / request.output_directory / review_sessions.JOURNAL_FILENAME).is_file()
    assert (tmp_path / request.output_directory / review_sessions.REPORT_FILENAME).is_file()
    assert (tmp_path / request.output_directory / review_sessions.HTML_FILENAME).is_file()
    assert (
        "fetch("
        not in (
            tmp_path
            / request.output_directory
            / "components"
            / "review_sessions"
            / review_sessions.ASSET_FILENAME
        ).read_text()
    )
    progress = review_sessions.progress(request, base=tmp_path)
    assert progress["status"] == "complete"
    assert progress["budget"]["executions_consumed"] == 2
    navigation = review_sessions.result_navigation(request, base=tmp_path)
    assert navigation["total"] == 1
    assert navigation["current"]["intervention_id"] == "speed-up"


def test_read_surfaces_bind_journal_to_request_and_current_source(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, request),
    )
    assert result.status == "complete"
    stale = replace(request, request_id="different-session")
    stale_progress = review_sessions.progress(stale, base=tmp_path)
    assert stale_progress["status"] == "failed"
    assert stale_progress["scientific_claim_allowed"] is False
    assert review_sessions.preview(stale, base=tmp_path)["status"] == "failed"
    assert review_sessions.result_navigation(stale, base=tmp_path)["status"] == "unavailable"
    (tmp_path / "recipe.json").write_text("tampered", encoding="utf-8")
    mutated = review_sessions.progress(request, base=tmp_path)
    assert mutated["status"] == "failed"
    assert "source_mutated" in mutated["reason"]


def test_nested_browser_asset_symlink_is_rejected(tmp_path: Path) -> None:
    request = _request(tmp_path)
    output = tmp_path / request.output_directory
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "components").symlink_to(outside, target_is_directory=True)
    with pytest.raises(review_sessions.ReviewSessionError, match="output_collision"):
        review_sessions._write_browser_view(
            output,
            request,
            {"schema_version": review_sessions.SESSION_VIEW_SCHEMA_VERSION, "status": "complete"},
        )
    assert not (outside / "review_sessions" / review_sessions.ASSET_FILENAME).exists()


def test_stop_waits_for_owned_work_and_terminal_journal(tmp_path: Path) -> None:
    request = _request(tmp_path)
    executor = SlowExecutor()
    session = review_sessions.ReviewSession(
        request,
        base=tmp_path,
        executor=executor,
        source_admission=_proof(tmp_path, request),
    )
    worker = threading.Thread(target=session.start)
    worker.start()
    assert executor.started.wait(timeout=2)
    started = time.monotonic()
    stopped = session.stop()
    elapsed = time.monotonic() - started
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert elapsed >= executor.delay_s
    assert stopped.status == "cancelled"
    assert session.progress()["status"] == "cancelled"


def test_concurrent_start_preserves_active_owner_for_stop(tmp_path: Path) -> None:
    request = _request(tmp_path)
    executor = SlowExecutor()
    session = review_sessions.ReviewSession(
        request,
        base=tmp_path,
        executor=executor,
        source_admission=_proof(tmp_path, request),
    )
    first_results: list[Any] = []
    second_results: list[Any] = []
    first = threading.Thread(target=lambda: first_results.append(session.start()))
    first.start()
    assert executor.started.wait(timeout=2)
    live_progress = session.progress()
    assert live_progress["status"] == "running", live_progress
    second = threading.Thread(target=lambda: second_results.append(session.start()))
    second.start()
    second.join(timeout=2)
    assert not second.is_alive()
    assert len(second_results) == 1
    assert second_results[0].status == "failed"
    assert "session_owner_active" in second_results[0].reason
    stopped = session.stop()
    first.join(timeout=2)
    assert not first.is_alive()
    assert len(first_results) == 1
    assert first_results[0].status == "cancelled"
    assert stopped.status == "cancelled"
    assert "session_lock_owned" not in stopped.reason
    assert session.progress()["status"] == "cancelled"


def test_tampered_or_malformed_journal_is_diagnostic_and_non_throwing(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, request),
    )
    assert result.status == "complete"
    journal_path = tmp_path / request.output_directory / review_sessions.JOURNAL_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["scientific_claim_allowed"] = True
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    forged = review_sessions.progress(request, base=tmp_path)
    assert forged["status"] == "failed"
    assert forged["scientific_claim_allowed"] is False
    assert review_sessions.preview(request, base=tmp_path)["scientific_claim_allowed"] is False
    journal["scientific_claim_allowed"] = False
    journal["executions_consumed"] = "bad"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    malformed = review_sessions.progress(request, base=tmp_path)
    assert malformed["status"] == "failed"
    assert malformed["scientific_claim_allowed"] is False


def test_persisted_integrity_variants_fail_closed_without_exposing_state(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, request),
    )
    assert result.status == "complete"
    journal_path = tmp_path / request.output_directory / review_sessions.JOURNAL_FILENAME
    original = json.loads(journal_path.read_text(encoding="utf-8"))
    mutations = [
        ("evidence_boundary", "benchmark"),
        ("dependent_family_status", "shared"),
        ("status", "bogus"),
        ("stop_reason", 1),
        ("source_admission", []),
        ("budget", {}),
        ("executions_consumed", "bad"),
        ("elapsed_s", "bad"),
        ("candidate_catalog", []),
        ("candidate_order", "bad"),
        ("candidates", []),
        ("outcomes", ["bad"]),
        ("operations", ["bad"]),
        ("policy", {}),
        ("config_identity_digest", "bad"),
        ("answerability", {"forged": True}),
        ("provenance", []),
    ]
    for key, value in mutations:
        mutated = deepcopy(original)
        mutated[key] = value
        journal_path.write_text(json.dumps(mutated), encoding="utf-8")
        document = review_sessions.progress(request, base=tmp_path)
        assert document["status"] == "failed", key
        assert document["scientific_claim_allowed"] is False
    journal_path.write_text(json.dumps(original), encoding="utf-8")
    assert (
        review_sessions._journal_progress({"budget": {"max_executions": "bad"}})["status"]
        == "failed"
    )
    assert (
        review_sessions._journal_progress({"status": "forged", "budget": {}})["status"] == "failed"
    )


def test_source_and_context_guards_reject_unsafe_values(tmp_path: Path) -> None:
    request = _request(tmp_path)
    recipe = request.config["recipe"]
    proof = _proof(tmp_path, request)
    assert review_sessions._safe_source_relative("../recipe.json") is None
    assert review_sessions._safe_source_relative("recipe\\.json") is None
    assert review_sessions._source_digest(tmp_path / "missing", Path("recipe.json")) is None
    malformed_root = dict(proof, source_root=[])  # type: ignore[arg-type]
    assert "source_root is malformed" in (
        review_sessions._source_integrity_error(tmp_path, request, recipe, malformed_root) or ""
    )
    malformed_source = dict(proof, source={"uri": "../escape"})
    assert "admission source differs" in (
        review_sessions._source_integrity_error(tmp_path, request, recipe, malformed_source) or ""
    )
    with pytest.raises(review_sessions.ReviewSessionError, match="session_context"):
        review_sessions.preview(
            _request(tmp_path, output="bad-context", session_context="forged"), base=tmp_path
        )


def test_tampered_report_is_not_navigable(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, request),
    )
    assert result.status == "complete"
    report_path = tmp_path / request.output_directory / review_sessions.REPORT_FILENAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["outcomes"][0]["status"] = "forged"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    navigation = review_sessions.result_navigation(request, base=tmp_path)
    assert navigation["status"] == "unavailable"
    assert "report_identity_mismatch" in navigation["reason"]


def test_semantic_journal_and_nested_admission_tampering_fail_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, request),
    )
    assert result.status == "complete"
    output = tmp_path / request.output_directory
    journal_path = output / review_sessions.JOURNAL_FILENAME
    report_path = output / review_sessions.REPORT_FILENAME
    original_journal = json.loads(journal_path.read_text(encoding="utf-8"))
    original_report = json.loads(report_path.read_text(encoding="utf-8"))

    forged_journal = deepcopy(original_journal)
    forged_journal["status"] = "running"
    forged_journal["stop_reason"] = "still-running"
    forged_journal["accounting"]["controls"] = 999
    forged_journal["outcomes"][0]["outcome"] = "falsified"
    forged_journal["outcomes"][0]["reason"] = "forged"
    forged_report = deepcopy(original_report)
    forged_report["status"] = "running"
    forged_report["budget"] = {
        **forged_journal["budget"],
        "executions_consumed": forged_journal["executions_consumed"],
        "reserved_executions": forged_journal["reserved_executions"],
        "elapsed_s": forged_journal["elapsed_s"],
    }
    forged_report["outcomes"] = [
        loop.ExperimentLoop._canonical_report_outcome(item) for item in forged_journal["outcomes"]
    ]
    forged_report["negative_outcomes"] = [
        item for item in forged_report["outcomes"] if item.get("negative") is True
    ]
    journal_path.write_text(json.dumps(forged_journal), encoding="utf-8")
    report_path.write_text(json.dumps(forged_report), encoding="utf-8")
    progress = review_sessions.progress(request, base=tmp_path)
    navigation = review_sessions.result_navigation(request, base=tmp_path)
    assert progress["status"] == "failed"
    assert progress["scientific_claim_allowed"] is False
    assert navigation["status"] == "unavailable"

    nested_journal = deepcopy(original_journal)
    nested_journal["source_admission"]["status"] = "forged"
    nested_journal["source_admission"]["request_digest"] = "forged"
    nested_journal["source_admission"]["evidence_boundary"] = "benchmark"
    nested_report = deepcopy(original_report)
    nested_report["source_admission"] = nested_journal["source_admission"]
    journal_path.write_text(json.dumps(nested_journal), encoding="utf-8")
    report_path.write_text(json.dumps(nested_report), encoding="utf-8")
    nested_progress = review_sessions.progress(request, base=tmp_path)
    nested_navigation = review_sessions.result_navigation(request, base=tmp_path)
    assert nested_progress["status"] == "failed"
    assert nested_progress["scientific_claim_allowed"] is False
    assert nested_navigation["status"] == "unavailable"


def test_canonical_journal_and_report_symlinks_are_not_read(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, request),
    )
    assert result.status == "complete"
    output = tmp_path / request.output_directory
    journal_path = output / review_sessions.JOURNAL_FILENAME
    report_path = output / review_sessions.REPORT_FILENAME
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_journal = outside / "journal.json"
    outside_report = outside / "report.json"
    shutil.copyfile(journal_path, outside_journal)
    shutil.copyfile(report_path, outside_report)

    journal_path.unlink()
    journal_path.symlink_to(outside_journal)
    progress = review_sessions.progress(request, base=tmp_path)
    assert progress["status"] == "failed"
    assert "symlink" in progress["reason"]
    journal_path.unlink()
    shutil.copyfile(outside_journal, journal_path)

    report_path.unlink()
    report_path.symlink_to(outside_report)
    navigation = review_sessions.result_navigation(request, base=tmp_path)
    assert navigation["status"] == "unavailable"
    assert "symlink" in navigation["reason"]


def test_cancelled_result_retains_journal_but_no_complete_artifacts(tmp_path: Path) -> None:
    request = _request(tmp_path)
    executor = FakeExecutor()
    result = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=executor,
        source_admission=_proof(tmp_path, request),
        cancel=lambda: True,
    )
    assert result.status == "cancelled"
    assert result.artifacts == ()
    assert (tmp_path / request.output_directory / review_sessions.JOURNAL_FILENAME).is_file()
    assert executor.calls == []


def test_collision_missing_capability_and_incompatible_version_fail_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)
    output = tmp_path / request.output_directory
    output.mkdir()
    collision = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, request),
    )
    missing = review_sessions.run(
        _request(tmp_path, output="missing", required_capabilities=("remote-scheduler",)),
        base=tmp_path,
    )
    incompatible = review_sessions.run(
        _request(tmp_path, output="version", required_component_version="2.0.0"),
        base=tmp_path,
    )
    assert collision.status == "failed" and "collision" in collision.reason
    assert missing.status == "unavailable" and "missing_capabilities" in missing.reason
    assert (
        incompatible.status == "unavailable"
        and "incompatible_component_version" in incompatible.reason
    )


def test_resume_keeps_consumed_budget_and_stop_uses_same_loop_owner(tmp_path: Path) -> None:
    request = _request(tmp_path)
    executor = FakeExecutor()
    session = review_sessions.ReviewSession(
        request,
        base=tmp_path,
        executor=executor,
        source_admission=_proof(tmp_path, request),
    )
    first = session.start()
    assert first.status == "complete"
    before = session.progress()["budget"]["executions_consumed"]
    resumed = session.resume()
    assert resumed.status == "complete"
    assert session.progress()["budget"]["executions_consumed"] == before
    stopped = session.stop()
    assert stopped.status == "complete"
    assert session.progress()["budget"]["executions_consumed"] == before


def test_loopback_control_requires_exact_origin_and_token(tmp_path: Path) -> None:
    request = _request(tmp_path)
    session = review_sessions.ReviewSession(request, base=tmp_path)
    handler = review_sessions.make_control_handler(
        session,
        origin="http://127.0.0.1:8765",
        session_token="test-token",  # noqa: S106
    )
    with pytest.raises(review_sessions.ControlAuthorizationError, match="origin"):
        handler(
            {
                "schema_version": review_sessions.CONTROL_SCHEMA_VERSION,
                "action": "progress",
                "origin": "http://example.com",
                "session_token": "test-token",
            }
        )
    with pytest.raises(review_sessions.ControlAuthorizationError, match="mismatch"):
        handler(
            {
                "schema_version": review_sessions.CONTROL_SCHEMA_VERSION,
                "action": "progress",
                "origin": "http://127.0.0.1:8765",
                "session_token": "wrong",
            }
        )
    assert review_sessions.is_loopback_origin("http://localhost:8000")
    assert not review_sessions.is_loopback_origin("http://127.0.0.1:8000/path")


def test_context_revision_binds_session_identity(tmp_path: Path) -> None:
    first = _request(
        tmp_path,
        output="context-session",
        session_context={
            "campaign_id": "campaign-1",
            "episode_id": "episode-1",
            "source_revision": "source-1",
            "selection_revision": "selection-1",
            "context_revision": "context-1",
        },
    )
    second = _request(
        tmp_path,
        output="context-session-2",
        session_context={
            "campaign_id": "campaign-1",
            "episode_id": "episode-1",
            "source_revision": "source-1",
            "selection_revision": "selection-1",
            "context_revision": "context-2",
        },
    )
    first_preview = review_sessions.preview(first, base=tmp_path)
    second_preview = review_sessions.preview(second, base=tmp_path)
    assert first_preview["context"]["context_revision"] == "context-1"
    assert second_preview["context"]["context_revision"] == "context-2"
    assert first_preview["session_id"] != second_preview["session_id"]
    token_first = _request(
        tmp_path,
        output="token-session-1",
        origin="http://localhost:8000",
        session_token="token-one",  # noqa: S106
    )
    token_second = _request(
        tmp_path,
        output="token-session-2",
        origin="http://localhost:8000",
        session_token="token-two",  # noqa: S106
    )
    assert (
        review_sessions.preview(token_first, base=tmp_path)["session_id"]
        != review_sessions.preview(token_second, base=tmp_path)["session_id"]
    )


def test_control_binding_includes_recipe_and_source_identity(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        output="bound-session",
        origin="http://localhost:8000",
        session_token="same-token",  # noqa: S106
        session_context={"context_revision": "same-context"},
    )
    changed_recipe = deepcopy(request.config["recipe"])
    changed_recipe["hypothesis"] = "different diagnostic hypothesis"
    recipe_request = _request(
        tmp_path,
        output="bound-recipe",
        origin="http://localhost:8000",
        session_token="same-token",  # noqa: S106
        session_context={"context_revision": "same-context"},
        recipe=changed_recipe,
    )
    changed_source = replace(request.sources[0], source_commit="different-source-commit")
    source_request = replace(request, sources=(changed_source,), output_directory="bound-source")
    request_context = review_sessions._control_context(request)
    recipe_context = review_sessions._control_context(recipe_request)
    source_context = review_sessions._control_context(source_request)
    assert request_context["recipe_digest"] != recipe_context["recipe_digest"]
    assert request_context["request_digest"] != source_context["request_digest"]
    assert (
        review_sessions.preview(request, base=tmp_path)["session_id"]
        != review_sessions.preview(recipe_request, base=tmp_path)["session_id"]
    )
    assert (
        review_sessions.preview(request, base=tmp_path)["session_id"]
        != review_sessions.preview(source_request, base=tmp_path)["session_id"]
    )

    handler = review_sessions.make_control_handler(
        review_sessions.ReviewSession(request, base=tmp_path),
        origin="http://localhost:8000",
        session_token="same-token",  # noqa: S106
    )
    with pytest.raises(review_sessions.ControlAuthorizationError, match="request digest"):
        handler(
            {
                "action": "progress",
                "origin": "http://localhost:8000",
                "session_token": "same-token",
                "session_id": request_context["session_id"],
                "context_revision": request_context["context_revision"],
                "request_digest": "forged-request",
                "recipe_digest": request_context["recipe_digest"],
            }
        )


def test_node_browser_runtime_is_offline_and_has_no_implicit_start() -> None:
    asset = (
        Path(__file__).resolve().parents[2]
        / "robot_sf"
        / "render"
        / "web_assets"
        / "components"
        / "review_sessions"
        / "review_sessions.js"
    )
    text = asset.read_text(encoding="utf-8")
    assert "fetch(" not in text
    assert "WebSocket" not in text
    assert ".start();" not in text
    if shutil.which("node") is None:
        pytest.skip("node is unavailable")
    script = f"""
      import {{ isLoopbackOrigin, ReviewSessionsController }} from {json.dumps(asset.as_uri())};
      if (!isLoopbackOrigin('http://127.0.0.1:8765')) throw new Error('loopback');
      if (isLoopbackOrigin('http://user@localhost:8765')) throw new Error('credential origin');
      if (isLoopbackOrigin('http://localhost:8765/')) throw new Error('path origin');
      let calls = [];
      const controller = new ReviewSessionsController({{origin: 'http://127.0.0.1:8765', sessionToken: 't', sessionId: 'session-1', requestDigest: 'request-digest', recipeDigest: 'recipe-digest', readOnly: false, controlRequest: (value) => {{ calls.push(value); return {{status: 'running'}}; }}}});
      await controller.start();
      if (calls.length !== 1 || calls[0].action !== 'start' || calls[0].request_digest !== 'request-digest' || calls[0].recipe_digest !== 'recipe-digest') throw new Error('explicit control');
    """
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0


def test_remaining_boundary_guards_and_nested_preview_paths(tmp_path: Path) -> None:
    with pytest.raises(review_sessions.ReviewSessionError, match="input exceeds"):
        review_sessions._strict_loads(b"x" * (review_sessions._MAX_JSON_BYTES + 1))
    with pytest.raises(review_sessions.ReviewSessionError, match="input exceeds"):
        review_sessions._strict_loads("x" * (review_sessions._MAX_JSON_BYTES + 1))
    with pytest.raises(review_sessions.ReviewSessionError, match="relative"):
        review_sessions._resolve_output(tmp_path, "", create=False)
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("file", encoding="utf-8")
    request = _request(tmp_path, output="not-a-directory")
    assert review_sessions.preview(request, base=tmp_path)["output"]["status"] == "invalid"
    with pytest.raises(review_sessions.ReviewSessionError, match="invalid_input"):
        review_sessions._normalise_request({"invalid": True})
    nested = _request(tmp_path, output="nested", loop={"recipe": request.config["recipe"]})
    assert review_sessions.preview(nested, base=tmp_path)["budget"]["max_candidates"] == 1
    assert (
        review_sessions._admission_preview(
            request,
            config=request.config,
            admission_config=None,
            executor=FakeExecutor(),
            source_admission=None,
            base=tmp_path,
        )["status"]
        == "unavailable"
    )
    assert review_sessions._preservation_preview({"preservation": []}, None)["status"] == "invalid"


def test_navigation_control_and_run_error_statuses(tmp_path: Path) -> None:
    request = _request(tmp_path)
    token = "token"
    proof = _proof(tmp_path, request)
    fake = FakeExecutor()
    complete = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=fake,
        source_admission=proof,
    )
    assert complete.status == "complete"
    assert (
        review_sessions.result_navigation(request, base=tmp_path, index="bad")["status"] == "failed"  # type: ignore[arg-type]
    )
    assert (
        review_sessions.navigate_result(request, base=tmp_path, index=0, direction=1)["index"] == 0
    )
    assert (
        review_sessions.validate_control(
            review_sessions.SessionControl("progress", "http://localhost:8000", token),
            expected_origin="http://localhost:8000",
            expected_session_token=token,
        ).action
        == "progress"
    )
    with pytest.raises(review_sessions.ControlAuthorizationError, match="origin mismatch"):
        review_sessions.validate_control(
            {"action": "progress", "origin": "http://localhost:8000", "session_token": "token"},
            expected_origin="http://localhost:8000:1",
            expected_session_token=token,
        )
    handler = review_sessions.make_control_handler(
        review_sessions.ReviewSession(request, base=tmp_path),
        origin="http://localhost:8000",
        session_token=token,
    )
    context = review_sessions._control_context(request)
    assert (
        handler(
            {
                "action": "progress",
                "origin": "http://localhost:8000",
                "session_token": "token",
                "session_id": context["session_id"],
                "request_digest": context["request_digest"],
                "recipe_digest": context["recipe_digest"],
            }
        )["status"]
        == "complete"
    )
    assert review_sessions.run({"bad": True}).status == "failed"
    wrong = review_sessions.run(replace(request, component_id="srev99-other"), base=tmp_path)
    assert wrong.status == "unavailable"
    no_proof = review_sessions.run(
        _request(tmp_path, output="no-proof"),
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
    )
    assert no_proof.status == "unavailable"
