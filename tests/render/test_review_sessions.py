"""Focused SREV-28 preview, lifecycle, provenance, and offline-browser tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
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
      let calls = [];
      const controller = new ReviewSessionsController({{origin: 'http://127.0.0.1:8765', sessionToken: 't', readOnly: false, controlRequest: (value) => {{ calls.push(value); return {{status: 'running'}}; }}}});
      await controller.start();
      if (calls.length !== 1 || calls[0].action !== 'start') throw new Error('explicit control');
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
        == "required"
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
    assert (
        handler(
            {"action": "progress", "origin": "http://localhost:8000", "session_token": "token"}
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
