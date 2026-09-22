"""Focused SREV-28 preview, lifecycle, provenance, and offline-browser tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from textwrap import dedent
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
    config.setdefault("session_token", "fixture-session-token")
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
    with pytest.raises(review_sessions.ReviewSessionError, match="not finite"):
        review_sessions._strict_loads("1e999")
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


def test_lifecycle_anchor_is_monotonic_and_state_path_is_contained(tmp_path: Path) -> None:
    request = _request(tmp_path, output="lifecycle-anchor")
    session_id = "manual-lifecycle-session"
    state = review_sessions._LifecycleState(2, "a" * 32, "complete")
    assert (
        review_sessions._remember_lifecycle_state(
            tmp_path,
            request,
            state,
            session_id=session_id,
            require_existing=True,
        )
        == "journal_lifecycle_anchor_unavailable: current process anchor is missing"
    )
    assert (
        review_sessions._remember_lifecycle_state(tmp_path, request, state, session_id=session_id)
        is None
    )
    assert (
        review_sessions._remember_lifecycle_state(
            tmp_path,
            request,
            review_sessions._LifecycleState(1, "b" * 32, "complete"),
            session_id=session_id,
        )
        == "journal_lifecycle_replay: lifecycle generation is stale"
    )
    assert (
        review_sessions._remember_lifecycle_state(
            tmp_path,
            request,
            review_sessions._LifecycleState(2, "b" * 32, "complete"),
            session_id=session_id,
        )
        == "journal_lifecycle_replay: lifecycle revision is stale"
    )
    assert (
        review_sessions._remember_lifecycle_state(
            tmp_path,
            request,
            review_sessions._LifecycleState(2, "a" * 32, "running"),
            session_id=session_id,
        )
        == "journal_lifecycle_replay: settled lifecycle cannot return to running"
    )
    assert (
        review_sessions._remember_lifecycle_state(
            tmp_path,
            request,
            review_sessions._LifecycleState(2, "a" * 32, "running"),
            session_id=session_id,
            allow_reopen=True,
        )
        is None
    )
    committed = review_sessions._new_lifecycle_state(
        tmp_path, request, session_id=session_id, commit=True
    )
    assert review_sessions._current_lifecycle_state(tmp_path, request, session_id) == committed
    newer = review_sessions._LifecycleState(committed.generation + 1, "c" * 32, "failed")
    review_sessions._set_lifecycle_status(
        tmp_path, request, newer, session_id=session_id, status="failed"
    )
    assert review_sessions._current_lifecycle_state(tmp_path, request, session_id) == newer
    review_sessions._set_lifecycle_status(
        tmp_path,
        request,
        review_sessions._LifecycleState(newer.generation, newer.revision, "complete"),
        session_id=session_id,
        status="complete",
    )
    assert review_sessions._current_lifecycle_state(tmp_path, request, session_id).status == (
        "complete"
    )
    with pytest.raises(review_sessions.ReviewSessionError, match="unavailable"):
        review_sessions._lifecycle_state_path(tmp_path / "missing", request, session_id)
    linked_base = tmp_path / "linked-base"
    linked_base.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(review_sessions.ReviewSessionError, match="not regular"):
        review_sessions._lifecycle_state_path(linked_base, request, session_id)


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


def test_native_and_injected_admission_shapes_are_validated_separately(tmp_path: Path) -> None:
    request = _request(tmp_path)
    proof = _proof(tmp_path, request)
    native = {
        "schema_version": "executor-admission.v1",
        "source_root": str(tmp_path),
        "receipt_reference": "receipts/source.json",
        "receipt_sha256": "a" * 64,
        "receipt_id": "source-receipt",
        "source": {
            "uri": request.sources[0].uri,
            "format": request.sources[0].format,
            "schema": request.sources[0].schema,
            "sha256": request.sources[0].sha256,
            "source_commit": request.sources[0].source_commit,
            "config_identity": request.sources[0].config_identity,
        },
        "preservation_destination": "external:fixture",
        "preservation_receipt_reference": "receipts/preservation.json",
        "preservation_receipt_sha256": "b" * 64,
        "config_identity": "fixture-config.v1",
        "evidence_boundary": review_sessions.EVIDENCE_BOUNDARY,
        "scientific_claim_allowed": False,
    }
    expected_request = loop._canonical_digest(
        loop._request_identity(review_sessions._loop_request(request))
    )
    expected_recipe = loop.experiment_recipe_canonical_digest(request.config["recipe"])
    assert (
        review_sessions._source_admission_integrity_error(
            native,
            expected_request_digest=expected_request,
            expected_recipe_digest=expected_recipe,
            base=tmp_path,
        )
        is None
    )
    missing_native = dict(native)
    del missing_native["receipt_sha256"]
    assert "native proof fields are missing" in (
        review_sessions._source_admission_integrity_error(
            missing_native,
            expected_request_digest=expected_request,
            expected_recipe_digest=expected_recipe,
            base=tmp_path,
        )
        or ""
    )
    assert (
        review_sessions._source_admission_integrity_error(
            proof,
            expected_request_digest=proof["request_digest"],
            expected_recipe_digest=proof["recipe_digest"],
            base=tmp_path,
        )
        is None
    )
    incomplete = dict(proof)
    del incomplete["recipe_digest"]
    assert "injected proof identity is incomplete" in (
        review_sessions._source_admission_integrity_error(
            incomplete,
            expected_request_digest=proof["request_digest"],
            expected_recipe_digest=proof["recipe_digest"],
            base=tmp_path,
        )
        or ""
    )


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


def test_complete_reads_require_token_mac_and_reject_coherent_rewrite(tmp_path: Path) -> None:
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
    seal = output / review_sessions.INTEGRITY_FILENAME
    assert seal.is_file()
    assert request.config["session_token"].encode() not in seal.read_bytes()
    assert review_sessions.progress(request, base=tmp_path)["status"] == "complete"

    journal_path = output / review_sessions.JOURNAL_FILENAME
    report_path = output / review_sessions.REPORT_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    treatment = journal["operations"][1]["result"]
    treatment["metrics"]["min_robot_ped_distance_m"] = 0.8
    treatment["metrics"]["ped_mean_speed_m_s"] = 0.8
    journal["outcomes"][0]["treatment"]["metrics"]["min_robot_ped_distance_m"] = 0.8
    journal["outcomes"][0]["treatment"]["metrics"]["ped_mean_speed_m_s"] = 0.8
    observation = loop._pair_telemetry(
        journal["outcomes"][0]["control"],
        journal["outcomes"][0]["treatment"],
        factor=journal["outcomes"][0]["factor"],
        measurement=request.config["recipe"]["measurements"][0],
        motion_epsilon=0.05,
    )
    journal["outcomes"][0]["outcome"] = observation["outcome"]
    journal["outcomes"][0]["reason"] = observation["reason"]
    journal["outcomes"][0]["negative"] = observation["outcome"] != "survived"
    journal["outcomes"][0]["verdict"] = observation["outcome"]
    report["operations"] = journal["operations"]
    report["outcomes"] = [
        loop.ExperimentLoop._canonical_report_outcome(item) for item in journal["outcomes"]
    ]
    report["negative_outcomes"] = [
        item for item in report["outcomes"] if item.get("negative") is True
    ]
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    forged_progress = review_sessions.progress(request, base=tmp_path)
    forged_navigation = review_sessions.result_navigation(request, base=tmp_path)
    assert forged_progress["status"] == "failed"
    assert "integrity_seal" in forged_progress["reason"]
    assert forged_navigation["status"] == "unavailable"
    assert "integrity_seal" in forged_navigation["reason"]

    rotated = replace(
        request,
        config={**request.config, "session_token": "rotated-session-token"},
    )
    assert review_sessions.progress(rotated, base=tmp_path)["status"] == "failed"
    missing = replace(
        request,
        config={key: value for key, value in request.config.items() if key != "session_token"},
    )
    assert review_sessions.progress(missing, base=tmp_path)["status"] == "failed"
    no_token_request = replace(missing, output_directory="no-token")
    no_token_result = review_sessions.run(
        no_token_request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=_proof(tmp_path, no_token_request),
    )
    assert no_token_result.status == "failed"
    assert "integrity_seal_required" in no_token_result.reason


def test_integrity_seal_failures_are_bounded_and_non_secret(tmp_path: Path) -> None:
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
    seal_path = output / review_sessions.INTEGRITY_FILENAME
    original = seal_path.read_bytes()

    seal_path.unlink()
    missing = review_sessions.progress(request, base=tmp_path)
    assert missing["status"] == "failed"
    assert "missing" in missing["reason"]

    seal_path.write_bytes(b"[]")
    malformed = review_sessions.progress(request, base=tmp_path)
    assert malformed["status"] == "failed"
    assert "seal" in malformed["reason"]

    seal = json.loads(original)
    seal["hmac_sha256"] = "0" * 64
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    wrong_mac = review_sessions.progress(request, base=tmp_path)
    assert wrong_mac["status"] == "failed"
    assert "MAC" in wrong_mac["reason"]

    seal = json.loads(original)
    seal["journal_sha256"] = "not-a-digest"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    malformed_digest = review_sessions.progress(request, base=tmp_path)
    assert malformed_digest["status"] == "failed"
    assert "sha256" in malformed_digest["reason"]

    seal = json.loads(original)
    seal["lifecycle_generation"] = "forged"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    malformed_lifecycle = review_sessions.progress(request, base=tmp_path)
    assert malformed_lifecycle["status"] == "failed"
    assert "lifecycle" in malformed_lifecycle["reason"]
    assert review_sessions.result_navigation(request, base=tmp_path)["status"] == "unavailable"

    seal_path.unlink()
    seal_path.symlink_to(output / review_sessions.JOURNAL_FILENAME)
    symlinked = review_sessions.progress(request, base=tmp_path)
    assert symlinked["status"] == "failed"
    assert "symlink" in symlinked["reason"]


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


def test_running_lease_tamper_is_bounded_while_owner_can_settle(tmp_path: Path) -> None:
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
    session_id = review_sessions._control_context(request)["session_id"]
    lease_path = review_sessions._lifecycle_state_path(tmp_path, request, str(session_id))
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease["hmac_sha256"] = "0" * 64
    lease_path.write_text(json.dumps(lease), encoding="utf-8")
    forged = session.progress()
    assert forged["status"] == "failed"
    assert "lifecycle_lease" in forged["reason"]
    stopped = session.stop()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert stopped.status == "cancelled"


def test_running_lease_shape_tamper_is_bounded(tmp_path: Path) -> None:
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
    session_id = review_sessions._control_context(request)["session_id"]
    lease_path = review_sessions._lifecycle_state_path(tmp_path, request, str(session_id))
    original = json.loads(lease_path.read_text(encoding="utf-8"))

    missing_field = dict(original)
    missing_field.pop("status")
    lease_path.write_text(json.dumps(missing_field), encoding="utf-8")
    assert "lease fields are malformed" in session.progress()["reason"]

    malformed_mac = dict(original, hmac_sha256="")
    lease_path.write_text(json.dumps(malformed_mac), encoding="utf-8")
    assert "MAC is malformed" in session.progress()["reason"]

    malformed_identity = dict(original, lifecycle_generation="forged")
    lease_path.write_text(json.dumps(malformed_identity), encoding="utf-8")
    assert "lifecycle identity is malformed" in session.progress()["reason"]

    stopped = session.stop()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert stopped.status == "cancelled"


def test_start_and_stop_race_serializes_the_inactive_owner_branch(tmp_path: Path) -> None:
    request = _request(tmp_path)
    executor = SlowExecutor(delay_s=0.05)
    session = review_sessions.ReviewSession(
        request,
        base=tmp_path,
        executor=executor,
        source_admission=_proof(tmp_path, request),
    )
    barrier = threading.Barrier(3)
    starts: list[Any] = []
    stops: list[Any] = []

    def start() -> None:
        barrier.wait()
        starts.append(session.start())

    def stop() -> None:
        barrier.wait()
        stops.append(session.stop())

    start_thread = threading.Thread(target=start)
    stop_thread = threading.Thread(target=stop)
    start_thread.start()
    stop_thread.start()
    barrier.wait()
    start_thread.join(timeout=3)
    stop_thread.join(timeout=3)
    assert not start_thread.is_alive() and not stop_thread.is_alive()
    assert len(starts) == len(stops) == 1
    assert all(
        "cannot resume: output directory does not exist" not in item.reason
        for item in starts + stops
    )
    assert {starts[0].status, stops[0].status} <= {"cancelled", "failed", "complete"}
    if starts[0].status == "complete":
        assert stops[0].status == "complete"
    else:
        assert "session_owner_active" in starts[0].reason or stops[0].status == "cancelled"


def test_distinct_controllers_do_not_replace_fresh_lifecycle_lease(tmp_path: Path) -> None:
    request = _request(tmp_path)
    barrier = threading.Barrier(3)
    results: list[Any] = []

    def start() -> None:
        controller = review_sessions.ReviewSession(
            request,
            base=tmp_path,
            executor=FakeExecutor(),
            source_admission=_proof(tmp_path, request),
        )
        barrier.wait()
        results.append(controller.start())

    first = threading.Thread(target=start)
    second = threading.Thread(target=start)
    first.start()
    second.start()
    barrier.wait()
    first.join(timeout=3)
    second.join(timeout=3)
    assert not first.is_alive() and not second.is_alive()
    assert sorted(item.status for item in results) == ["complete", "failed"]
    assert review_sessions.progress(request, base=tmp_path)["status"] == "complete"


def test_cross_controller_stop_reports_existing_owner_contract(tmp_path: Path) -> None:
    request = _request(tmp_path)
    executor = SlowExecutor()
    owner = review_sessions.ReviewSession(
        request,
        base=tmp_path,
        executor=executor,
        source_admission=_proof(tmp_path, request),
    )
    other = review_sessions.ReviewSession(
        request,
        base=tmp_path,
        executor=executor,
        source_admission=_proof(tmp_path, request),
    )
    worker = threading.Thread(target=owner.start)
    worker.start()
    assert executor.started.wait(timeout=2)
    foreign_stop = other.stop()
    assert foreign_stop.status == "failed"
    assert "session_lock_owned" in foreign_stop.reason
    assert worker.is_alive()
    owner_stop = owner.stop()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert owner_stop.status == "cancelled"


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


def test_deep_and_huge_durable_json_fail_closed_without_raising(tmp_path: Path) -> None:
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
    nested: Any = "forged"
    for _ in range(review_sessions._MAX_JSON_DEPTH + 20):
        nested = [nested]
    journal["untrusted_nested_value"] = nested
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    assert review_sessions.progress(request, base=tmp_path)["status"] == "failed"
    assert review_sessions.result_navigation(request, base=tmp_path)["status"] == "unavailable"
    assert review_sessions.preview(request, base=tmp_path)["status"] == "failed"

    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal.pop("untrusted_nested_value")
    journal["budget"]["wall_timeout_s"] = 10**100
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    huge = review_sessions.progress(request, base=tmp_path)
    assert huge["status"] == "failed"
    assert any(marker in huge["reason"] for marker in ("journal", "JSON integer"))


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
    progress = review_sessions.progress(request, base=tmp_path)
    assert progress["status"] == "failed"
    assert "integrity_seal" in progress["reason"]
    navigation = review_sessions.result_navigation(request, base=tmp_path)
    assert navigation["status"] == "unavailable"
    assert any(
        marker in navigation["reason"]
        for marker in ("report_identity_mismatch", "journal_integrity_seal_invalid")
    )


def test_cancelled_journal_cannot_be_reopened_without_running_lease(tmp_path: Path) -> None:
    request = _request(tmp_path)
    first_executor = FakeExecutor()
    cancelled = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=first_executor,
        source_admission=_proof(tmp_path, request),
        cancel=lambda: True,
    )
    assert cancelled.status == "cancelled"
    output = tmp_path / request.output_directory
    journal_path = output / review_sessions.JOURNAL_FILENAME
    report_path = output / review_sessions.REPORT_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    journal["status"] = "running"
    journal["stop_reason"] = ""
    report["status"] = "running"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    resumed_executor = FakeExecutor()
    resumed = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=resumed_executor,
        source_admission=_proof(tmp_path, request),
    )
    assert resumed.status == "failed"
    assert "lifecycle_lease" in resumed.reason
    assert resumed_executor.calls == []
    assert review_sessions.progress(request, base=tmp_path)["status"] == "failed"


def test_crash_running_lease_reconnects_through_srev24_recovery(tmp_path: Path) -> None:
    request = _request(tmp_path)
    proof = _proof(tmp_path, request)

    class CrashAfterDispatch(FakeExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.crash = True

        def execute(
            self,
            operation_id: str,
            candidate: dict[str, Any],
            kind: str,
            spec: dict[str, Any],
            attempt: int,
        ) -> dict[str, Any]:
            result = super().execute(operation_id, candidate, kind, spec, attempt)
            if self.crash:
                self.crash = False
                raise KeyboardInterrupt("simulated wrapper crash")
            return result

    executor = CrashAfterDispatch()
    with pytest.raises(KeyboardInterrupt):
        review_sessions.run(
            request,
            base=tmp_path,
            autonomous=True,
            executor=executor,
            source_admission=proof,
        )
    running = review_sessions.progress(request, base=tmp_path)
    assert running["status"] == "running"
    resumed = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=executor,
        source_admission=proof,
    )
    assert resumed.status == "complete", resumed.reason
    assert review_sessions.progress(request, base=tmp_path)["status"] == "complete"


def test_running_lease_replay_fails_closed_in_a_fresh_process(tmp_path: Path) -> None:
    request = _request(tmp_path)
    proof = _proof(tmp_path, request)
    output = tmp_path / request.output_directory
    saved_output = tmp_path / "saved-running"
    saved_lease = tmp_path / "saved-running.lease.json"
    captured = False

    def capture_running_then_cancel() -> bool:
        nonlocal captured
        lease_path = review_sessions._lifecycle_state_path(
            tmp_path, request, str(review_sessions._control_context(request)["session_id"])
        )
        journal_path = output / review_sessions.JOURNAL_FILENAME
        if not captured and journal_path.is_file() and lease_path.is_file():
            shutil.copytree(output, saved_output)
            saved_lease.write_bytes(lease_path.read_bytes())
            captured = True
        return captured

    first = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=proof,
        cancel=capture_running_then_cancel,
    )
    assert first.status == "cancelled"
    assert captured

    shutil.rmtree(output)
    newer = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=proof,
        cancel=lambda: True,
    )
    assert newer.status == "cancelled"
    shutil.rmtree(output)
    shutil.copytree(saved_output, output)
    (output / loop.SESSION_LOCK_FILENAME).unlink(missing_ok=True)
    session_id = str(review_sessions._control_context(request)["session_id"])
    lease_path = review_sessions._lifecycle_state_path(tmp_path, request, session_id)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_bytes(saved_lease.read_bytes())

    request_path = tmp_path / "replay-request.json"
    proof_path = tmp_path / "replay-proof.json"
    calls_path = tmp_path / "replay-calls.txt"
    request_document = {"schema_version": "component-request.v1", **asdict(request)}
    request_path.write_text(json.dumps(request_document), encoding="utf-8")
    proof_path.write_text(json.dumps(proof), encoding="utf-8")
    child_script = dedent(
        """
        import json
        import sys
        from pathlib import Path

        from robot_sf.analysis_workbench.review_contracts import component_request_from_dict
        from robot_sf.render import review_sessions

        base = Path(sys.argv[1])
        request = component_request_from_dict(json.loads(Path(sys.argv[2]).read_text()))
        proof = json.loads(Path(sys.argv[3]).read_text())
        calls_path = Path(sys.argv[4])

        class CountingExecutor:
            def __init__(self):
                self.results = {}

            def execute(self, operation_id, candidate, kind, spec, attempt):
                del candidate, kind, spec, attempt
                calls_path.write_text("dispatched", encoding="utf-8")
                result = {
                    "status": "ok",
                    "metrics": {
                        "ped_displacement_m": 1.0,
                        "robot_displacement_m": 1.0,
                        "ped_mean_speed_m_s": 1.0,
                        "min_robot_ped_distance_m": 1.0,
                        "robot_goal_reached": 1,
                        "ped_motion_onset_step": 1,
                    },
                    "mechanism_activated": True,
                }
                self.results[operation_id] = result
                return result

            def result_for(self, operation_id):
                return self.results.get(operation_id)

        progress = review_sessions.progress(request, base=base)
        result = review_sessions.run(
            request,
            base=base,
            autonomous=True,
            resume=True,
            executor=CountingExecutor(),
            source_admission=proof,
        )
        print(json.dumps({
            "progress_status": progress.get("status"),
            "progress_reason": progress.get("reason", ""),
            "result_status": result.status,
            "result_reason": result.reason,
            "calls": calls_path.exists(),
        }, sort_keys=True))
        """
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            child_script,
            str(tmp_path),
            str(request_path),
            str(proof_path),
            str(calls_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0, child.stderr
    child_payload = json.loads(child.stdout.strip().splitlines()[-1])
    assert child_payload["progress_status"] == "failed"
    assert "journal_lifecycle_anchor_unavailable" in child_payload["progress_reason"]
    assert child_payload["result_status"] == "failed"
    assert "journal_lifecycle_anchor_unavailable" in child_payload["result_reason"]
    assert child_payload["calls"] is False


def test_settled_anchor_rejects_replayed_running_lease(tmp_path: Path) -> None:
    request = _request(tmp_path)
    proof = _proof(tmp_path, request)
    executor = SlowExecutor()
    session = review_sessions.ReviewSession(
        request,
        base=tmp_path,
        executor=executor,
        source_admission=proof,
    )
    worker = threading.Thread(target=session.start)
    worker.start()
    assert executor.started.wait(timeout=2)
    session_id = review_sessions._control_context(request)["session_id"]
    lease_path = review_sessions._lifecycle_state_path(tmp_path, request, str(session_id))
    saved_lease = lease_path.read_bytes()
    stopped = session.stop()
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert stopped.status == "cancelled"

    output = tmp_path / request.output_directory
    journal_path = output / review_sessions.JOURNAL_FILENAME
    report_path = output / review_sessions.REPORT_FILENAME
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    journal["status"] = "running"
    journal["stop_reason"] = ""
    report["status"] = "running"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_bytes(saved_lease)
    resumed_executor = FakeExecutor()
    resumed = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=resumed_executor,
        source_admission=proof,
    )
    assert resumed.status == "failed"
    assert "lifecycle_replay" in resumed.reason
    assert resumed_executor.calls == []


def test_lifecycle_anchor_rejects_complete_rollback_after_new_cancellation(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    proof = _proof(tmp_path, request)
    completed = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=proof,
    )
    assert completed.status == "complete"
    output = tmp_path / request.output_directory
    saved = tmp_path / "saved-complete"
    shutil.copytree(output, saved)
    shutil.rmtree(output)

    cancelled = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        executor=FakeExecutor(),
        source_admission=proof,
        cancel=lambda: True,
    )
    assert cancelled.status == "cancelled"
    shutil.rmtree(output)
    shutil.copytree(saved, output)

    replayed_progress = review_sessions.progress(request, base=tmp_path)
    replayed_navigation = review_sessions.result_navigation(request, base=tmp_path)
    assert replayed_progress["status"] == "failed"
    assert "lifecycle_replay" in replayed_progress["reason"]
    assert replayed_navigation["status"] == "unavailable"
    replay_executor = FakeExecutor()
    replayed_resume = review_sessions.run(
        request,
        base=tmp_path,
        autonomous=True,
        resume=True,
        executor=replay_executor,
        source_admission=proof,
    )
    assert replayed_resume.status == "failed"
    assert "lifecycle_replay" in replayed_resume.reason
    assert replay_executor.calls == []


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


def test_unbound_session_ids_include_source_identity(tmp_path: Path) -> None:
    request = _request(tmp_path, output="unbound-one")
    unbound_config = {key: value for key, value in request.config.items() if key != "session_token"}
    first = replace(request, config=unbound_config)
    changed_source = replace(first.sources[0], source_commit="different-source-commit")
    second = replace(
        first,
        output_directory="unbound-two",
        sources=(changed_source,),
    )
    assert (
        review_sessions.preview(first, base=tmp_path)["session_id"]
        != review_sessions.preview(second, base=tmp_path)["session_id"]
    )


def test_report_provenance_boundary_fields_fail_closed(tmp_path: Path) -> None:
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
    report_path = output / review_sessions.REPORT_FILENAME
    original = json.loads(report_path.read_text(encoding="utf-8"))
    for key, value in (
        ("evidence_boundary", "benchmark"),
        ("benchmark_success", True),
        ("dependent_family_status", "shared"),
    ):
        mutated = deepcopy(original)
        mutated["provenance"][key] = value
        report_path.write_text(json.dumps(mutated), encoding="utf-8")
        navigation = review_sessions.result_navigation(request, base=tmp_path)
        assert navigation["status"] == "unavailable", key
    report_path.write_text(json.dumps(original), encoding="utf-8")


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
