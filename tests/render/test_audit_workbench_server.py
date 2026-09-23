"""Exact-origin transport checks for the server-held BA-06 route."""

from __future__ import annotations

import json
import socket
import threading
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pytest

from robot_sf.analysis_workbench.audit_contracts import Annotation
from robot_sf.analysis_workbench.audit_service import AuditSelectionContext
from robot_sf.analysis_workbench.audit_store import StoredRecord
from robot_sf.analysis_workbench.review_contracts import SourceRef
from robot_sf.render.audit_workbench import ServiceAuditWorkbenchFacade
from robot_sf.render.audit_workbench_server import (
    _dispatch_facade_operation,
    _materialization_source_alias,
    _sanitize_codex_reference,
    _sanitize_codex_result,
    _sanitize_codex_text,
    _sanitize_materialization_result,
    _sanitize_native_diagnostic_result,
    _validate_sync_finding_arguments,
    make_audit_workbench_server,
)
from robot_sf.render.review_workbench import live_audit_document


def _document() -> dict:
    return {
        "time_base": {"authority": "simulation", "video_alignment": "unavailable"},
        "episodes": [],
        "artifacts": [],
        "presentation": {},
        "extension_slots": ["panels"],
        "extensions": {"audit_workbench": {"slot": "panels", "mode": "diagnostic_fixture"}},
    }


@contextmanager
def _server(
    secret_override: str | None = None,
    *,
    service: Any | None = None,
    session: Any | None = None,
    configure=None,
    materialization_output_root=None,
):
    secret = (
        secret_override
        if secret_override is not None
        else f"private-audit-token-{uuid.uuid4().hex}"
    )
    facade = ServiceAuditWorkbenchFacade(
        object() if service is None else service,
        SimpleNamespace(context={"campaign_id": "retained"}) if session is None else session,
        token=secret,
    )
    calls: list[tuple[str, dict]] = []
    if service is None:
        facade.snapshot = lambda: {"status": "complete", "selection_revision": 0}  # type: ignore[method-assign]

    def select_next(**kwargs):
        calls.append(("next", kwargs))
        return {
            "status": "complete",
            "presentation_status": "selected",
            "selection_revision": 1,
            "packet": {"episode_id": "retained-episode"},
            "reason": f"{secret} must be redacted",
        }

    facade.next = select_next  # type: ignore[method-assign]

    def persist_finding(**kwargs):
        calls.append(("persist_finding", kwargs))
        return {"status": "unavailable", "reason": "test dispatch"}

    facade.persist_finding = persist_finding  # type: ignore[method-assign]
    if configure is not None:
        configure(facade, calls, secret)
    document = live_audit_document(_document(), facade)
    server = make_audit_workbench_server(
        document, facade, materialization_output_root=materialization_output_root
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(f"{url}/", timeout=5) as response:
            cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        yield url, calls, document, secret, cookie
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(url: str, payload: dict, *, origin: str, cookie: str | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json", "Origin": origin}
    if cookie is not None:
        headers["Cookie"] = cookie
    request = Request(
        f"{url}/api/audit",
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def test_materialization_action_uses_only_server_output_root_and_curated_result(tmp_path) -> None:
    output_root = tmp_path / "trusted-output"

    def configure(facade, calls, secret):
        facade._selected_episode_id = "case-1"
        facade._selection_epoch = 3
        facade._context = {"campaign_id": "retained", "episode_id": "case-1", "context_revision": 4}

        def materialize_selected(**kwargs):
            calls.append(("materialize_selected", kwargs))
            return {
                "status": "complete",
                "reason": f"hidden {secret} /private/source",
                "value": {
                    "status": "complete",
                    "episode_id": "case-1",
                    "materialization_kind": "historical_original",
                    "fidelity": "verified",
                    "diagnostic_only": True,
                    "output_directory": "/private/output",
                    "artifacts": [{"uri": "file:///private/output/trace.jsonl"}],
                },
            }

        facade.materialize_selected = materialize_selected

    with _server(configure=configure, materialization_output_root=output_root) as (
        url,
        calls,
        _document,
        secret,
        cookie,
    ):
        arguments = {
            "operation_id": "materialize-1",
            "expected_selection_revision": 3,
            "expected_context_revision": 4,
        }
        status, body = _post(
            url,
            {"operation": "materialize_selected", "arguments": arguments},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body == {
            "status": "complete",
            "classification": "historical_original",
            "fidelity": "verified",
            "diagnostic_only": True,
            "scientific_claim_allowed": False,
        }
        assert calls == [
            (
                "materialize_selected",
                {
                    "output_root": output_root,
                    "context": {
                        "campaign_id": "retained",
                        "episode_id": "case-1",
                        "context_revision": 4,
                    },
                    "operation_id": "materialize-1",
                },
            )
        ]
        assert secret not in json.dumps(body)
        for bad_arguments in (
            {**arguments, "output_root": "/private/elsewhere"},
            {**arguments, "expected_context_revision": 3},
            {**arguments, "expected_selection_revision": 2},
        ):
            bad_status, bad_body = _post(
                url,
                {"operation": "materialize_selected", "arguments": bad_arguments},
                origin=url,
                cookie=cookie,
            )
            assert bad_status in {400, 409}
            assert bad_body["status"] in {"failed", "conflict"}
        assert len(calls) == 1


def test_materialization_action_is_unavailable_without_server_capability() -> None:
    with _server() as (url, calls, _document, _secret, cookie):
        status, body = _post(
            url,
            {
                "operation": "materialize_selected",
                "arguments": {
                    "operation_id": "materialize-1",
                    "expected_selection_revision": 0,
                    "expected_context_revision": 0,
                },
            },
            origin=url,
            cookie=cookie,
        )
        assert status == 501
        assert body["status"] == "unavailable"
        assert calls == []


def test_materialization_projection_rejects_foreign_and_malformed_results() -> None:
    valid = {
        "status": "complete",
        "value": {
            "status": "complete",
            "episode_id": "other-case",
            "materialization_kind": "historical_original",
            "fidelity": "verified",
            "diagnostic_only": True,
            "output_directory": "/private/path",
        },
    }
    foreign = _sanitize_materialization_result(valid, episode_id="selected-case")
    assert foreign["status"] == "conflict"
    assert "/private/path" not in json.dumps(foreign)
    malformed = _sanitize_materialization_result({"status": []}, episode_id="selected-case")
    assert malformed["status"] == "failed"
    valid["value"]["episode_id"] = "selected-case"
    valid["value"]["fidelity"] = []
    bad_fidelity = _sanitize_materialization_result(valid, episode_id="selected-case")
    assert bad_fidelity["status"] == "failed"
    assert "/private/path" not in json.dumps(bad_fidelity)
    valid["value"].update(fidelity="verified", status="failed")
    failed = _sanitize_materialization_result(valid, episode_id="selected-case")
    assert failed["status"] == "failed"
    assert "classification" not in failed
    valid["value"].update(status="complete", materialization_kind="unavailable")
    unavailable = _sanitize_materialization_result(valid, episode_id="selected-case")
    assert unavailable["status"] == "unavailable"
    assert "classification" not in unavailable


def test_materialization_row_alias_requires_current_selected_read() -> None:
    context = {"episode_id": "selected-ref", "context_revision": 2}
    episode_result = SimpleNamespace(
        status="complete",
        context=context,
        value=SimpleNamespace(row={"episode_id": "literal-row"}),
    )
    facade = SimpleNamespace(
        _context=context,
        _last_episode=episode_result,
        _context_matches_request=lambda expected, actual: expected == actual,
    )
    assert _materialization_source_alias(facade, selected_episode_id="selected-ref") == (
        "literal-row"
    )
    assert _materialization_source_alias(facade, selected_episode_id="foreign-ref") is None
    facade._context = {"episode_id": "selected-ref", "context_revision": 3}
    assert _materialization_source_alias(facade, selected_episode_id="selected-ref") is None


def test_native_diagnostic_action_keeps_source_and_runner_authority_server_only() -> None:
    def configure(facade, calls, secret):
        facade._service = SimpleNamespace(
            native_diagnostic_config=SimpleNamespace(bindings=(object(),))
        )
        facade._selected_episode_id = "selected-ref"
        facade._selection_epoch = 3
        facade._context = {
            "campaign_id": "retained",
            "episode_id": "selected-ref",
            "context_revision": 4,
        }

        def run_native_diagnostic(**kwargs):
            calls.append(("run_native_diagnostic", kwargs))
            return {
                "status": "complete",
                "reason": f"hidden {secret} /private/runner",
                "value": {
                    "status": "complete",
                    "evidence_boundary": "diagnostic_only",
                    "scientific_claim_allowed": False,
                    "original_identity": {"episode_id": "selected-ref", "uri": "/private/source"},
                    "fidelity": {"status": "verified", "private": "/private/fidelity"},
                    "activation": {
                        "status": "verified",
                        "input_changed": True,
                        "trace_changed": True,
                        "private": "/private/trace",
                    },
                    "provenance": {"source_path": "/private/source"},
                },
            }

        facade.run_native_diagnostic = run_native_diagnostic

    with _server(configure=configure) as (url, calls, _document, secret, cookie):
        arguments = {
            "operation_id": "native-action-1",
            "expected_selection_revision": 3,
            "expected_context_revision": 4,
            "intervention_id": "goal-change",
            "robot_goal": [4.0, 4.0],
            "activation_epsilon_m": 1e-9,
            "deadline_s": 30.0,
        }
        status, body = _post(
            url,
            {"operation": "run_native_diagnostic", "arguments": arguments},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body == {
            "status": "complete",
            "diagnostic_only": True,
            "scientific_claim_allowed": False,
            "control_fidelity": "verified",
            "activation": "verified",
        }
        assert calls == [
            (
                "run_native_diagnostic",
                {
                    "intervention_id": "goal-change",
                    "robot_goal": [4.0, 4.0],
                    "activation_epsilon_m": 1e-9,
                    "deadline_s": 30.0,
                    "context": {
                        "campaign_id": "retained",
                        "episode_id": "selected-ref",
                        "context_revision": 4,
                    },
                    "operation_id": "native-action-1",
                },
            )
        ]
        assert secret not in json.dumps(body)
        assert "/private" not in json.dumps(body)
        for invalid in (
            {**arguments, "expected_context_revision": 3},
            {**arguments, "expected_selection_revision": 2},
            {**arguments, "robot_goal": [True, 4.0]},
            {**arguments, "robot_goal": [10**400, 4.0]},
            {**arguments, "deadline_s": 61.0},
            {**arguments, "source_root": "/private/source"},
        ):
            invalid_status, _ = _post(
                url,
                {"operation": "run_native_diagnostic", "arguments": invalid},
                origin=url,
                cookie=cookie,
            )
            assert invalid_status in {400, 403, 409}
        assert len(calls) == 1


def test_native_diagnostic_action_requires_configuration_and_rejects_foreign_result() -> None:
    with _server() as (url, calls, _document, _secret, cookie):
        status, body = _post(
            url,
            {
                "operation": "run_native_diagnostic",
                "arguments": {
                    "operation_id": "native-action-1",
                    "expected_selection_revision": 0,
                    "expected_context_revision": 0,
                    "intervention_id": "goal-change",
                    "robot_goal": [4.0, 4.0],
                    "activation_epsilon_m": 0.0,
                    "deadline_s": 30.0,
                },
            },
            origin=url,
            cookie=cookie,
        )
        assert status == 501
        assert body["status"] == "unavailable"
        assert calls == []
    valid = {
        "status": "complete",
        "value": {
            "status": "complete",
            "evidence_boundary": "diagnostic_only",
            "scientific_claim_allowed": False,
            "original_identity": {"episode_id": "foreign"},
            "fidelity": {"status": "verified"},
            "activation": {"status": "verified", "input_changed": True, "trace_changed": True},
        },
    }
    assert _sanitize_native_diagnostic_result(valid, episode_id="selected")["status"] == (
        "conflict"
    )
    valid["value"]["original_identity"]["episode_id"] = "selected"
    valid["value"]["activation"]["trace_changed"] = False
    assert _sanitize_native_diagnostic_result(valid, episode_id="selected")["status"] == ("failed")


def test_live_route_keeps_token_server_only_and_dispatches_exact_origin() -> None:
    with _server() as (url, calls, document, secret, cookie):
        with urlopen(f"{url}/", timeout=5) as response:
            html = response.read().decode()
            set_cookie = response.headers["Set-Cookie"]
        assert "createServiceFacade()" in html
        assert "createFixtureFacade(data)" not in html
        assert secret not in html
        assert "HttpOnly" in set_cookie
        assert "SameSite=Strict" in set_cookie
        assert secret not in cookie
        assert cookie.startswith("audit_workbench_session=")
        assert document["extensions"]["audit_workbench"]["mode"] == "service_live"
        status, body = _post(
            url,
            {"operation": "next", "arguments": {"expected_selection_revision": 0}},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body["packet"]["episode_id"] == "retained-episode"
        assert body["status"] == "complete"
        assert secret not in json.dumps(body)
        assert calls == [("next", {"expected_selection_revision": 0})]


def test_live_route_redacts_typed_source_ref_uri_from_snapshot() -> None:
    """HTTP snapshot transport cannot re-expose local or encoded SourceRef URIs."""

    secret = "http-private-audit-token"
    unsafe_uris = (
        "/private/allowed/campaign.json",
        " /private/allowed/campaign.json",
        "%2Fprivate%2Fallowed%2Fcampaign.json",
        "https://example.invalid/campaign.json%3Ftoken=unbound-secret",
        "https://example.invalid/%68%74%74%70%2D%70%72%69%76%61%74%65%2D%61%75%64%69%74%2D%74%6F%6B%65%6E",
    )
    context = AuditSelectionContext(
        campaign_id="retained",
        episode_id="retained-episode",
        source_identity="source-digest",
        source_revision=0,
    )
    stored = tuple(
        StoredRecord(
            record_id=f"annotation-http-uri-{index}",
            record_type="annotation",
            revision=1,
            deleted=False,
            operation_id=f"annotation-http-uri-op-{index}",
            committed_at=f"2026-09-20T00:00:{index + 1:02d}Z",
            global_revision=index + 1,
            record=Annotation(
                annotation_id=f"annotation-http-uri-{index}",
                episode_id="retained-episode",
                classification="unclear",
                author_kind="agent",
                source_identity="source-digest",
                source_revision=0,
                source_ref=SourceRef(
                    artifact_id="campaign",
                    uri=uri,
                    format="campaign-json",
                ),
            ),
        )
        for index, uri in enumerate(unsafe_uris)
    )

    class SnapshotService:
        def __init__(self) -> None:
            self.context = context

        def _result(self, value: Any) -> Any:
            return SimpleNamespace(
                status="complete", value=value, reason="", operation=None, context=self.context
            )

        def read_context(self, _session: Any, **_kwargs: Any) -> Any:
            return self._result(self.context)

        def read_queue(self, _session: Any, **_kwargs: Any) -> Any:
            return self._result({"items": [], "state_revision": 1, "input_revision": 1})

        def read_coverage(self, _session: Any, **_kwargs: Any) -> Any:
            return self._result({"reviewed": 0, "remaining": 1})

        def read_episode(self, _session: Any, _episode_id: str, **_kwargs: Any) -> Any:
            return self._result({"episode_id": "retained-episode", "status": "readable"})

        def read_saved_records(self, _session: Any, _episode_id: str, **_kwargs: Any) -> Any:
            return self._result(stored)

    session = SimpleNamespace(
        session_id="session-http-uri",
        session_token=secret,
        actor=SimpleNamespace(actor_id="agent-http-uri"),
        context=context,
    )
    with _server(secret, service=SnapshotService(), session=session) as (
        url,
        _calls,
        _document,
        _,
        cookie,
    ):
        status, body = _post(
            url,
            {"operation": "snapshot", "arguments": {}},
            origin=url,
            cookie=cookie,
        )

    assert status == 200
    payload = json.dumps(body, sort_keys=True)
    assert all(uri not in payload for uri in unsafe_uris)
    assert secret not in payload
    assert all(
        annotation["source_ref"]["uri"] == "<redacted-local-uri>"
        for annotation in body["annotations"]
    )


def test_live_route_forwards_finding_service_cas_contract() -> None:
    with _server() as (url, calls, _document, _secret, cookie):
        arguments = {
            "annotation": {"annotation_id": "annotation-http"},
            "expected_selection_revision": 1,
            "expected_context": {"context_revision": 4, "episode_id": "episode-http"},
            "expected_context_revision": 4,
            "expected_source_revision": "source-r4",
            "expected_queue_state_revision": 17,
            "expected_queue_input_revision": 23,
            "expected_queue_input_identity": "queue-input-23",
            "operation_id": "finding-http",
            "finding_id": "finding-http",
            "title": "review finding",
        }
        status, body = _post(
            url,
            {"operation": "persist_finding", "arguments": arguments},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body["status"] == "unavailable"
        assert calls[-1] == ("persist_finding", arguments)


def _sync_arguments(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "finding_id": "finding-http",
        "repository": "ll7/robot_sf_ll7",
        "expected_finding_revision": 3,
        "expected_selection_revision": 1,
        "expected_context_revision": 4,
        "expected_source_revision": "source-r4",
        "retry_ambiguous": False,
        "operation_id": "github-sync-http",
    }
    arguments.update(overrides)
    return arguments


def test_sync_finding_validation_and_dispatch_keep_the_route_closed() -> None:
    valid = _sync_arguments()

    class Facade:
        def sync_finding(self, **kwargs: Any) -> dict[str, Any]:
            return {"status": "accepted", "arguments": kwargs}

    assert _dispatch_facade_operation(Facade(), "sync_finding", valid)["status"] == "accepted"
    invalid_cases = (
        {"extra": True},
        {"finding_id": "bad/finding"},
        {"repository": "not-a-repository"},
        {"expected_finding_revision": -1},
        {"expected_source_revision": True},
        {"expected_source_revision": -1},
        {"expected_source_revision": ""},
        {"operation_id": "not valid"},
        {"retry_ambiguous": "false"},
    )
    for overrides in invalid_cases:
        with pytest.raises(ValueError):
            _validate_sync_finding_arguments({**valid, **overrides})


def test_live_route_sync_finding_is_closed_and_redacts_provider_authority() -> None:
    secret = "sync-private-token"

    class SyncService:
        context = SimpleNamespace(
            campaign_id="retained",
            episode_id="retained-episode",
            context_revision=4,
            source_revision="source-r4",
            source_identity="source-digest-r4",
        )

        def sync_finding(self, session: Any, **kwargs: Any) -> Any:
            del session
            return SimpleNamespace(
                status="committed",
                reason="",
                value={
                    "status": "created",
                    "finding_id": kwargs["finding_id"],
                    "repository": kwargs["repository"],
                    "source_path": "/private/source.json",
                    "token": secret,
                },
                operation=None,
                context=self.context,
            )

    service = SyncService()

    def configure(facade, _calls, _secret):
        facade._selected_episode_id = "retained-episode"
        facade._selection_epoch = 1
        facade._context = service.context
        facade._last_finding = {"finding_id": "finding-http", "revision": 3}

    with _server(secret, service=service, configure=configure) as (
        url,
        _calls,
        _document,
        _,
        cookie,
    ):
        status, body = _post(
            url,
            {"operation": "sync_finding", "arguments": _sync_arguments()},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body["status"] == "committed"
        assert body["value"]["status"] == "created"
        payload = json.dumps(body, sort_keys=True)
        assert secret not in payload
        assert "/private/source.json" not in payload

        forbidden = _sync_arguments(provider="client-provider")
        assert (
            _post(
                url,
                {"operation": "sync_finding", "arguments": forbidden},
                origin=url,
                cookie=cookie,
            )[0]
            == 403
        )


def test_live_route_sync_finding_rejects_stale_selection_or_context() -> None:
    service = SimpleNamespace(context=SimpleNamespace(context_revision=4))

    def configure(facade, _calls, _secret):
        facade._selected_episode_id = "retained-episode"
        facade._selection_epoch = 2
        facade._context = service.context
        facade._last_finding = {"finding_id": "finding-http", "revision": 3}

    with _server(service=service, configure=configure) as (url, _calls, _document, _, cookie):
        status, body = _post(
            url,
            {
                "operation": "sync_finding",
                "arguments": _sync_arguments(expected_selection_revision=1),
            },
            origin=url,
            cookie=cookie,
        )
        assert status == 409
        assert body["status"] == "conflict"


def test_live_route_sync_finding_reports_provider_unavailable_without_fallback() -> None:
    def configure(facade, _calls, _secret):
        facade._selected_episode_id = "retained-episode"
        facade._selection_epoch = 1
        facade._context = SimpleNamespace(
            context_revision=4,
            source_revision="source-r4",
            source_identity="source-digest-r4",
            episode_id="retained-episode",
        )
        facade._last_finding = {"finding_id": "finding-http", "revision": 3}

    with _server(configure=configure) as (url, _calls, _document, _, cookie):
        status, body = _post(
            url,
            {"operation": "sync_finding", "arguments": _sync_arguments()},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body["status"] == "unavailable"
        assert "sync_finding" in body["reason"]


def test_live_route_rejects_serialized_slash_secret_in_retained_document() -> None:
    """The final HTML form of ``</...>`` must never contain the server token."""
    secret = "</retained-private-token>"
    facade = ServiceAuditWorkbenchFacade(
        object(), SimpleNamespace(context={"campaign_id": "retained"}), token=secret
    )
    document = _document()
    document["artifacts"] = [{"artifact_id": "retained", "detail": secret}]
    live_document = live_audit_document(document, facade)

    with pytest.raises(ValueError, match="credential leaked"):
        make_audit_workbench_server(live_document, facade)


def test_live_route_rejects_cross_origin_and_client_authority_fields() -> None:
    with _server() as (url, calls, _document, _secret, cookie):
        payload = {"operation": "next", "arguments": {"expected_selection_revision": 0}}
        assert _post(url, payload, origin=url)[0] == 403
        assert _post(url, payload, origin=url, cookie="audit_workbench_session=forged")[0] == 403
        assert _post(url, payload, origin=url, cookie=f"{cookie}; {cookie}")[0] == 403
        assert _post(url, payload, origin="http://127.0.0.1:1", cookie=cookie)[0] == 403
        assert _post(url, payload, origin="https://attacker.invalid", cookie=cookie)[0] == 403
        assert (
            _post(url, {**payload, "arguments": {"token": "forged"}}, origin=url, cookie=cookie)[0]
            == 403
        )
        assert (
            _post(url, {**payload, "arguments": {"context": {}}}, origin=url, cookie=cookie)[0]
            == 400
        )
        assert calls == []


def test_live_route_serves_only_declared_static_assets() -> None:
    with _server() as (url, _calls, _document, _secret, _cookie):
        with urlopen(f"{url}/components/audit_workbench/audit_workbench.js", timeout=5) as response:
            script = response.read().decode()
        assert "createServiceFacade" in script
        try:
            urlopen(f"{url}/../../.git/config", timeout=5)
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError("path escape unexpectedly served")


def test_live_route_rejects_dns_rebinding_host() -> None:
    with _server() as (url, calls, _document, _secret, _cookie):
        request = Request(
            f"{url}/",
            headers={"Host": "attacker.invalid"},
        )
        try:
            urlopen(request, timeout=5)
        except HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("cross-host request unexpectedly served")
        assert calls == []


def test_non_ascii_audit_token_is_redacted_after_json_escaping() -> None:
    with _server("ž-private-token") as (url, _calls, _document, secret, cookie):
        status, body = _post(
            url,
            {"operation": "next", "arguments": {"expected_selection_revision": 0}},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert secret not in json.dumps(body, ensure_ascii=False)
        assert "\\u017e-private-token" not in json.dumps(body)


def test_deep_json_request_fails_with_bounded_response() -> None:
    with _server() as (url, _calls, _document, _secret, cookie):
        nested = "[" * 1500 + "0" + "]" * 1500
        body = f'{{"operation":"save_annotation","arguments":{{"annotation":{nested}}}}}'.encode()
        request = Request(
            f"{url}/api/audit",
            data=body,
            headers={"Content-Type": "application/json", "Origin": url, "Cookie": cookie},
            method="POST",
        )
        try:
            urlopen(request, timeout=5)
        except HTTPError as error:
            assert error.code == 403
            assert json.load(error)["status"] == "denied"
        else:
            raise AssertionError("deep JSON unexpectedly admitted")


def test_codex_projection_redacts_encoded_text_and_path_like_references() -> None:
    encoded_secret = "%73%65%72%76%65%72%2D%73%65%63%72%65%74"
    unc_path = r"\\server\share\provider.json"
    relative_path = "relative provider/secret.json"

    assert "server-secret" not in _sanitize_codex_text(f"encoded {encoded_secret}")
    assert encoded_secret not in _sanitize_codex_text(f"encoded {encoded_secret}")
    assert _sanitize_codex_text(unc_path) == "<path redacted>"
    assert _sanitize_codex_text(relative_path) == "relative <path redacted>"
    for mixed_secret in ("server%2Dsecret", "server-%73ecret", "%73erver-secret"):
        assert _sanitize_codex_text(mixed_secret) == "<encoded redacted>"
    for single_component_path in ("/srv", "/opt", "/root"):
        assert _sanitize_codex_text(single_component_path) == "<path redacted>"
    assert _sanitize_codex_reference(encoded_secret) is None
    assert _sanitize_codex_reference(unc_path) is None
    assert _sanitize_codex_reference(relative_path) is None

    for safe_reference in ("route-local", "evidence-1", "provider:gpt-5", "source-7"):
        assert _sanitize_codex_reference(safe_reference) == safe_reference

    projected = _sanitize_codex_result(
        {
            "status": "complete",
            "route_id": encoded_secret,
            "evidence_ids": [encoded_secret, relative_path, "evidence-1"],
            "activity": [
                {"message": f"encoded {encoded_secret}"},
                {"message": unc_path},
                {"message": relative_path},
            ],
        }
    )
    projected_json = json.dumps(projected)
    assert encoded_secret not in projected_json
    assert "server-secret" not in projected_json
    assert unc_path not in projected_json
    assert relative_path not in projected_json
    assert projected["evidence_ids"] == ["evidence-1"]


def test_live_route_redacts_encoded_codex_token_and_provider_paths() -> None:
    secret = "server-secret"
    encoded_secret = "%73%65%72%76%65%72%2D%73%65%63%72%65%74"
    unc_path = r"\\server\share\provider.json"
    relative_path = "relative provider/secret.json"
    mixed_secrets = ("server%2Dsecret", "server-%73ecret", "%73erver-secret")
    single_component_paths = ("/srv", "/opt", "/root")

    def configure(facade, _calls, _secret):
        facade.codex_start = lambda **_kwargs: {
            "status": "complete",
            "route_id": encoded_secret,
            "evidence_ids": [encoded_secret, relative_path, "evidence-1"],
            "activity": [
                {"message": f"encoded {encoded_secret}"},
                {"message": unc_path},
                {"message": relative_path},
                *({"message": value} for value in mixed_secrets),
                *({"message": value} for value in single_component_paths),
            ],
        }

    with _server(secret, configure=configure) as (url, _calls, _document, _secret, cookie):
        status, body = _post(
            url,
            {
                "operation": "codex_start",
                "arguments": {
                    "prompt": "inspect",
                    "operation_id": "codex-path-regression",
                    "token_budget": 128,
                    "compute_budget": 1.0,
                },
            },
            origin=url,
            cookie=cookie,
        )

    assert status == 200
    payload = json.dumps(body, sort_keys=True)
    for unsafe in (
        secret,
        encoded_secret,
        unc_path,
        relative_path,
        *mixed_secrets,
        *single_component_paths,
    ):
        assert unsafe not in payload
    assert "route_id" not in body
    assert body["evidence_ids"] == ["evidence-1"]
    assert [event["message"] for event in body["activity"]] == [
        "<encoded redacted>",
        "<path redacted>",
        "relative <path redacted>",
        *("<encoded redacted>" for _ in mixed_secrets),
        *("<path redacted>" for _ in single_component_paths),
    ]


def test_live_route_dispatches_only_bounded_codex_facade_methods() -> None:
    def configure(facade, calls, secret):
        def codex_start(**kwargs):
            calls.append(("codex_start", kwargs))
            return {
                "status": "complete",
                "operation_id": kwargs["operation_id"],
                "context": {
                    "context_revision": 7,
                    "episode_id": "episode-codex",
                    "scenario_id": "scenario-codex",
                    "execution_id": "execution-codex",
                },
                "source": {"source_revision": "source-7", "source_digest": "d" * 64},
                "route_id": "route-local",
                "provider_path": "/private/provider",
                "sourcePath": "/private/source-camel",
                "providerPath": "/private/provider-camel",
                "source_root": "/private/source-root",
                "provider_root": "/private/provider-root",
                "sessionToken": "returned-session-token",
                "client_secret": "private-client-secret",
                "authorization_header": "Bearer private-auth",
                "routeId": "private-route-camel",
                "evidence_ids": ["evidence-1"],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 9,
                    "reserved_compute": 1.0,
                    "provider_path": "/private/provider",
                },
                "activity": [{"message": f"{secret} is redacted", "evidence_ids": ["evidence-1"]}],
            }

        def codex_read(**kwargs):
            calls.append(("codex_read", kwargs))
            return {"status": "complete", "operation_id": kwargs.get("operation_id")}

        def codex_cancel(**kwargs):
            calls.append(("codex_cancel", kwargs))
            return {"status": "cancelled", "operation_id": kwargs["operation_id"]}

        def codex_reconnect(**kwargs):
            calls.append(("codex_reconnect", kwargs))
            return {
                "status": "complete",
                "operation_id": kwargs["operation_id"],
                "codex_session_id": kwargs["codex_session_id"],
                "context": {
                    "context_revision": kwargs["expected_context_revision"],
                    "episode_id": "episode-codex",
                },
                "source": {
                    "source_revision": "source-7",
                    "source_digest": "d" * 64,
                },
                "provider_session_id": "private-provider-session",
            }

        facade.codex_start = codex_start
        facade.codex_read = codex_read
        facade.codex_cancel = codex_cancel
        facade.codex_reconnect = codex_reconnect

    with _server(configure=configure) as (url, calls, _document, secret, cookie):
        start_arguments = {
            "prompt": "inspect this selected case",
            "operation_id": "codex-test-1",
            "token_budget": 2048,
            "compute_budget": 1.0,
        }
        status, body = _post(
            url,
            {"operation": "codex_start", "arguments": start_arguments},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body["status"] == "complete"
        assert body["operation_id"] == "codex-test-1"
        assert body["route_id"] == "route-local"
        assert body["context"]["context_revision"] == 7
        assert body["context"]["episode_id"] == "episode-codex"
        assert body["context"]["scenario_id"] == "scenario-codex"
        assert body["context"]["execution_id"] == "execution-codex"
        assert body["source"]["source_revision"] == "source-7"
        assert secret not in json.dumps(body)
        body_json = json.dumps(body)
        for forbidden in (
            "/private/provider",
            "/private/source-camel",
            "/private/provider-camel",
            "/private/source-root",
            "/private/provider-root",
            "returned-session-token",
            "private-client-secret",
            "Bearer private-auth",
            "private-route-camel",
        ):
            assert forbidden not in body_json
        for forbidden_key in (
            "provider_path",
            "sourcePath",
            "providerPath",
            "source_root",
            "provider_root",
            "sessionToken",
            "client_secret",
            "authorization_header",
            "routeId",
        ):
            assert forbidden_key not in body
        assert calls[-1] == ("codex_start", start_arguments)

        status, body = _post(
            url,
            {"operation": "codex_read", "arguments": {"operation_id": "codex-test-1"}},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body == {"operation_id": "codex-test-1", "status": "complete"}
        reconnect_arguments = {
            "codex_session_id": "codex-session-private",
            "operation_id": "codex-reconnect-1",
            "expected_selection_revision": 1,
            "expected_context_revision": 7,
        }
        status, body = _post(
            url,
            {"operation": "codex_reconnect", "arguments": reconnect_arguments},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body == {
            "codex_session_id": "codex-session-private",
            "context": {"context_revision": 7, "episode_id": "episode-codex"},
            "operation_id": "codex-reconnect-1",
            "source": {"source_digest": "d" * 64, "source_revision": "source-7"},
            "status": "complete",
        }
        assert calls[-1] == ("codex_reconnect", reconnect_arguments)
        status, body = _post(
            url,
            {
                "operation": "codex_cancel",
                "arguments": {"reason": "post-turn review", "operation_id": "codex-test-1"},
            },
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body == {"operation_id": "codex-test-1", "status": "cancelled"}
        for bad_arguments in (
            {**reconnect_arguments, "codex_session_id": "provider/session"},
            {**reconnect_arguments, "session_id": "forged"},
            {**reconnect_arguments, "expected_context_revision": -1},
        ):
            bad_status, bad_body = _post(
                url,
                {"operation": "codex_reconnect", "arguments": bad_arguments},
                origin=url,
                cookie=cookie,
            )
            assert bad_status in {400, 403}
            assert bad_body.get("status") in {"failed", "denied"}
        assert (
            _post(
                url,
                {"operation": "arbitrary", "arguments": {}},
                origin=url,
                cookie=cookie,
            )[0]
            == 400
        )


def test_live_route_fails_closed_for_absent_codex_capability() -> None:
    with _server() as (url, calls, _document, _secret, cookie):
        status, body = _post(
            url,
            {"operation": "codex_read", "arguments": {}},
            origin=url,
            cookie=cookie,
        )
        assert status == 501
        assert body == {"reason": "Codex capability is unavailable", "status": "unavailable"}
        assert calls == []


def test_live_route_bounds_malformed_codex_result_projection() -> None:
    def configure(facade, _calls, _secret):
        def codex_start(**kwargs):
            prompt = kwargs["prompt"]
            if prompt == "deep":
                result = {"status": "complete", "client_secret": "deep-secret"}
                for _ in range(1100):
                    result = {"result": result}
                return result
            if prompt == "cycle":
                result = {"status": "complete", "authorization_header": "cycle-secret"}
                result["result"] = result
                return result
            return {
                "status": "complete",
                "operation_id": kwargs["operation_id"],
                "activity": [{"message": "bounded activity"} for _ in range(1000)],
            }

        facade.codex_start = codex_start

    with _server(configure=configure) as (url, _calls, _document, _secret, cookie):
        for prompt, secret in (("deep", "deep-secret"), ("cycle", "cycle-secret")):
            status, body = _post(
                url,
                {
                    "operation": "codex_start",
                    "arguments": {
                        "prompt": prompt,
                        "operation_id": f"codex-{prompt}",
                        "token_budget": 128,
                        "compute_budget": 1.0,
                    },
                },
                origin=url,
                cookie=cookie,
            )
            assert status == 500
            assert body == {"status": "failed"}
            assert secret not in json.dumps(body)

        status, body = _post(
            url,
            {
                "operation": "codex_start",
                "arguments": {
                    "prompt": "fanout",
                    "operation_id": "codex-fanout",
                    "token_budget": 128,
                    "compute_budget": 1.0,
                },
            },
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body["status"] == "complete"
        assert len(body["activity"]) == 64


def test_live_route_rejects_non_string_codex_operation_with_bounded_json() -> None:
    with _server() as (url, calls, _document, _secret, cookie):
        for operation in ([], {}, 1, None, True):
            status, body = _post(
                url,
                {"operation": operation, "arguments": {}},
                origin=url,
                cookie=cookie,
            )
            assert status == 400
            assert body == {"status": "failed"}
        assert calls == []


def test_live_route_rejects_codex_authority_paths_and_unbounded_arguments() -> None:
    def configure(facade, _calls, _secret):
        facade.codex_start = lambda **kwargs: {"status": "complete", **kwargs}
        facade.codex_read = lambda **kwargs: {"status": "complete", **kwargs}
        facade.codex_cancel = lambda **kwargs: {"status": "cancelled", **kwargs}

    with _server(configure=configure) as (url, calls, _document, _secret, cookie):
        base = {
            "operation": "codex_start",
            "arguments": {
                "prompt": "inspect",
                "operation_id": "codex-test-2",
                "token_budget": 128,
                "compute_budget": 1.0,
            },
        }
        assert (
            _post(
                url,
                {**base, "arguments": {**base["arguments"], "source_path": "/tmp/secret"}},
                origin=url,
                cookie=cookie,
            )[0]
            == 403
        )
        assert (
            _post(
                url,
                {**base, "arguments": {**base["arguments"], "prompt": "x" * 8193}},
                origin=url,
                cookie=cookie,
            )[0]
            == 400
        )
        assert (
            _post(
                url,
                {**base, "arguments": {**base["arguments"], "operation_id": "/tmp/codex"}},
                origin=url,
                cookie=cookie,
            )[0]
            == 400
        )
        assert (
            _post(
                url,
                {"operation": "codex_read", "arguments": {"session_id": "forged"}},
                origin=url,
                cookie=cookie,
            )[0]
            == 403
        )
        assert (
            _post(
                url,
                {"operation": "codex_cancel", "arguments": {"operation_id": "codex-test-2"}},
                origin=url,
                cookie=cookie,
            )[0]
            == 400
        )
        assert calls == []


def test_live_route_dispatches_server_held_related_cases_as_unconfirmed_candidates() -> None:
    class RelatedService:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self.response: dict[str, Any] = {
                "status": "complete",
                "value": [
                    {
                        "query_id": "source-row-episode",
                        "candidate_id": "peer-episode",
                        "mode": "same_scenario_across_planners",
                        "score": 0.75,
                        "compatibility": "compatible",
                        "reasons": ["same scenario", "/private/provider.json"],
                        "features": {
                            "planner_id": "peer-planner",
                            "confirmed_members": ["must-not-cross"],
                            "source_path": "/private/source.json",
                        },
                        "confirmed_members": ["must-not-cross"],
                    }
                ],
            }

        def related_cases(self, session: Any, episode_id: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append({"session": session, "episode_id": episode_id, **kwargs})
            return self.response

    service = RelatedService()

    def configure(facade, _calls, _secret):
        facade._selected_episode_id = "episode-generated-123"
        facade._context = SimpleNamespace(context_revision=4, episode_id="episode-generated-123")
        facade._last_episode = SimpleNamespace(
            value=SimpleNamespace(row={"episode_id": "source-row-episode"})
        )
        facade.snapshot = lambda: {"status": "complete"}  # type: ignore[method-assign]

    with _server(service=service, configure=configure) as (
        url,
        _calls,
        _document,
        server_secret,
        cookie,
    ):
        status, body = _post(
            url,
            {
                "operation": "related_cases",
                "arguments": {
                    "episode_id": "episode-generated-123",
                    "mode": "same_scenario_across_planners",
                    "limit": 4,
                    "expected_context_revision": 4,
                    "operation_id": "related-http-1",
                },
            },
            origin=url,
            cookie=cookie,
        )

    assert status == 200
    assert body["status"] == "complete"
    assert body["candidate_count"] == 1
    assert body["value"][0]["query_id"] == "episode-generated-123"
    assert body["value"][0]["candidate_id"] == "peer-episode"
    assert body["value"][0]["features"] == {"planner_id": "peer-planner"}
    assert "confirmed_members" not in body["value"][0]
    assert body["membership_boundary"] == "candidates_are_unconfirmed"
    assert body["claim_boundary"] == "retrieval_only_not_benchmark_evidence"
    assert "/private" not in json.dumps(body)
    assert service.calls[0]["episode_id"] == "episode-generated-123"
    assert service.calls[0]["token"] == server_secret


def test_live_route_related_cases_fails_closed_for_denied_stale_foreign_and_unsafe_requests() -> (
    None
):
    def configure_denied(facade, _calls, _secret):
        facade._selected_episode_id = "retained-episode"

    with _server(secret_override="", configure=configure_denied) as (
        url,
        calls,
        _document,
        _secret,
        cookie,
    ):
        status, body = _post(
            url,
            {"operation": "related_cases", "arguments": {}},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body["status"] == "denied"
        assert calls == []

    class EmptyRelatedService:
        def related_cases(self, _session: Any, _episode_id: str, **_kwargs: Any) -> dict[str, Any]:
            return {"status": "complete", "value": []}

    def configure(facade, _calls, _secret):
        facade._selected_episode_id = "retained-episode"
        facade._context = SimpleNamespace(context_revision=4, episode_id="retained-episode")
        facade.snapshot = lambda: {"status": "complete"}  # type: ignore[method-assign]

    with _server(service=EmptyRelatedService(), configure=configure) as (
        url,
        _calls,
        _document,
        _secret,
        cookie,
    ):
        stale = _post(
            url,
            {
                "operation": "related_cases",
                "arguments": {"expected_context_revision": 3},
            },
            origin=url,
            cookie=cookie,
        )
        foreign = _post(
            url,
            {
                "operation": "related_cases",
                "arguments": {"episode_id": "foreign-episode"},
            },
            origin=url,
            cookie=cookie,
        )
        unsafe = _post(
            url,
            {
                "operation": "related_cases",
                "arguments": {"mode": "same_scenario", "limit": 101},
            },
            origin=url,
            cookie=cookie,
        )

    assert stale == (
        409,
        {"reason": "related-case request targets a stale context revision", "status": "conflict"},
    )
    assert foreign == (
        409,
        {"reason": "related-case request targets a foreign selected episode", "status": "conflict"},
    )
    assert unsafe == (400, {"status": "failed"})


def test_live_route_related_cases_requires_selection_and_rejects_unsafe_envelopes() -> None:
    class RelatedService:
        def __init__(self) -> None:
            self.calls = 0
            self.response: dict[str, Any] = {"status": "complete", "value": []}

        def related_cases(self, _session: Any, _episode_id: str, **_kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            return self.response

    service = RelatedService()

    def configure_unselected(facade, _calls, _secret):
        facade._selected_episode_id = None

    with _server(service=service, configure=configure_unselected) as (
        url,
        _calls,
        _document,
        _secret,
        cookie,
    ):
        status, body = _post(
            url,
            {"operation": "related_cases", "arguments": {"episode_id": "arbitrary-row"}},
            origin=url,
            cookie=cookie,
        )
    assert status == 409
    assert body["status"] == "conflict"
    assert service.calls == 0

    def configure_selected(facade, _calls, _secret):
        facade._selected_episode_id = "episode-generated-123"
        facade._context = SimpleNamespace(context_revision=4, episode_id="episode-generated-123")

    with _server(service=service, configure=configure_selected) as (
        url,
        _calls,
        _document,
        _secret,
        cookie,
    ):
        service.response = {"status": "/secret/token", "value": []}
        bad_status, bad_status_body = _post(
            url,
            {"operation": "related_cases", "arguments": {}},
            origin=url,
            cookie=cookie,
        )
        service.response = {
            "schema_version": "/secret/token",
            "status": "complete",
            "value": [],
        }
        bad_schema, bad_schema_body = _post(
            url,
            {"operation": "related_cases", "arguments": {}},
            origin=url,
            cookie=cookie,
        )
        service.response = {"status": ["complete"], "value": []}
        malformed_status, malformed_status_body = _post(
            url,
            {"operation": "related_cases", "arguments": {}},
            origin=url,
            cookie=cookie,
        )
    assert (bad_status, bad_status_body) == (500, {"status": "failed"})
    assert (bad_schema, bad_schema_body) == (500, {"status": "failed"})
    assert (malformed_status, malformed_status_body) == (500, {"status": "failed"})


def test_server_handles_client_disconnect_without_broken_pipe_crash() -> None:
    handler_entered = threading.Event()
    allow_response = threading.Event()

    def configure(facade, _calls, _secret):
        def slow_next(**kwargs):
            handler_entered.set()
            if not allow_response.wait(timeout=5):
                raise TimeoutError("test did not release the delayed response")
            return {"status": "complete", "value": {}}

        facade.next = slow_next

    with _server(configure=configure) as (url, _calls, _document, _secret, cookie):
        parsed_url = urlsplit(url)
        body = json.dumps(
            {"operation": "next", "arguments": {"expected_selection_revision": 0}}
        ).encode()
        request = (
            f"POST /api/audit HTTP/1.1\r\n"
            f"Host: {parsed_url.netloc}\r\n"
            "Content-Type: application/json\r\n"
            f"Origin: {url}\r\n"
            f"Cookie: {cookie}\r\n"
            "Connection: close\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode() + body
        client = socket.create_connection((parsed_url.hostname, parsed_url.port), timeout=5)
        try:
            client.sendall(request)
            assert handler_entered.wait(timeout=5), "request did not reach the audit facade"
        finally:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
            allow_response.set()

        status, body = _post(
            url,
            {"operation": "next", "arguments": {"expected_selection_revision": 0}},
            origin=url,
            cookie=cookie,
        )
        assert status == 200
        assert body["status"] == "complete"
