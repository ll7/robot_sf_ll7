"""Bounded MCP stdio and private-bridge contract tests."""

from __future__ import annotations

import json
import select
import shutil
import socket
import subprocess
import time
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.analysis_workbench import audit_mcp_stdio
from robot_sf.analysis_workbench.audit_contracts import (
    DetectorRuleProposal,
    EpisodeRef,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_mcp import AuditMCPDispatcher
from robot_sf.analysis_workbench.audit_mcp_stdio import (
    AUDIT_MCP_TOOLS,
    MAX_UNIX_SOCKET_PATH_BYTES,
    MCP_SOCKET_DIR_ENV,
    AuditMCPBridge,
    AuditMCPStdioServer,
)
from robot_sf.analysis_workbench.audit_queue import AuditQueue, QueueDataset
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    AuditSession,
    SessionPolicy,
)
from robot_sf.analysis_workbench.audit_service_adapters import (
    AuditSourceBinding,
    QueueNextAdapter,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)


def _setup(tmp_path: Path) -> tuple[AuditService, object, AuditMCPDispatcher]:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    session = service.open_session(
        AuditSelectionContext(campaign_id="audit-fixture-campaign"),
        actor="agent",
        actor_id="mcp-stdio-agent",
        policy=SessionPolicy(allowed_roots=(str(root),), token_budget=4, compute_budget=4.0),
    )
    return service, session, AuditMCPDispatcher(service)


def _enable_next(
    service: AuditService, session: AuditSession, tmp_path: Path, *, count: int = 2
) -> tuple[QueueDataset, Path]:
    """Bind the real BA-02 Next adapter to the authenticated stdio session."""

    bound = service.update_context(
        session,
        session.context.next_revision(source_identity=session.source_digest),
        expected_context_revision=0,
    )
    assert bound.status == "committed", bound.reason
    campaign_digest = "a" * 64
    source_digest = session.source_digest
    dataset = QueueDataset(
        candidates=tuple(
            EpisodeRef(
                campaign_digest=campaign_digest,
                source_digest=source_digest,
                execution_id=f"stdio-mcp-execution-{index}",
                planner_id="simple_policy",
                scenario_id="corridor",
                seed=index,
                config_digest="c" * 64,
                environment_digest="d" * 64,
            )
            for index in range(count)
        ),
        campaign_digest=campaign_digest,
        source_digest=source_digest,
    )
    queue_path = tmp_path / "queue.json"
    adapter = QueueNextAdapter(
        AuditSourceBinding(
            session.context.campaign_id,
            campaign_digest,
            source_digest,
            session.source_revision,
        ),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    return dataset, queue_path


def _proposal(service: AuditService, session: AuditSession) -> DetectorRuleProposal:
    """Build one source-bound proposal for the stdio JSON-RPC path."""

    campaign = service._scan(session)
    registry = campaign.report.detector_registry
    return DetectorRuleProposal(
        proposal_id="proposal-stdio-1",
        proposal_kind="threshold_change",
        target_detector_id="goal_adjacent_timeout",
        candidate_rule={"predicate": "goal_adjacent_timeout.v1", "parameters": {"tail_steps": 120}},
        detector_registry_version=registry.version,
        detector_registry_digest=registry.digest,
        campaign_digest=campaign.report.audit.campaign_digest,
        source_identity=session.source_digest,
        source_revision=session.source_revision,
        rationale="Source-bound stdio candidate.",
        metadata={"evidence_boundary": "diagnostic_only"},
        proposer_kind="agent",
        proposer_id=session.actor.actor_id,
        author_kind="agent",
        author_id=session.actor.actor_id,
    )


def _initialize(server: AuditMCPStdioServer) -> dict[str, object]:
    response = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": "initialize",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1"},
            },
        }
    )
    assert response is not None
    assert "error" not in response
    assert response["result"]["serverInfo"]["name"] == "robot-sf-audit"
    assert (
        server.handle_message(
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        )
        is None
    )
    return response


def test_stdio_dispatches_real_tools_and_keeps_session_authority_server_owned(
    tmp_path: Path,
) -> None:
    service, session, dispatcher = _setup(tmp_path)
    try:
        server = AuditMCPStdioServer(
            dispatcher,
            session_id=session.session_id,
            session_token=session.session_token,
        )
        initialize = _initialize(server)
        assert initialize["result"]["protocolVersion"] == "2025-06-18"

        listed = server.handle_message(
            {"jsonrpc": "2.0", "id": "list", "method": "tools/list", "params": {}}
        )
        assert listed is not None
        assert [tool["name"] for tool in listed["result"]["tools"]] == list(AUDIT_MCP_TOOLS)
        materialize = next(
            tool for tool in listed["result"]["tools"] if tool["name"] == "materialize_selected"
        )
        assert materialize["inputSchema"]["required"] == ["output_root"]
        next_tool = next(tool for tool in listed["result"]["tools"] if tool["name"] == "next")
        next_properties = next_tool["inputSchema"]["properties"]
        assert next_properties["expected_context_revision"]["type"] == ["integer", "null"]
        assert next_properties["expected_queue_state_revision"]["type"] == ["integer", "null"]
        assert next_properties["expected_queue_input_revision"]["type"] == ["integer", "null"]
        assert next_properties["force_current"] == {"type": "boolean"}
        proposal_tool = next(
            tool
            for tool in listed["result"]["tools"]
            if tool["name"] == "write_detector_rule_proposal"
        )
        assert proposal_tool["inputSchema"]["additionalProperties"] is False
        assert set(proposal_tool["inputSchema"]["properties"]) == {
            "context",
            "operation_id",
            "record",
            "expected_revision",
            "expected_source_revision",
        }
        assert "decided_at" not in proposal_tool["inputSchema"]["properties"]
        assert "approve_detector_rule_proposal" not in AUDIT_MCP_TOOLS

        called = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "read",
                "method": "tools/call",
                "params": {
                    "name": "read_episode",
                    "arguments": {
                        "episode_id": "fixture-readable",
                        "session_token": "model-cannot-replace-this",
                    },
                },
            }
        )
        assert called is not None
        assert called["result"]["isError"] is False
        assert called["result"]["structuredContent"]["status"] == "complete"

        wrong_token = AuditMCPStdioServer(
            dispatcher,
            session_id=session.session_id,
            session_token="not-the-session-token",  # noqa: S106 - deliberate denial case
        )
        _initialize(wrong_token)
        denied = wrong_token.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "denied",
                "method": "tools/call",
                "params": {"name": "read_campaign", "arguments": {}},
            }
        )
        assert denied is not None
        assert denied["result"]["isError"] is True
        assert denied["result"]["structuredContent"]["status"] == "denied"

        denied_next = wrong_token.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "denied-next",
                "method": "tools/call",
                "params": {"name": "next", "arguments": {"operation_id": "denied-next"}},
            }
        )
        assert denied_next is not None
        assert denied_next["result"]["isError"] is True
        assert denied_next["result"]["structuredContent"]["status"] == "denied"
    finally:
        service.close()


def test_stdio_proposal_write_and_read_are_json_safe_and_decisionless(tmp_path: Path) -> None:
    service, session, dispatcher = _setup(tmp_path)
    try:
        server = AuditMCPStdioServer(
            dispatcher,
            session_id=session.session_id,
            session_token=session.session_token,
        )
        _initialize(server)
        proposal = _proposal(service, session)
        created = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "proposal-write",
                "method": "tools/call",
                "params": {
                    "name": "write_detector_rule_proposal",
                    "arguments": {
                        "record": record_to_dict(proposal),
                        "expected_revision": 0,
                    },
                },
            }
        )
        assert created is not None
        assert created["result"]["isError"] is False
        structured = created["result"]["structuredContent"]
        assert structured["status"] == "committed"
        assert structured["result"]["value"]["revision"] == 1
        assert session.session_token not in json.dumps(created, sort_keys=True)

        read = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "proposal-read",
                "method": "tools/call",
                "params": {
                    "name": "read_detector_rule_proposal",
                    "arguments": {"proposal_id": proposal.proposal_id},
                },
            }
        )
        assert read is not None
        assert read["result"]["structuredContent"]["status"] == "complete"
        assert (
            read["result"]["structuredContent"]["result"]["value"]["record"]["activation_status"]
            == "inactive"
        )

        malformed = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "proposal-null-operation-id",
                "method": "tools/call",
                "params": {
                    "name": "read_detector_rule_proposal",
                    "arguments": {
                        "proposal_id": proposal.proposal_id,
                        "operation_id": None,
                    },
                },
            }
        )
        assert malformed is not None
        assert malformed["result"]["isError"] is True
        assert "operation_id" in malformed["result"]["structuredContent"]["reason"]

        decision = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "proposal-decision",
                "method": "tools/call",
                "params": {
                    "name": "approve_detector_rule_proposal",
                    "arguments": {"proposal_id": proposal.proposal_id},
                },
            }
        )
        assert decision is not None
        assert decision["error"]["code"] == -32602
    finally:
        service.close()


def test_stdio_next_validation_replay_and_secret_boundary(tmp_path: Path) -> None:
    service, session, dispatcher = _setup(tmp_path)
    try:
        server = AuditMCPStdioServer(
            dispatcher,
            session_id=session.session_id,
            session_token=session.session_token,
        )
        _initialize(server)

        for request_id, operation_id in (
            ("next-null-operation-id", None),
            ("next-empty-operation-id", ""),
        ):
            malformed_operation = server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {
                        "name": "next",
                        "arguments": {"operation_id": operation_id},
                    },
                }
            )
            assert malformed_operation is not None
            assert malformed_operation["result"]["isError"] is True
            assert (
                malformed_operation["result"]["structuredContent"]["reason"]
                == "MCP next operation_id must be a non-empty bounded string"
            )
            assert session.session_token not in json.dumps(malformed_operation, sort_keys=True)

        malformed_force = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "next-string-force",
                "method": "tools/call",
                "params": {"name": "next", "arguments": {"force_current": "false"}},
            }
        )
        assert malformed_force is not None
        assert malformed_force["result"]["isError"] is True
        assert (
            malformed_force["result"]["structuredContent"]["reason"]
            == "MCP next force_current must be a boolean"
        )

        secret_operation = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "next-secret-operation-id",
                "method": "tools/call",
                "params": {
                    "name": "next",
                    "arguments": {"operation_id": session.session_token},
                },
            }
        )
        assert secret_operation is not None
        assert secret_operation["result"]["isError"] is True
        assert (
            secret_operation["result"]["structuredContent"]["reason"]
            == "MCP next operation_id cannot contain session credentials"
        )
        assert session.session_token not in json.dumps(secret_operation, sort_keys=True)

        next_response = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "next-unavailable",
                "method": "tools/call",
                "params": {
                    "name": "next",
                    "arguments": {
                        "operation_id": "stdio-next-unavailable",
                        "expected_context_revision": 0,
                        "expected_queue_state_revision": 0,
                        "expected_queue_input_revision": 0,
                        "force_current": False,
                    },
                },
            }
        )
        assert next_response is not None
        assert next_response["result"]["isError"] is False
        assert next_response["result"]["structuredContent"]["status"] == "unavailable"
        assert session.session_token not in json.dumps(next_response, sort_keys=True)

        dataset, queue_path = _enable_next(service, session, tmp_path)
        for request_id in (session.session_token, f"prefix-{session.session_token}-suffix"):
            token_id_request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": "next", "arguments": {"force_current": False}},
            }
            token_id_response = server.handle_message(token_id_request)
            assert token_id_response is not None
            serialized = json.dumps(token_id_response, sort_keys=True)
            assert token_id_response["id"] == "<redacted-request-id>"
            assert session.session_token not in serialized
            assert (
                token_id_response["result"]["structuredContent"]["request_id"]
                == "<redacted-request-id>"
            )
            assert token_id_response["result"]["structuredContent"]["status"] == "failed"
        assert AuditQueue(dataset, state_path=queue_path).state_revision == 0

        next_success_request = {
            "jsonrpc": "2.0",
            "id": "next-success",
            "method": "tools/call",
            "params": {"name": "next", "arguments": {"force_current": False}},
        }
        next_success = server.handle_message(next_success_request)
        assert next_success is not None
        assert next_success["id"] == "next-success"
        assert next_success["result"]["isError"] is False
        assert next_success["result"]["structuredContent"]["status"] == "complete"
        assert (
            next_success["result"]["structuredContent"]["result"]["operation"]["operation_id"]
            == "next-success"
        )
        assert session.session_token not in json.dumps(next_success, sort_keys=True)

        next_replay = server.handle_message(next_success_request)
        assert next_replay is not None
        assert next_replay["result"]["isError"] is False
        assert next_replay["result"]["structuredContent"]["result"]["operation"]["replayed"] is True
        assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    finally:
        service.close()


def test_stdio_text_stream_and_oversized_frame_fail_closed(tmp_path: Path) -> None:
    service, session, dispatcher = _setup(tmp_path)
    try:
        server = AuditMCPStdioServer(
            dispatcher,
            session_id=session.session_id,
            session_token=session.session_token,
        )
        messages = (
            "\n".join(
                json.dumps(message)
                for message in (
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {"name": "text-stream", "version": "1"},
                        },
                    },
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                )
            )
            + "\n"
        )
        output = StringIO()
        assert server.run_stdio(StringIO(messages), output, max_messages=2) == 0
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        assert [item["id"] for item in responses] == [1, 2]

        oversized = AuditMCPStdioServer(
            dispatcher,
            session_id=session.session_id,
            session_token=session.session_token,
        )
        bad_output = StringIO()
        bad_line = json.dumps({"jsonrpc": "2.0", "method": "ping", "x": "x" * 2_000})
        assert (
            oversized.run_stdio(
                StringIO(bad_line + "\n"),
                bad_output,
                max_frame_bytes=512,
                lifetime_seconds=1.0,
            )
            == 2
        )
        assert json.loads(bad_output.getvalue())["error"]["code"] == -32600
    finally:
        service.close()


def test_private_bridge_runs_external_stdio_proxy_and_validates_dispatcher_token(
    tmp_path: Path,
) -> None:
    service, session, dispatcher = _setup(tmp_path)
    bridge = AuditMCPBridge(
        dispatcher, session_id=session.session_id, session_token=session.session_token
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        bridge.start()
        assert session.session_token not in " ".join(bridge.command())
        environment = bridge.environment(
            {audit_mcp_stdio.MCP_SESSION_TOKEN_ENV: session.session_token}
        )
        assert audit_mcp_stdio.MCP_SESSION_TOKEN_ENV not in environment
        assert session.session_token not in json.dumps(environment, sort_keys=True)
        process = subprocess.Popen(
            list(bridge.command()),
            cwd=str(Path.cwd()),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None and process.stdout is not None

        def exchange(message: dict[str, object]) -> dict[str, object]:
            process.stdin.write(json.dumps(message).encode() + b"\n")
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], 3.0)
            assert ready, "MCP proxy did not return a bounded response"
            line = process.stdout.readline()
            assert line
            return json.loads(line)

        initialized = exchange(
            {
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "external-proxy", "version": "1"},
                },
            }
        )
        assert initialized["result"]["serverInfo"]["name"] == "robot-sf-audit"
        process.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
            ).encode()
            + b"\n"
        )
        process.stdin.flush()
        listed = exchange({"jsonrpc": "2.0", "id": "list", "method": "tools/list", "params": {}})
        assert len(listed["result"]["tools"]) == len(AUDIT_MCP_TOOLS)
        called = exchange(
            {
                "jsonrpc": "2.0",
                "id": "call",
                "method": "tools/call",
                "params": {
                    "name": "read_episode",
                    "arguments": {"episode_id": "fixture-readable"},
                },
            }
        )
        assert called["result"]["structuredContent"]["status"] == "complete"
        assert session.session_token not in json.dumps(
            (initialized, listed, called), sort_keys=True
        )
    finally:
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=3.0)
        bridge.close()
        service.close()


def test_bridge_close_unblocks_silent_client(tmp_path: Path) -> None:
    service, session, dispatcher = _setup(tmp_path)
    bridge = AuditMCPBridge(
        dispatcher,
        session_id=session.session_id,
        session_token=session.session_token,
    )
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        bridge.start()
        client.connect(str(bridge.socket_path))
        deadline = time.monotonic() + 1.0
        while not bridge._client_threads and time.monotonic() < deadline:
            time.sleep(0.01)
        assert bridge._client_threads
        client_threads = tuple(bridge._client_threads)
        bridge.close()
        assert all(not thread.is_alive() for thread in client_threads)
    finally:
        client.close()
        bridge.close()
        service.close()


@pytest.mark.parametrize(
    "message",
    [
        {"jsonrpc": "1.0", "id": "bad", "method": "ping"},
        {"jsonrpc": "2.0", "id": True, "method": "ping"},
        {"jsonrpc": "2.0", "id": "bad", "method": ""},
        {"jsonrpc": "2.0", "id": "bad", "method": "ping", "params": []},
        {"jsonrpc": "2.0", "id": "bad", "method": "ping", "params": {"x": float("nan")}},
    ],
)
def test_stdio_rejects_malformed_json_rpc_before_dispatch(message: dict) -> None:
    with pytest.raises(audit_mcp_stdio.MCPProtocolError):
        audit_mcp_stdio._mapping_message(message)


def test_stdio_initialization_and_tool_errors_are_bounded(tmp_path: Path) -> None:
    service, session, dispatcher = _setup(tmp_path)
    try:
        server = AuditMCPStdioServer(
            dispatcher,
            session_id=session.session_id,
            session_token=session.session_token,
        )
        before = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert before["error"]["code"] == -32002
        assert server.handle_message({"jsonrpc": "2.0", "method": "tools/list"}) is None
        early_notification = server.handle_message(
            {"jsonrpc": "2.0", "id": "early", "method": "notifications/initialized"}
        )
        assert early_notification["error"]["code"] == -32002
        missing_client = server.handle_message(
            {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}}
        )
        assert missing_client["error"]["code"] == -32602
        incomplete_client = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "incomplete",
                "method": "initialize",
                "params": {"clientInfo": {"name": "test"}},
            }
        )
        assert incomplete_client["error"]["code"] == -32602
        invalid_version = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": "version",
                "method": "initialize",
                "params": {"clientInfo": {"name": "test", "version": "1"}, "protocolVersion": 1},
            }
        )
        assert invalid_version["error"]["code"] == -32602
        _initialize(server)
        duplicate = server.handle_message(
            {"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {}}
        )
        assert duplicate["error"]["code"] == -32600
        assert server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "ping"})["result"] == {}
        assert server.handle_message({"jsonrpc": "2.0", "method": "ping"}) is None
        assert server.handle_message({"jsonrpc": "2.0", "method": "tools/list"}) is None
        assert server.handle_message({"jsonrpc": "2.0", "method": "tools/call"}) is None
        missing_name = server.handle_message(
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {}}
        )
        assert missing_name["error"]["code"] == -32602
        invalid_args = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "read_episode", "arguments": []},
            }
        )
        assert invalid_args["error"]["code"] == -32602
        unknown_tool = server.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "run_shell", "arguments": {}},
            }
        )
        assert unknown_tool["error"]["code"] == -32602
        unknown_method = server.handle_message(
            {"jsonrpc": "2.0", "id": 8, "method": "resources/read"}
        )
        assert unknown_method["error"]["code"] == -32601
        assert server.handle_message({"jsonrpc": "2.0", "method": "resources/read"}) is None
    finally:
        service.close()


class _RetainedBytesIO(BytesIO):
    """Keep captured stdout readable after the proxy closes its stream."""

    def close(self) -> None:
        """The proxy owns the stream lifetime, not this test's captured bytes."""


def test_in_process_proxy_forwards_complete_frames_without_token_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, session, dispatcher = _setup(tmp_path)
    bridge = AuditMCPBridge(
        dispatcher,
        session_id=session.session_id,
        session_token=session.session_token,
    )
    try:
        bridge.start()
        messages = [
            {
                "jsonrpc": "2.0",
                "id": "init",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "in-process", "version": "1"},
                },
            },
            {
                "jsonrpc": "2.0",
                "id": "case",
                "method": "tools/call",
                "params": {"name": "read_episode", "arguments": {"episode_id": "fixture-readable"}},
            },
        ]
        input_bytes = BytesIO(b"".join(json.dumps(item).encode() + b"\n" for item in messages))
        output_bytes = _RetainedBytesIO()
        monkeypatch.setattr(audit_mcp_stdio.sys, "stdin", SimpleNamespace(buffer=input_bytes))
        monkeypatch.setattr(audit_mcp_stdio.sys, "stdout", SimpleNamespace(buffer=output_bytes))
        monkeypatch.setattr(
            audit_mcp_stdio.select,
            "select",
            lambda readers, _writers, _errors, _timeout: (readers, [], []),
        )
        assert audit_mcp_stdio._proxy_stdio(bridge.socket_path) == 0
        responses = [json.loads(line) for line in output_bytes.getvalue().splitlines()]
        assert [item["id"] for item in responses] == ["init", "case"]
        assert responses[1]["result"]["structuredContent"]["status"] == "complete"
        assert session.session_token.encode() not in output_bytes.getvalue()
    finally:
        bridge.close()
        service.close()


def test_mcp_entrypoints_fail_closed_without_private_bridge_or_dispatcher(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="injected dispatcher"):
        audit_mcp_stdio._direct_server_from_environment()
    assert audit_mcp_stdio.main(["--bridge-socket", str(tmp_path / "missing.sock")]) == 2


def test_bridge_allocates_safe_socket_under_long_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deeply nested TMPDIR falls back to a safe short root within platform limits."""
    long_tmpdir = tmp_path / ("nested_scratch_dir_" + "x" * 70) / "deep_tmp"
    long_tmpdir.mkdir(parents=True)
    monkeypatch.setenv("TMPDIR", str(long_tmpdir))

    service, session, dispatcher = _setup(tmp_path)
    bridge = AuditMCPBridge(
        dispatcher,
        session_id=session.session_id,
        session_token=session.session_token,
    )
    try:
        assert len(str(bridge.socket_path).encode()) < MAX_UNIX_SOCKET_PATH_BYTES
        assert bridge.socket_path.parent.exists()
        bridge.start()
        assert bridge.running
    finally:
        socket_dir = bridge.socket_path.parent
        bridge.close()
        service.close()
        assert not bridge.socket_path.exists()
        assert not socket_dir.exists()


def test_bridge_respects_configured_socket_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configured ROBOT_SF_AUDIT_MCP_SOCKET_DIR is prioritized when safe."""
    custom_root = Path("/tmp/mcp_cfg_root")
    custom_root.mkdir(exist_ok=True)
    try:
        monkeypatch.setenv(MCP_SOCKET_DIR_ENV, str(custom_root))

        service, session, dispatcher = _setup(tmp_path)
        bridge = AuditMCPBridge(
            dispatcher,
            session_id=session.session_id,
            session_token=session.session_token,
        )
        try:
            assert custom_root in bridge.socket_path.parents
            assert len(str(bridge.socket_path).encode()) < MAX_UNIX_SOCKET_PATH_BYTES
        finally:
            bridge.close()
            service.close()
    finally:
        shutil.rmtree(custom_root, ignore_errors=True)


def test_bridge_fails_closed_when_explicit_socket_path_oversized(tmp_path: Path) -> None:
    """An explicit socket path that exceeds platform limit fails closed."""
    service, session, dispatcher = _setup(tmp_path)
    oversized_path = tmp_path / ("oversized_path_" + "y" * 100 + ".sock")
    try:
        with pytest.raises(ValueError, match="Unix socket path exceeds the platform limit"):
            AuditMCPBridge(
                dispatcher,
                session_id=session.session_id,
                session_token=session.session_token,
                socket_path=oversized_path,
            )
    finally:
        service.close()


def test_allocate_temporary_socket_fails_closed_when_all_candidates_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no candidate root produces a valid path, fail closed with ValueError."""

    class FakeOversizedTempDir:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.name = (
                "/a/very/long/temporary/directory/path/that/is/guaranteed/to/exceed/"
                "the/unix/domain/socket/limit"
            )

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(audit_mcp_stdio.tempfile, "TemporaryDirectory", FakeOversizedTempDir)
    with pytest.raises(ValueError, match="Unix socket path exceeds the platform limit"):
        audit_mcp_stdio._allocate_temporary_socket()
