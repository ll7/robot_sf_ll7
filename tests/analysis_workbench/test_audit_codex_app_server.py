"""Focused App Server v2 transport, failure, and lifecycle tests."""

from __future__ import annotations

import os
import shutil
import signal
import stat
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from robot_sf.analysis_workbench.audit_codex import AuditCodexClient
from robot_sf.analysis_workbench.audit_codex_app_server import (
    APP_SERVER_PROTOCOL,
    UNMEASURED_TURN_COMPUTE_CHARGE,
    AppServerFrame,
    AppServerMCPConfig,
    AppServerProcessLost,
    AppServerProtocolError,
    AppServerTimeout,
    AppServerUnavailable,
    CodexAppServerConfig,
    CodexAppServerProvider,
    CodexAppServerTransport,
    inspect_live_capabilities,
)
from robot_sf.analysis_workbench.audit_mcp import AuditMCPDispatcher
from robot_sf.analysis_workbench.audit_mcp_stdio import AuditMCPBridge
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    SessionPolicy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)

FAKE_CODEX_BODY = r"""
import json
import os
from pathlib import Path
import sys
import time

if "--version" in sys.argv:
    print("codex-cli 0.154.0", flush=True)
    raise SystemExit(0)

MODE = __MODE__
turn_number = 0
thread_ephemeral = False
thread_ready = False
state_path = os.environ.get("FAKE_THREAD_STATE")

if MODE == "sigterm-resistant":
    import signal

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("not-json", flush=True)
    while True:
        time.sleep(0.05)
elif MODE == "malformed":
    print("not-json", flush=True)
    time.sleep(5)
elif MODE == "oversized":
    print("x" * 5000, flush=True)
    time.sleep(5)
else:
    for raw in sys.stdin:
        request = json.loads(raw)
        method = request.get("method")
        request_id = request.get("id")

        def reply(result):
            print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)

        def error(code, message):
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": code, "message": message},
                    }
                ),
                flush=True,
            )

        if method == "initialize":
            reply({"userAgent": "fake-codex/0.154.0"})
        elif method == "initialized":
            pass
        elif method == "model/list":
            reply({"data": [{"id": "model-fixture", "isDefault": True}]})
        elif method == "thread/start":
            thread_ephemeral = bool(request["params"].get("ephemeral", False))
            thread_ready = False
            reply(
                {
                    "thread": {"id": "thread-fixture", "ephemeral": thread_ephemeral},
                    "model": "model-fixture",
                    "modelProvider": "provider-fixture",
                }
            )
        elif method == "thread/resume":
            durable_rollout = bool(state_path and Path(state_path).exists())
            if thread_ephemeral or (not thread_ready and not durable_rollout):
                error(-32600, "no rollout found")
            else:
                reply(
                    {
                        "thread": {"id": "thread-fixture", "ephemeral": False},
                        "model": "model-fixture",
                        "modelProvider": "provider-fixture",
                    }
                )
        elif method == "turn/start":
            turn_number += 1
            if not thread_ephemeral:
                thread_ready = True
                if state_path:
                    Path(state_path).write_text("durable", encoding="utf-8")
            turn_id = "turn-" + str(turn_number)
            reply({"turn": {"id": turn_id}})
            thread_id = request["params"]["threadId"]
            usage = {
                "last": {
                    "cachedInputTokens": 0,
                    "inputTokens": 3,
                    "outputTokens": 4,
                    "reasoningOutputTokens": 0,
                    "totalTokens": 7,
                },
                "total": {
                    "cachedInputTokens": 0,
                    "inputTokens": 3 * turn_number,
                    "outputTokens": 4 * turn_number,
                    "reasoningOutputTokens": 0,
                    "totalTokens": 7 * turn_number,
                },
            }
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "thread/tokenUsage/updated",
                        "params": {"threadId": thread_id, "turnId": turn_id, "tokenUsage": usage},
                    }
                ),
                flush=True,
            )
            print(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "turn/completed",
                        "params": {"threadId": thread_id, "turn": {"id": turn_id, "status": "completed"}},
                    }
                ),
                flush=True,
            )
        elif method == "turn/interrupt":
            reply({})
        else:
            error(-32601, "unknown method")
"""


def _fake_codex(tmp_path: Path, mode: str = "normal") -> Path:
    executable = tmp_path / f"fake-codex-{mode}"
    executable.write_text(
        "#!" + os.sys.executable + "\n" + FAKE_CODEX_BODY.replace("__MODE__", repr(mode)),
        encoding="utf-8",
    )
    executable.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    return executable


def _config(
    executable: Path,
    *,
    timeout: float = 2.0,
    frame_bytes: int = 1_048_576,
    lifetime: float = 10.0,
):
    return CodexAppServerConfig(
        executable=str(executable),
        cwd=executable.parent,
        env={"FAKE_THREAD_STATE": str(executable.with_suffix(".state"))},
        request_timeout_seconds=timeout,
        startup_timeout_seconds=timeout,
        lifetime_seconds=lifetime,
        close_timeout_seconds=0.5,
        max_frame_bytes=frame_bytes,
    )


def _meter_setup(
    tmp_path: Path,
    *,
    token_budget: int = 100,
    compute_budget: float = UNMEASURED_TURN_COMPUTE_CHARGE,
):
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    session = service.open_session(
        AuditSelectionContext(campaign_id="audit-fixture-campaign"),
        actor="agent",
        actor_id="codex-agent",
        policy=SessionPolicy(
            allowed_roots=(str(root),),
            token_budget=token_budget,
            compute_budget=compute_budget,
        ),
    )
    return service, session


@pytest.mark.parametrize(
    "invalid",
    [
        {"cwd": 123},
        {"request_timeout_seconds": 0},
        {"startup_timeout_seconds": float("nan")},
        {"lifetime_seconds": -1},
        {"close_timeout_seconds": 0},
        {"max_frame_bytes": 100},
        {"max_events": 0},
    ],
)
def test_app_server_config_rejects_unbounded_or_invalid_settings(invalid: dict) -> None:
    with pytest.raises(ValueError):
        CodexAppServerConfig(**invalid)


def test_app_server_config_rejects_empty_client_identity() -> None:
    with pytest.raises(AppServerProtocolError):
        CodexAppServerConfig(client_name="")


def test_app_server_private_mcp_config_keeps_token_out_of_process_arguments(
    tmp_path: Path,
) -> None:
    service, audit_session = _meter_setup(tmp_path)
    bridge = AuditMCPBridge(
        AuditMCPDispatcher(service),
        session_id=audit_session.session_id,
        session_token=audit_session.session_token,
    )
    try:
        with pytest.raises(ValueError):
            AppServerMCPConfig(bridge, name="bad.name")
        config = CodexAppServerConfig(
            executable=str(_fake_codex(tmp_path)),
            mcp=AppServerMCPConfig(bridge),
            env={"BA05_TEST": "value"},
        )
        command = config.command()
        assert "-c" in command
        assert audit_session.session_token not in " ".join(command)
        assert any("mcp_servers.robot_sf_audit" in item for item in command)
        environment = config.process_environment()
        assert "ROBOT_SF_AUDIT_MCP_SESSION_TOKEN" not in environment
        assert audit_session.session_token not in str(environment)
        assert environment["BA05_TEST"] == "value"
        with pytest.raises(ValueError):
            replace(config, env={"INVALID": 7}).process_environment()
        with pytest.raises(ValueError, match="cannot override"):
            replace(
                config, env={"ROBOT_SF_AUDIT_MCP_SESSION_TOKEN": "explicit-secret"}
            ).process_environment()
    finally:
        bridge.close()
        service.close()


def test_app_server_without_mcp_drops_ambient_audit_session_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROBOT_SF_AUDIT_MCP_SESSION_TOKEN", "ambient-secret")
    environment = CodexAppServerConfig().process_environment()
    assert "ROBOT_SF_AUDIT_MCP_SESSION_TOKEN" not in environment


def test_app_server_provider_refuses_guessed_or_wrong_version_route_before_work(
    tmp_path: Path,
) -> None:
    config = _config(_fake_codex(tmp_path))
    inspection = inspect_live_capabilities(config)
    route = inspection.choose()
    assert route is not None
    provider = CodexAppServerProvider(config=config)
    try:
        for invalid_route in (
            replace(route, protocol="guessed-protocol"),
            replace(route, client_version="unverified-version"),
        ):
            with pytest.raises(AppServerUnavailable):
                provider.start(
                    route=invalid_route,
                    prompt="do not run",
                    evidence=[],
                    reserved_compute=UNMEASURED_TURN_COMPUTE_CHARGE,
                )
            assert provider.transport.process is None
    finally:
        provider.close()


@pytest.mark.parametrize("reserved_compute", [None, True, 0, 0.5, float("nan")])
def test_provider_rejects_invalid_turn_compute_before_transport(
    tmp_path: Path, reserved_compute: float | None
) -> None:
    provider = CodexAppServerProvider(config=_config(_fake_codex(tmp_path)))
    try:
        with pytest.raises(AppServerUnavailable):
            provider._require_turn_reservation(reserved_compute)
        assert provider.transport.process is None
    finally:
        provider.close()


@pytest.mark.parametrize(
    "token_usage",
    [
        None,
        {},
        {"total": {}},
        {"total": {"totalTokens": True}},
        {"total": {"totalTokens": -1}},
    ],
)
def test_provider_missing_or_invalid_usage_never_gets_a_success_meter(token_usage: object) -> None:
    frames = [
        AppServerFrame(
            1,
            {
                "method": "thread/tokenUsage/updated",
                "params": {"tokenUsage": token_usage},
            },
        )
    ]
    assert CodexAppServerProvider._usage(frames) == {}


def test_provider_settles_last_turn_tokens_not_cumulative_thread_tokens() -> None:
    frames = [
        AppServerFrame(
            1,
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "tokenUsage": {
                        "total": {"totalTokens": 14},
                        "last": {"totalTokens": 7},
                    }
                },
            },
        )
    ]
    usage = CodexAppServerProvider._usage(frames)
    assert usage["tokens"] == 7
    assert usage["compute"] == UNMEASURED_TURN_COMPUTE_CHARGE


def test_provider_turn_receipts_require_typed_thread_and_turn_ids() -> None:
    with pytest.raises(AppServerProtocolError):
        CodexAppServerProvider._thread_id({"thread": None})
    with pytest.raises(AppServerProtocolError):
        CodexAppServerProvider._turn_id({"turn": None})
    prompt = CodexAppServerProvider._prompt_with_evidence(
        "inspect",
        [None, {"evidence_id": "e1", "locator": "source:episode"}],
    )
    assert "e1: source:episode" in prompt


class _ReservedProvider(CodexAppServerProvider):
    """Test authority seam that forwards an explicit pre-admitted charge."""

    def __init__(self, *, config: CodexAppServerConfig, reserved_compute: float) -> None:
        super().__init__(config=config)
        self.reserved_compute = reserved_compute

    def start(self, **kwargs):
        return super().start(reserved_compute=self.reserved_compute, **kwargs)

    def resume(self, **kwargs):
        return super().resume(reserved_compute=self.reserved_compute, **kwargs)


def test_live_capability_discovery_and_provider_lifecycle_use_actual_route(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    config = _config(executable)
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "available"
    assert len(inspection.routes) == 1
    route = inspection.routes[0]
    assert route.protocol == APP_SERVER_PROTOCOL
    assert route.provider == "provider-fixture"
    assert route.model_id == "model-fixture"

    provider = CodexAppServerProvider(config=config)
    try:
        started = provider.start(
            route=route,
            prompt="read the scoped evidence",
            evidence=[],
            reserved_compute=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert started["status"] == "complete"
        assert started["usage"]["tokens"] == 7
        assert started["usage"]["compute_measured"] is False
        session_id = started["provider_session_id"]

        resumed = provider.resume(
            provider_session_id=session_id,
            route=route,
            evidence=[],
            reserved_compute=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert resumed["status"] == "complete"
        assert resumed["usage"]["tokens"] == 7
        provider.transport.close()
        reconnected = provider.reconnect(provider_session_id=session_id, route=route)
        assert reconnected["status"] == "complete"
        assert reconnected["restarted"] is True
        provider._turns[session_id] = "turn-for-cancel"
        cancelled = provider.cancel(provider_session_id=session_id, route=route)
        assert cancelled["status"] == "cancelled"
        assert provider.transport.last_sequence > 0
        sequences = [frame.sequence for frame in provider.transport.events_since()]
        assert sequences == sorted(sequences)
    finally:
        provider.close()


def test_unmeasured_turn_charge_advances_aggregate_budget_before_repeated_work(
    tmp_path: Path,
) -> None:
    executable = _fake_codex(tmp_path)
    config = _config(executable)
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "available"
    route = inspection.choose()
    assert route is not None
    service, audit_session = _meter_setup(tmp_path)
    provider = _ReservedProvider(
        config=config,
        reserved_compute=UNMEASURED_TURN_COMPUTE_CHARGE,
    )
    client = AuditCodexClient(
        service,
        inspector=lambda: inspection,
        provider=provider,
    )
    try:
        first = client.start(
            audit_session,
            prompt="inspect the scoped evidence once",
            route_id=route.route_id,
            operation_id="meter-first",
            token_budget=7,
            compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert first.status == "complete"
        assert first.receipt is not None
        assert first.receipt.usage["compute"] == UNMEASURED_TURN_COMPUTE_CHARGE
        assert first.receipt.usage["compute_measured"] is False
        assert first.receipt.usage["compute_basis"] == ("bounded-per-turn-unmeasured-charge")
        assert audit_session.usage.compute == UNMEASURED_TURN_COMPUTE_CHARGE
        sequence_after_first = provider.transport.last_sequence

        second = client.start(
            audit_session,
            prompt="attempt the same live work again",
            route_id=route.route_id,
            operation_id="meter-second",
            token_budget=7,
            compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert second.status == "denied"
        assert "budget" in second.reason
        assert audit_session.usage.compute == UNMEASURED_TURN_COMPUTE_CHARGE
        assert provider.transport.last_sequence == sequence_after_first
    finally:
        provider.close()
        service.close()


def test_durable_resume_settles_per_turn_tokens_from_cumulative_usage(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path, "cumulative")
    config = _config(executable)
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "available"
    route = inspection.choose()
    assert route is not None
    service, audit_session = _meter_setup(
        tmp_path,
        token_budget=14,
        compute_budget=2.0,
    )
    provider = _ReservedProvider(
        config=config,
        reserved_compute=UNMEASURED_TURN_COMPUTE_CHARGE,
    )
    client = AuditCodexClient(
        service,
        inspector=lambda: inspection,
        provider=provider,
    )
    try:
        first = client.start(
            audit_session,
            prompt="inspect the scoped evidence once",
            route_id=route.route_id,
            operation_id="cumulative-first",
            token_budget=7,
            compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert first.status == "complete"
        assert first.session is not None

        second = client.resume(
            first.session,
            operation_id="cumulative-resume",
            token_budget=7,
            compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
        )

        assert second.status == "complete"
        assert second.receipt is not None
        assert second.receipt.usage["tokens"] == 7
        assert audit_session.usage.to_dict() == {
            "tokens": 14,
            "compute": 2.0,
            "issue_writes": 0,
        }
    finally:
        provider.close()
        service.close()


def test_unmeasured_turn_zero_policy_fails_before_transport_work(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    config = _config(executable)
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "available"
    service, audit_session = _meter_setup(tmp_path, compute_budget=0.0)
    provider = CodexAppServerProvider(config=config)
    client = AuditCodexClient(
        service,
        inspector=lambda: inspection,
        provider=provider,
    )
    try:
        result = client.start(
            audit_session,
            prompt="must be rejected before a live turn",
            operation_id="meter-zero-policy",
            token_budget=7,
        )
        assert result.status == "failed"
        assert "reserved_compute" in result.reason
        assert provider.transport.last_sequence == 0
        assert provider.transport.process is None
        assert audit_session.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
    finally:
        provider.close()
        service.close()


@pytest.mark.parametrize("reserved_compute", [0.0, 0.5])
def test_caller_insufficient_pre_admission_fails_before_transport_work(
    tmp_path: Path, reserved_compute: float
) -> None:
    executable = _fake_codex(tmp_path)
    config = _config(executable)
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "available"
    route = inspection.choose()
    assert route is not None
    service, audit_session = _meter_setup(tmp_path)
    provider = _ReservedProvider(config=config, reserved_compute=reserved_compute)
    client = AuditCodexClient(
        service,
        inspector=lambda: inspection,
        provider=provider,
    )
    try:
        result = client.start(
            audit_session,
            prompt="caller omitted the required compute reservation",
            route_id=route.route_id,
            operation_id=f"meter-insufficient-caller-{reserved_compute}",
            token_budget=7,
            compute_budget=reserved_compute,
        )
        assert result.status == "failed"
        assert "reserved_compute" in result.reason
        assert provider.transport.last_sequence == 0
        assert provider.transport.process is None
        assert audit_session.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
    finally:
        provider.close()
        service.close()


@pytest.mark.parametrize(
    "mode, error_type",
    [("malformed", AppServerProtocolError), ("oversized", AppServerProtocolError)],
)
def test_transport_rejects_malformed_or_oversized_frames(
    tmp_path: Path, mode: str, error_type: type[Exception]
) -> None:
    transport = CodexAppServerTransport(_config(_fake_codex(tmp_path, mode), frame_bytes=1_024))
    try:
        with pytest.raises(error_type):
            transport.initialize()
        assert transport.failure is not None
        process = transport.process
        assert process is not None
        deadline = time.monotonic() + 1.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert process.poll() is not None
    finally:
        transport.close()


def test_transport_force_kills_sigterm_resistant_protocol_child(tmp_path: Path) -> None:
    transport = CodexAppServerTransport(
        _config(_fake_codex(tmp_path, "sigterm-resistant"), timeout=0.5)
    )
    try:
        started_at = time.monotonic()
        with pytest.raises(AppServerProtocolError):
            transport.initialize()
        elapsed = time.monotonic() - started_at
        process = transport.process
        assert process is not None
        assert process.poll() == -signal.SIGKILL
        assert elapsed < 1.5
    finally:
        transport.close()


def test_transport_terminates_idle_process_at_configured_lifetime(tmp_path: Path) -> None:
    transport = CodexAppServerTransport(_config(_fake_codex(tmp_path), lifetime=0.1))
    try:
        transport.start()
        process = transport.process
        assert process is not None
        deadline = time.monotonic() + 1.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert isinstance(transport.failure, AppServerTimeout)
        assert process.poll() is not None
        assert not transport.is_alive
    finally:
        transport.close()


def test_transport_reports_process_loss_and_request_timeout(tmp_path: Path) -> None:
    lost = _fake_codex(tmp_path, "lost")
    lost.write_text(
        "#!"
        + os.sys.executable
        + "\nimport sys\nif '--version' in sys.argv:\n print('codex-cli 0.154.0')\nelse:\n raise SystemExit(0)\n",
        encoding="utf-8",
    )
    lost.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    lost_transport = CodexAppServerTransport(_config(lost, timeout=0.5))
    try:
        with pytest.raises(AppServerProcessLost):
            lost_transport.initialize()
    finally:
        lost_transport.close()

    slow = tmp_path / "fake-codex-slow"
    slow.write_text(
        "#!" + os.sys.executable + "\n"
        "import json, sys, time\n"
        "if '--version' in sys.argv: print('codex-cli 0.154.0'); raise SystemExit(0)\n"
        "for raw in sys.stdin:\n"
        " request=json.loads(raw)\n"
        " if request.get('method') == 'initialize':\n"
        "  print(json.dumps({'id':request['id'],'result':{}}), flush=True)\n"
        " elif request.get('method') == 'model/list':\n"
        "  time.sleep(2)\n",
        encoding="utf-8",
    )
    slow.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    slow_transport = CodexAppServerTransport(_config(slow, timeout=0.1))
    try:
        slow_transport.initialize()
        with pytest.raises(AppServerTimeout):
            slow_transport.request("model/list", {}, timeout_seconds=0.1)
    finally:
        slow_transport.close()


@pytest.mark.skipif(
    os.environ.get("ROBOT_SF_RUN_LIVE_CODEX_REGRESSION") != "1",
    reason="set ROBOT_SF_RUN_LIVE_CODEX_REGRESSION=1 for the credentialed installed-CLI probe",
)
def test_installed_app_server_durable_thread_resumes_after_completed_turn() -> None:
    executable = shutil.which("codex")
    if executable is None:
        pytest.fail("ROBOT_SF_RUN_LIVE_CODEX_REGRESSION=1 but codex is unavailable")
    config = CodexAppServerConfig(
        executable=executable,
        cwd=Path.cwd(),
        request_timeout_seconds=45.0,
        startup_timeout_seconds=15.0,
        lifetime_seconds=90.0,
    )
    transport = CodexAppServerTransport(config)
    try:
        transport.initialize()
        listed = transport.request("model/list", {"includeHidden": False, "limit": 100})
        models = listed.get("data")
        assert isinstance(models, list)
        model = next(item["id"] for item in models if item.get("isDefault") is True)
        started = transport.request(
            "thread/start",
            {"model": model, "ephemeral": False, "approvalPolicy": "never", "sandbox": "read-only"},
        )
        thread = started["thread"]
        thread_id = thread["id"]
        assert thread["ephemeral"] is False
        turn = transport.request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": "Reply exactly RESUME_REGRESSION."}],
            },
        )
        turn_id = turn["turn"]["id"]
        completed = transport.wait_for_notification(
            lambda frame: (
                frame.message.get("method") == "turn/completed"
                and frame.message.get("params", {}).get("turn", {}).get("id") == turn_id
            ),
            timeout_seconds=45.0,
        )
        assert completed.message["params"]["turn"]["status"] == "completed"
        resumed = transport.request("thread/resume", {"threadId": thread_id, "model": model})
        assert resumed["thread"]["id"] == thread_id
        assert resumed["thread"]["ephemeral"] is False
    finally:
        transport.close()


@pytest.mark.skipif(
    os.environ.get("ROBOT_SF_RUN_LIVE_CODEX_REGRESSION") != "1",
    reason="set ROBOT_SF_RUN_LIVE_CODEX_REGRESSION=1 for the credentialed audit-client probe",
)
def test_installed_app_server_audit_client_charges_live_turn(
    tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    """A real selected route must settle measured tokens and bounded compute."""

    executable = shutil.which("codex")
    if executable is None:
        pytest.fail("ROBOT_SF_RUN_LIVE_CODEX_REGRESSION=1 but codex is unavailable")
    config = CodexAppServerConfig(
        executable=executable,
        cwd=tmp_path,
        request_timeout_seconds=45.0,
        startup_timeout_seconds=15.0,
        lifetime_seconds=90.0,
    )
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "available", inspection.reason
    route = inspection.choose()
    assert route is not None
    service, audit_session = _meter_setup(
        tmp_path,
        token_budget=30_000,
        compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
    )
    provider = CodexAppServerProvider(config=config)
    client = AuditCodexClient(service, inspector=lambda: inspection, provider=provider)
    try:
        result = client.start(
            audit_session,
            prompt="Reply with the single word READY. Do not use tools.",
            route_id=route.route_id,
            operation_id="live-audit-client-start",
            token_budget=30_000,
            compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert result.status == "complete", (
            result.status,
            result.reason,
            result.receipt.usage if result.receipt is not None else None,
            audit_session.usage.to_dict(),
        )
        assert result.receipt is not None
        assert result.receipt.usage["tokens"] > 0
        assert result.receipt.usage["compute"] == UNMEASURED_TURN_COMPUTE_CHARGE
        assert result.receipt.usage["compute_measured"] is False
        record_property("route_id", route.route_id)
        record_property("route_version", route.client_version)
        record_property("tokens_used", result.receipt.usage["tokens"])
        record_property("compute_charge", result.receipt.usage["compute"])
        assert audit_session.usage.tokens == result.receipt.usage["tokens"]
        assert audit_session.usage.compute == UNMEASURED_TURN_COMPUTE_CHARGE
        sequence = provider.transport.last_sequence
        replay = client.start(
            audit_session,
            prompt="Reply with the single word READY. Do not use tools.",
            route_id=route.route_id,
            operation_id="live-audit-client-start",
            token_budget=30_000,
            compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert replay.status == "complete", replay.reason
        assert replay.receipt is not None and replay.receipt.replayed
        assert provider.transport.last_sequence == sequence
    finally:
        provider.close()
        service.close()


def test_d2_strict_provider_ceiling_requires_verified_capability() -> None:
    with pytest.raises(ValueError, match="verified provider_compute_ceiling"):
        CodexAppServerConfig(accounting_mode="strict_provider_ceiling")


def test_d2_model_list_never_falls_back_from_an_exact_requested_model(tmp_path: Path) -> None:
    config = replace(_config(_fake_codex(tmp_path)), required_model_id="premium-not-installed")
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "unavailable"
    assert "requested exact model" in inspection.reason


def test_d2_explicit_route_gate_rejects_default_selection_before_transport(
    tmp_path: Path,
) -> None:
    executable = _fake_codex(tmp_path)
    config = replace(
        _config(executable),
        require_explicit_route=True,
        required_route_id="provider-fixture:model-fixture",
        required_provider="provider-fixture",
        required_model_id="model-fixture",
    )
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "available"
    service, audit_session = _meter_setup(tmp_path)
    provider = _ReservedProvider(
        config=config,
        reserved_compute=UNMEASURED_TURN_COMPUTE_CHARGE,
    )
    client = AuditCodexClient(service, inspector=lambda: inspection, provider=provider)
    try:
        result = client.start(
            audit_session,
            prompt="explicit route is required",
            operation_id="d2-explicit-route",
            token_budget=7,
            compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert result.status == "unavailable"
        assert "explicit provider/model route" in result.reason
        assert provider.transport.process is None
        assert audit_session.usage.to_dict() == {
            "tokens": 0,
            "compute": 0.0,
            "issue_writes": 0,
        }
    finally:
        provider.close()
        service.close()


def test_d2_offline_mode_refuses_provider_work_before_transport(tmp_path: Path) -> None:
    config = replace(_config(_fake_codex(tmp_path)), accounting_mode="offline")
    inspection = inspect_live_capabilities(config)
    assert inspection.status == "available"
    route = inspection.choose()
    assert route is not None
    provider = CodexAppServerProvider(config=config)
    try:
        with pytest.raises(AppServerUnavailable, match="offline"):
            provider.start(
                route=route,
                prompt="provider-free",
                evidence=[],
                reserved_compute=UNMEASURED_TURN_COMPUTE_CHARGE,
            )
        assert provider.transport.process is None
    finally:
        provider.close()


def test_d2_strict_ceiling_rejects_a_reservation_above_verified_cap(tmp_path: Path) -> None:
    config = replace(
        _config(_fake_codex(tmp_path)),
        accounting_mode="strict_provider_ceiling",
        require_explicit_route=True,
        required_route_id="provider-fixture:model-fixture",
        required_provider="provider-fixture",
        required_model_id="model-fixture",
        provider_ceiling_verified=True,
        provider_compute_ceiling=UNMEASURED_TURN_COMPUTE_CHARGE,
    )
    inspection = inspect_live_capabilities(config)
    route = inspection.choose()
    assert route is not None
    provider = CodexAppServerProvider(config=config)
    try:
        with pytest.raises(AppServerUnavailable, match="exceeds"):
            provider.start(
                route=route,
                prompt="too much reserved compute",
                evidence=[],
                reserved_compute=UNMEASURED_TURN_COMPUTE_CHARGE + 1.0,
            )
        assert provider.transport.process is None
    finally:
        provider.close()


def test_d2_local_accounting_records_provider_overspend_without_retry(tmp_path: Path) -> None:
    class OverrunProvider(CodexAppServerProvider):
        def start(self, *, route, prompt, evidence, reserved_compute=None):
            del prompt, evidence, reserved_compute
            self._check_route(route)
            return {
                "status": "complete",
                "provider_session_id": "overrun-thread",
                "message": "simulated provider overspend",
                "usage": {
                    "tokens": 20,
                    "compute": UNMEASURED_TURN_COMPUTE_CHARGE,
                    "issue_writes": 0,
                },
            }

    config = _config(_fake_codex(tmp_path))
    inspection = inspect_live_capabilities(config)
    route = inspection.choose()
    assert route is not None
    service, audit_session = _meter_setup(
        tmp_path,
        token_budget=7,
        compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
    )
    provider = OverrunProvider(config=config)
    client = AuditCodexClient(service, inspector=lambda: inspection, provider=provider)
    try:
        result = client.start(
            audit_session,
            prompt="record overspend",
            route_id=route.route_id,
            operation_id="d2-overspend",
            token_budget=7,
            compute_budget=UNMEASURED_TURN_COMPUTE_CHARGE,
        )
        assert result.status == "denied"
        assert result.receipt is not None
        assert result.receipt.overspent
        assert result.receipt.accounting_mode == "local_accounting"
        assert result.receipt.provider_ceiling_verified is False
        assert result.receipt.usage["tokens"] == 20
        assert audit_session.usage.to_dict() == {
            "tokens": 7,
            "compute": UNMEASURED_TURN_COMPUTE_CHARGE,
            "issue_writes": 0,
        }
    finally:
        provider.close()
        service.close()


@pytest.mark.parametrize(
    "invalid",
    [
        {"accounting_mode": "unknown"},
        {"require_explicit_route": 1},
        {"required_route_id": 7},
        {"required_provider": "provider with whitespace"},
        {"require_explicit_route": True},
        {"provider_ceiling_verified": 1},
        {
            "accounting_mode": "strict_provider_ceiling",
            "require_explicit_route": False,
            "provider_ceiling_verified": True,
            "provider_compute_ceiling": 1.0,
        },
    ],
)
def test_d2_accounting_config_rejects_ambiguous_route_or_cap_settings(
    invalid: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        CodexAppServerConfig(**invalid)


def test_d2_route_and_capability_evidence_mismatches_fail_before_transport(tmp_path: Path) -> None:
    executable = _fake_codex(tmp_path)
    inspection = inspect_live_capabilities(_config(executable))
    route = inspection.choose()
    assert route is not None

    provider = CodexAppServerProvider(config=_config(executable))
    try:
        for invalid_route in (
            replace(route, accounting_mode="offline"),
            replace(route, provider_ceiling_verified=True),
        ):
            with pytest.raises(AppServerUnavailable):
                provider._check_route(invalid_route)

        model_provider = CodexAppServerProvider(
            config=replace(_config(executable), required_model_id="model-fixture")
        )
        try:
            with pytest.raises(AppServerUnavailable, match="required_model_id"):
                model_provider._check_route(replace(route, model_id="other-model"))
        finally:
            model_provider.close()

        required_route = replace(
            _config(executable),
            required_route_id="other-provider:other-model",
        )
        required_inspection = inspect_live_capabilities(required_route)
        assert required_inspection.status == "unavailable"
        assert "required_route_id" in required_inspection.reason
    finally:
        provider.close()


def test_d2_strict_reservation_and_usage_annotation_keep_cap_boundary_explicit(
    tmp_path: Path,
) -> None:
    with pytest.raises(AppServerUnavailable, match="verified provider ceiling"):
        CodexAppServerProvider._require_turn_reservation(
            UNMEASURED_TURN_COMPUTE_CHARGE,
            accounting_mode="strict_provider_ceiling",
            provider_ceiling_verified=False,
            provider_compute_ceiling=None,
        )

    config = replace(_config(_fake_codex(tmp_path)), provider_compute_ceiling=3.0)
    provider = CodexAppServerProvider(config=config)
    try:
        usage = provider._annotate_accounting_usage({"tokens": 1, "compute": 2.0})
        assert usage["provider_compute_ceiling"] == 3.0
    finally:
        provider.close()
