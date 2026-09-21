# ruff: noqa: DOC201

"""Bounded Model Context Protocol (MCP) stdio transport for audit tools.

``audit_mcp`` owns the audit envelope and :class:`AuditMCPDispatcher` owns
the service policy.  This module only translates the MCP JSON-RPC wire
protocol to that existing dispatcher.  A bridge is provided for the Codex App
Server case: Codex starts a short-lived stdio child, while the child forwards
messages over a private Unix socket to the parent process that owns the
dispatcher and session token.

The transport is intentionally local and experimental.  It does not expose an
HTTP listener, persist credentials, or construct a second service authority.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import selectors
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from robot_sf.analysis_workbench.audit_mcp import (
    MAX_MCP_PAYLOAD_BYTES,
    MAX_MCP_REQUEST_BYTES,
    REDACTED_MCP_REQUEST_ID,
    AuditMCPRequest,
    AuditMCPResponse,
    _walk,
)
from robot_sf.analysis_workbench.audit_service import AuditValidationError

MCP_JSONRPC_VERSION = "2.0"
# Keep the newest MCP revision first while accepting revisions supported by
# current Codex clients.  The server always returns the negotiated value.
MCP_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MCP_SERVER_NAME = "robot-sf-audit"
MCP_SERVER_VERSION = "0.1.0"
MAX_MCP_STDIO_FRAME_BYTES = MAX_MCP_REQUEST_BYTES
MAX_MCP_STDIO_MESSAGES = 4_096
MAX_MCP_STDIO_LIFETIME_SECONDS = 600.0
MAX_MCP_STDIO_IDLE_SECONDS = 300.0
MAX_MCP_BRIDGE_CLIENTS = 4
MCP_SESSION_ID_ENV = "ROBOT_SF_AUDIT_MCP_SESSION_ID"
MCP_SESSION_TOKEN_ENV = "ROBOT_SF_AUDIT_MCP_SESSION_TOKEN"
MCP_ORIGIN_ENV = "ROBOT_SF_AUDIT_MCP_ORIGIN"

# The names deliberately match the closed operation vocabulary in
# AuditMCPDispatcher.  Aliases accepted by the dispatcher are omitted so a
# model sees one tool per authority operation.
AUDIT_MCP_TOOLS: tuple[str, ...] = (
    "read_campaign",
    "read_episode",
    "read_metrics",
    "read_events",
    "read_geometry",
    "read_signals",
    "related_cases",
    "read_queue",
    "next",
    "read_coverage",
    "run_native_diagnostic",
    "materialize_selected",
    "write_annotation",
    "write_reference",
    "write_finding",
    "sync_finding",
    "cancel",
)


class MCPStdioError(RuntimeError):
    """Base class for bounded MCP transport failures."""


class MCPProtocolError(MCPStdioError):
    """Raised when a JSON-RPC/MCP message violates the wire contract."""


class MCPTransportClosed(MCPStdioError):
    """Raised when a bridge or stdio stream closes unexpectedly."""


class MCPDispatcherLike(Protocol):
    """Small protocol accepted by the stdio server and bridge."""

    def dispatch(self, request: AuditMCPRequest | Mapping[str, Any]) -> AuditMCPResponse:
        """Dispatch an authenticated audit request."""


def _json_size(value: Any, *, limit: int, name: str) -> bytes:
    """Serialize strict JSON while enforcing structural and byte bounds."""

    try:
        _walk(value)
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (AuditValidationError, TypeError, ValueError, UnicodeError) as exc:
        raise MCPProtocolError(f"{name} is not bounded strict JSON: {exc}") from exc
    if len(encoded) > limit:
        raise MCPProtocolError(f"{name} exceeds {limit} bytes")
    return encoded


def _mapping_message(value: Any) -> dict[str, Any]:
    """Validate one inbound JSON-RPC object and return a detached mapping."""

    if not isinstance(value, Mapping):
        raise MCPProtocolError("MCP message must be an object")
    copied = dict(value)
    _json_size(copied, limit=MAX_MCP_STDIO_FRAME_BYTES, name="MCP message")
    if copied.get("jsonrpc") != MCP_JSONRPC_VERSION:
        raise MCPProtocolError("MCP message must use JSON-RPC 2.0")
    method = copied.get("method")
    if not isinstance(method, str) or not method.strip() or len(method) > 256:
        raise MCPProtocolError("MCP method must be a bounded string")
    if "id" in copied:
        request_id = copied["id"]
        if (
            request_id is None
            or isinstance(request_id, bool)
            or not isinstance(request_id, (str, int))
        ):
            raise MCPProtocolError("MCP request id must be a string or integer")
        if isinstance(request_id, str) and len(request_id) > 256:
            raise MCPProtocolError("MCP request id exceeds the text limit")
    params = copied.get("params", {})
    if not isinstance(params, Mapping):
        raise MCPProtocolError("MCP params must be an object")
    _json_size(dict(params), limit=MAX_MCP_PAYLOAD_BYTES, name="MCP params")
    return copied


def _request_id(message: Mapping[str, Any]) -> Any:
    """Return a response id, or ``None`` for notifications."""

    return message.get("id") if "id" in message else None


def _response(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    """Build a bounded JSON-RPC success response."""

    output = {"jsonrpc": MCP_JSONRPC_VERSION, "id": request_id, "result": dict(result)}
    _json_size(output, limit=MAX_MCP_STDIO_FRAME_BYTES, name="MCP response")
    return output


def _error(request_id: Any, code: int, message: str, *, data: Any = None) -> dict[str, Any]:
    """Build a bounded JSON-RPC error response without leaking credentials."""

    error: dict[str, Any] = {"code": code, "message": str(message)[:1_024]}
    if data is not None:
        error["data"] = data
    output = {"jsonrpc": MCP_JSONRPC_VERSION, "id": request_id, "error": error}
    _json_size(output, limit=MAX_MCP_STDIO_FRAME_BYTES, name="MCP error response")
    return output


def _tool_schema(name: str) -> dict[str, Any]:
    """Return a conservative input schema for one dispatcher operation."""

    required = {
        "read_episode",
        "read_metrics",
        "read_events",
        "read_geometry",
        "related_cases",
    }
    properties: dict[str, Any] = {
        "episode_id": {"type": "string"},
        "finding_id": {"type": "string"},
        "context": {"type": "object"},
        "operation_id": {"type": "string", "minLength": 1, "maxLength": 4096},
        "record": {"type": "object"},
        "expected_revision": {"type": ["integer", "null"]},
        "expected_finding_revision": {"type": "integer", "minimum": 0},
        "expected_source_revision": {"type": ["integer", "string", "null"]},
        "expected_context_revision": {"type": ["integer", "null"], "minimum": 0},
        "expected_queue_state_revision": {"type": ["integer", "null"], "minimum": 0},
        "expected_queue_input_revision": {"type": ["integer", "null"], "minimum": 0},
        "force_current": {"type": "boolean"},
        "limit": {"type": "integer", "minimum": 0},
        "mode": {"type": "string"},
        "reason": {"type": "string"},
        "config": {"type": ["object", "null"]},
        "detector_ids": {"type": ["array", "null"], "items": {"type": "string"}},
        "repository": {"type": "string", "minLength": 1},
        "evidence": {"type": ["object", "null"]},
        "retry_ambiguous": {"type": "boolean"},
        "worker_id": {"type": "string"},
        "recipe_id": {"type": ["string", "null"]},
        "output_root": {"type": "string"},
        "output_directory": {"type": ["string", "null"]},
        "render_config": {"type": ["object", "null"]},
        "intervention_id": {"type": "string"},
        "robot_goal": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
        "activation_epsilon_m": {"type": "number", "minimum": 0, "maximum": 1},
        "deadline_s": {"type": "number", "exclusiveMinimum": 0},
    }
    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        # Dispatcher and service perform the authoritative closed validation;
        # this keeps forward-compatible payloads from being silently dropped.
        "additionalProperties": True,
    }
    if name in required:
        result["required"] = ["episode_id"]
    if name in {"write_annotation", "write_reference", "write_finding"}:
        result["required"] = ["record"]
    if name == "sync_finding":
        result["required"] = ["finding_id", "repository", "expected_finding_revision"]
    if name == "materialize_selected":
        result["required"] = ["output_root"]
    if name == "run_native_diagnostic":
        result["required"] = ["intervention_id", "robot_goal"]
        result["additionalProperties"] = False
    return result


def _tool_descriptors() -> list[dict[str, Any]]:
    """Build deterministic MCP tool descriptors from the closed vocabulary."""

    return [
        {
            "name": name,
            "title": f"Audit {name.replace('_', ' ')}",
            "description": (
                "Read or update one scoped Robot SF audit operation. "
                "The audit service enforces session, source, context, and budget policy."
            ),
            "inputSchema": _tool_schema(name),
        }
        for name in AUDIT_MCP_TOOLS
    ]


class AuditMCPStdioServer:
    """Translate MCP JSON-RPC requests into one authenticated dispatcher."""

    def __init__(
        self,
        dispatcher: MCPDispatcherLike,
        *,
        session_id: str,
        session_token: str,
        origin: str = "http://127.0.0.1",
        server_name: str = MCP_SERVER_NAME,
        server_version: str = MCP_SERVER_VERSION,
        supported_protocols: Sequence[str] = MCP_PROTOCOL_VERSIONS,
    ) -> None:
        """Bind the server to a service dispatcher and opaque session token."""

        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if not isinstance(session_token, str) or not session_token.strip():
            raise ValueError("session_token must be non-empty")
        if not isinstance(origin, str) or not origin.strip():
            raise ValueError("origin must be non-empty")
        protocols = tuple(protocol for protocol in supported_protocols if isinstance(protocol, str))
        if not protocols:
            raise ValueError("at least one MCP protocol version is required")
        self.dispatcher = dispatcher
        self.session_id = session_id
        self._session_token = session_token
        self.origin = origin
        self.server_name = server_name
        self.server_version = server_version
        self.supported_protocols = protocols
        self.initialized = False
        self.protocol_version: str | None = None
        self.message_count = 0

    def _authenticated_request(
        self, request_id: Any, name: str, arguments: Mapping[str, Any]
    ) -> AuditMCPResponse:
        """Construct the closed audit envelope without trusting tool arguments."""

        if name not in AUDIT_MCP_TOOLS:
            raise MCPProtocolError(f"unknown audit tool: {name}")
        # The session id/token/origin are server-owned values.  A model cannot
        # replace them by putting same-named keys in the tool arguments.
        envelope = AuditMCPRequest(
            request_id=str(request_id),
            session_id=self.session_id,
            session_token=self._session_token,
            origin=self.origin,
            operation=name,
            payload=dict(arguments),
        )
        return self.dispatcher.dispatch(envelope)

    def _safe_response_id(self, request_id: Any) -> Any:
        """Redact bearer-containing JSON-RPC IDs before response serialization."""

        if isinstance(request_id, str) and self._session_token in request_id:
            return REDACTED_MCP_REQUEST_ID
        return request_id

    @staticmethod
    def _tool_result(response: AuditMCPResponse) -> dict[str, Any]:
        """Map the shared audit response to an MCP ``tools/call`` result."""

        safe = response.to_dict()
        result = {
            "content": [{"type": "text", "text": json.dumps(safe, sort_keys=True)}],
            "structuredContent": safe,
            "isError": response.status in {"denied", "failed", "conflict", "cancelled"},
        }
        _json_size(result, limit=MAX_MCP_STDIO_FRAME_BYTES, name="MCP tool result")
        return result

    def handle_message(self, raw: Mapping[str, Any]) -> dict[str, Any] | None:  # noqa: C901, PLR0912
        """Handle one decoded JSON-RPC message; notifications return ``None``."""

        message = _mapping_message(raw)
        self.message_count += 1
        if self.message_count > MAX_MCP_STDIO_MESSAGES:
            raise MCPTransportClosed("MCP message limit reached")
        method = message["method"]
        request_id = _request_id(message)
        response_id = self._safe_response_id(request_id)
        params = dict(message.get("params", {}))

        if method == "initialize":
            if request_id is None:
                return _error(None, -32600, "initialize must be a request")
            if self.initialized:
                return _error(response_id, -32600, "MCP server is already initialized")
            client_info = params.get("clientInfo")
            if not isinstance(client_info, Mapping):
                return _error(response_id, -32602, "initialize.clientInfo is required")
            if not isinstance(client_info.get("name"), str) or not isinstance(
                client_info.get("version"), str
            ):
                return _error(response_id, -32602, "initialize.clientInfo is incomplete")
            requested = params.get("protocolVersion")
            if requested is not None and not isinstance(requested, str):
                return _error(response_id, -32602, "protocolVersion must be a string")
            if requested in self.supported_protocols:
                selected = requested
            else:
                selected = self.supported_protocols[0]
            self.protocol_version = selected
            self.initialized = True
            return _response(
                response_id,
                {
                    "protocolVersion": selected,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.server_name, "version": self.server_version},
                },
            )

        if method == "notifications/initialized":
            # Notifications have no response.  Treat an early notification as
            # a protocol error only when a caller asks for a response id.
            if not self.initialized:
                if request_id is not None:
                    return _error(response_id, -32002, "MCP server is not initialized")
                return None
            return None

        if not self.initialized:
            if request_id is None:
                return None
            return _error(response_id, -32002, "MCP server is not initialized")

        if method == "ping":
            return _response(response_id, {}) if request_id is not None else None
        if method == "tools/list":
            if request_id is None:
                return None
            return _response(response_id, {"tools": _tool_descriptors(), "nextCursor": None})
        if method == "tools/call":
            if request_id is None:
                return None
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str) or not name.strip():
                return _error(response_id, -32602, "tools/call.name is required")
            if not isinstance(arguments, Mapping):
                return _error(response_id, -32602, "tools/call.arguments must be an object")
            try:
                audit_response = self._authenticated_request(request_id, name, arguments)
                return _response(response_id, self._tool_result(audit_response))
            except (MCPProtocolError, AuditValidationError, TypeError, ValueError) as exc:
                return _error(response_id, -32602, str(exc))
            except Exception as exc:  # noqa: BLE001 - fail closed at the protocol edge.
                return _error(response_id, -32603, f"audit tool dispatch failed: {exc}")
        if request_id is None:
            return None
        return _error(response_id, -32601, f"MCP method is not supported: {method}")

    def run_stdio(  # noqa: C901, PLR0912
        self,
        stdin: Any = None,
        stdout: Any = None,
        *,
        max_frame_bytes: int = MAX_MCP_STDIO_FRAME_BYTES,
        max_messages: int = MAX_MCP_STDIO_MESSAGES,
        lifetime_seconds: float = MAX_MCP_STDIO_LIFETIME_SECONDS,
        idle_seconds: float = MAX_MCP_STDIO_IDLE_SECONDS,
    ) -> int:
        """Serve newline-delimited MCP JSON until EOF or a bound is reached."""

        input_stream = stdin if stdin is not None else sys.stdin.buffer
        output_stream = stdout if stdout is not None else sys.stdout.buffer
        started = time.monotonic()
        last_message = started
        selector: selectors.BaseSelector | None = None
        # Real file descriptors receive an idle/lifetime bound.  StringIO and
        # BytesIO test streams use the same frame checks without select().
        try:
            try:
                selector = selectors.DefaultSelector()
                selector.register(input_stream, selectors.EVENT_READ)
            except (AttributeError, OSError, ValueError):
                if selector is not None:
                    selector.close()
                selector = None
            while True:
                now = time.monotonic()
                if now - started >= lifetime_seconds:
                    return 0
                if selector is not None:
                    remaining = min(lifetime_seconds - (now - started), idle_seconds)
                    ready = selector.select(max(0.0, remaining))
                    if not ready:
                        return 0
                line = input_stream.readline(max_frame_bytes + 1)
                if not line:
                    return 0
                if isinstance(line, str):
                    raw_line = line.encode("utf-8")
                else:
                    raw_line = bytes(line)
                if len(raw_line) > max_frame_bytes:
                    response = _error(None, -32600, "MCP frame exceeds the byte limit")
                    self._write_response(output_stream, response)
                    return 2
                raw_line = raw_line.rstrip(b"\r\n")
                if not raw_line:
                    continue
                try:
                    decoded = json.loads(raw_line.decode("utf-8"))
                    response = self.handle_message(decoded)
                except (json.JSONDecodeError, UnicodeDecodeError, MCPProtocolError) as exc:
                    response = _error(None, -32700, str(exc))
                    self._write_response(output_stream, response)
                    return 2
                last_message = time.monotonic()
                if response is not None:
                    self._write_response(output_stream, response)
                if self.message_count >= max_messages:
                    return 0
                if last_message - started >= lifetime_seconds:
                    return 0
        finally:
            if selector is not None:
                selector.close()

    @staticmethod
    def _write_response(output_stream: Any, response: Mapping[str, Any]) -> None:
        """Write one bounded JSONL response and flush it immediately."""

        encoded = _json_size(response, limit=MAX_MCP_STDIO_FRAME_BYTES, name="MCP response")
        try:
            output_stream.write(encoded + b"\n")
        except TypeError:
            # Text streams are useful for embedding and deterministic tests;
            # subprocess stdio remains byte-oriented.
            output_stream.write((encoded + b"\n").decode("utf-8"))
        if hasattr(output_stream, "flush"):
            output_stream.flush()


class AuditMCPBridge:
    """Private Unix-socket bridge for an external Codex MCP child."""

    def __init__(
        self,
        dispatcher: MCPDispatcherLike,
        *,
        session_id: str,
        session_token: str,
        origin: str = "http://127.0.0.1",
        socket_path: str | Path | None = None,
        max_clients: int = MAX_MCP_BRIDGE_CLIENTS,
    ) -> None:
        """Create a bridge whose socket is private to this audit process."""

        self.dispatcher = dispatcher
        self.session_id = session_id
        self.session_token = session_token
        self.origin = origin
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        if socket_path is None:
            self._temporary_directory = tempfile.TemporaryDirectory(prefix="robot-sf-audit-mcp-")
            socket_path = Path(self._temporary_directory.name) / "bridge.sock"
        self.socket_path = Path(socket_path)
        if len(str(self.socket_path).encode()) >= 104:
            raise ValueError("Unix socket path exceeds the platform limit")
        self.max_clients = max(1, int(max_clients))
        self._server_socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._client_threads: set[threading.Thread] = set()
        self._client_sockets: set[socket.socket] = set()
        self._client_lock = threading.Lock()

    @property
    def running(self) -> bool:
        """Return whether the bridge listener is active."""

        return self._thread is not None and self._thread.is_alive()

    def start(self) -> AuditMCPBridge:
        """Bind the private socket and start its bounded accept loop."""

        if self.running:
            return self
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            raise MCPStdioError(f"MCP bridge socket path already exists: {self.socket_path}")
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        server_socket.listen(self.max_clients)
        server_socket.settimeout(0.2)
        self._server_socket = server_socket
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._accept_loop, daemon=True, name="audit-mcp-bridge"
        )
        self._thread.start()
        return self

    def command(self) -> tuple[str, ...]:
        """Return the child command used by an App Server MCP config."""

        return (
            sys.executable,
            "-m",
            "robot_sf.analysis_workbench.audit_mcp_stdio",
            "--bridge-socket",
            str(self.socket_path),
        )

    def environment(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return process environment metadata without exposing token in args."""

        result = dict(base or os.environ)
        result[MCP_SESSION_ID_ENV] = self.session_id
        # The proxy does not need this value, but retaining it in the child
        # makes direct server mode possible and keeps the context explicit.
        result[MCP_SESSION_TOKEN_ENV] = self.session_token
        result[MCP_ORIGIN_ENV] = self.origin
        return result

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            server_socket = self._server_socket
            if server_socket is None:
                return
            try:
                client, _ = server_socket.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._client_lock:
                active = sum(thread.is_alive() for thread in self._client_threads)
                if active >= self.max_clients:
                    client.close()
                    continue
                thread = threading.Thread(
                    target=self._serve_client,
                    args=(client,),
                    daemon=True,
                    name="audit-mcp-bridge-client",
                )
                self._client_threads.add(thread)
                self._client_sockets.add(client)
                thread.start()

    def _serve_client(self, client: socket.socket) -> None:
        server = AuditMCPStdioServer(
            self.dispatcher,
            session_id=self.session_id,
            session_token=self.session_token,
            origin=self.origin,
        )
        started = time.monotonic()
        try:
            with client.makefile("rb") as input_stream, client.makefile("wb") as output_stream:
                while not self._stop.is_set():
                    remaining = MAX_MCP_STDIO_LIFETIME_SECONDS - (time.monotonic() - started)
                    if remaining <= 0:
                        return
                    client.settimeout(min(MAX_MCP_STDIO_IDLE_SECONDS, remaining))
                    try:
                        line = input_stream.readline(MAX_MCP_STDIO_FRAME_BYTES + 1)
                    except (OSError, TimeoutError):
                        return
                    if not line:
                        return
                    if len(line) > MAX_MCP_STDIO_FRAME_BYTES:
                        response = _error(None, -32600, "MCP frame exceeds the byte limit")
                        output_stream.write(json.dumps(response).encode() + b"\n")
                        output_stream.flush()
                        return
                    try:
                        message = json.loads(line.rstrip(b"\r\n").decode("utf-8"))
                        response = server.handle_message(message)
                    except (json.JSONDecodeError, UnicodeDecodeError, MCPProtocolError) as exc:
                        response = _error(None, -32700, str(exc))
                        output_stream.write(json.dumps(response).encode() + b"\n")
                        output_stream.flush()
                        return
                    if response is not None:
                        output_stream.write(
                            _json_size(
                                response,
                                limit=MAX_MCP_STDIO_FRAME_BYTES,
                                name="MCP bridge response",
                            )
                            + b"\n"
                        )
                        output_stream.flush()
        finally:
            try:
                client.close()
            except OSError:
                pass
            with self._client_lock:
                self._client_threads.discard(threading.current_thread())
                self._client_sockets.discard(client)

    def close(self) -> None:  # noqa: C901
        """Stop the listener and remove only the task-owned socket."""

        self._stop.set()
        server_socket = self._server_socket
        self._server_socket = None
        if server_socket is not None:
            try:
                server_socket.close()
            except OSError:
                pass
        with self._client_lock:
            client_sockets = tuple(self._client_sockets)
        for client in client_sockets:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        with self._client_lock:
            client_threads = tuple(self._client_threads)
        for client_thread in client_threads:
            if client_thread is not threading.current_thread():
                client_thread.join(timeout=0.2)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __enter__(self) -> AuditMCPBridge:
        """Start the bridge for a context-managed scope."""

        return self.start()

    def __exit__(self, *_: Any) -> None:
        """Close the bridge at context exit."""

        self.close()


def _proxy_stdio(socket_path: str | Path) -> int:
    """Proxy one MCP stdio connection to a private bridge socket."""

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(MAX_MCP_STDIO_IDLE_SECONDS)
        sock.connect(str(socket_path))
        stdin_buffer = sys.stdin.buffer
        # BufferedReader may prefetch a notification and a request together;
        # selecting its raw descriptor avoids losing readiness for the second
        # frame while preserving the idle/lifetime bound.
        stdin = getattr(stdin_buffer, "raw", stdin_buffer)
        with (
            sock.makefile("rwb") as bridge_in,
            sys.stdout.buffer as stdout,
        ):
            count = 0
            started = time.monotonic()
            while (
                count < MAX_MCP_STDIO_MESSAGES
                and time.monotonic() - started < MAX_MCP_STDIO_LIFETIME_SECONDS
            ):
                remaining = min(
                    MAX_MCP_STDIO_LIFETIME_SECONDS - (time.monotonic() - started),
                    MAX_MCP_STDIO_IDLE_SECONDS,
                )
                ready, _, _ = select.select([stdin], [], [], max(0.0, remaining))
                if not ready:
                    return 0
                line = stdin.readline(MAX_MCP_STDIO_FRAME_BYTES + 1)
                if not line:
                    return 0
                if len(line) > MAX_MCP_STDIO_FRAME_BYTES:
                    return 2
                try:
                    message = _mapping_message(json.loads(line.rstrip(b"\r\n").decode("utf-8")))
                except (json.JSONDecodeError, UnicodeDecodeError, MCPProtocolError):
                    return 2
                bridge_in.write(
                    _json_size(message, limit=MAX_MCP_STDIO_FRAME_BYTES, name="MCP proxy message")
                    + b"\n"
                )
                bridge_in.flush()
                count += 1
                if "id" in message:
                    response = bridge_in.readline(MAX_MCP_STDIO_FRAME_BYTES + 1)
                    if not response or len(response) > MAX_MCP_STDIO_FRAME_BYTES:
                        return 2
                    stdout.write(response)
                    stdout.flush()
            return 0
    except (OSError, MCPTransportClosed):
        return 2
    finally:
        try:
            sock.close()
        except OSError:
            pass


def _direct_server_from_environment(
    dispatcher: MCPDispatcherLike | None = None,
) -> AuditMCPStdioServer:
    """Build direct mode from explicit environment context for embedding/tests."""

    if dispatcher is None:
        raise RuntimeError("direct MCP server mode requires an injected dispatcher")
    return AuditMCPStdioServer(
        dispatcher,
        session_id=os.environ.get(MCP_SESSION_ID_ENV, ""),
        session_token=os.environ.get(MCP_SESSION_TOKEN_ENV, ""),
        origin=os.environ.get(MCP_ORIGIN_ENV, "http://127.0.0.1"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the external bridge proxy used by Codex App Server."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-socket", required=True)
    args = parser.parse_args(argv)
    return _proxy_stdio(args.bridge_socket)


__all__ = [
    "AUDIT_MCP_TOOLS",
    "MCP_JSONRPC_VERSION",
    "MCP_ORIGIN_ENV",
    "MCP_PROTOCOL_VERSIONS",
    "MCP_SESSION_ID_ENV",
    "MCP_SESSION_TOKEN_ENV",
    "AuditMCPBridge",
    "AuditMCPStdioServer",
    "MCPDispatcherLike",
    "MCPProtocolError",
    "MCPStdioError",
    "MCPTransportClosed",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests.
    raise SystemExit(main())
