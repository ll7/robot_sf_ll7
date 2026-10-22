"""Security gate failures must distinguish bad certificates from an offline client."""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import ssl
import subprocess
import sysconfig
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "tls_trust", Path(__file__).resolve().parents[2] / "scripts/ci/check_tls_trust.py"
)
assert SPEC and SPEC.loader
trust = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trust)


@pytest.mark.parametrize(
    ("exit_code", "expected"),
    [
        (0, "accepted"),
        (60, "certificate-rejected"),
        (6, "inconclusive"),
        (28, "inconclusive"),
        (35, "inconclusive"),
    ],
)
def test_curl_transport_errors_do_not_count_as_protection(
    tmp_path, monkeypatch, exit_code, expected
):
    """Exercise real subprocess handling with curl's distinct failure codes."""
    client = tmp_path / "curl"
    client.write_text(f"#!/bin/sh\nexit {exit_code}\n")
    client.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert trust.command_probe("curl", trust.BAD_HOSTS[0]).startswith(expected)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ssl.SSLCertVerificationError("bad issuer"), "certificate-rejected"),
        (TimeoutError("offline"), "inconclusive"),
        (OSError("DNS failure"), "inconclusive"),
    ],
)
def test_python_only_certificate_errors_count_as_rejection(monkeypatch, error, expected):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(trust.ssl, "create_default_context", fail)
    assert trust.python_probe(trust.BAD_HOSTS[0], None).startswith(expected)


def test_positive_control_failure_blocks_gate(monkeypatch, capsys):
    """Rejecting every connection is not a successful security check."""
    monkeypatch.setattr(trust, "python_probe", lambda *args: "certificate-rejected")
    monkeypatch.setattr(trust.shutil, "which", lambda *args: None)
    monkeypatch.setattr(trust.glob, "glob", lambda *args: [])
    assert trust.main() == 1
    assert '"ok": false' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("exit_code", "stderr", "expected"),
    [
        (1, "invalid peer certificate: UnknownIssuer", "certificate-rejected"),
        (1, "invalid peer certificate: CertRevoked", "certificate-rejected"),
        (1, "dns error: failed to lookup address information", "inconclusive"),
        (1, "No solution found when resolving dependencies", "inconclusive"),
        (0, "", "accepted"),
    ],
)
def test_uv_transport_and_resolution_errors_do_not_count_as_protection(
    tmp_path, monkeypatch, exit_code, stderr, expected
):
    """Only typed certificate errors count, rather than uv's general failure exit."""
    client = tmp_path / "uv"
    client.write_text(f"#!/bin/sh\nprintf '%s\\n' '{stderr}' >&2\nexit {exit_code}\n")
    client.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert trust.command_probe("uv", trust.BAD_HOSTS[0]).startswith(expected)


def test_uv_timeout_is_inconclusive(monkeypatch):
    """An interrupted uv handshake must not qualify as certificate rejection."""

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(trust.subprocess, "run", timeout)
    assert trust.command_probe("uv", trust.BAD_HOSTS[0]).startswith("inconclusive")


@pytest.fixture(scope="module")
def node_runtime():
    """Use the installed Node runtime, including Playwright's private copy."""
    candidates = [
        shutil.which("node"),
        str(Path(sysconfig.get_path("purelib")) / "playwright/driver/node"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    pytest.skip("Node runtime required for actual HTTPS error-classification tests")


@pytest.fixture(params=["path", "embedded"])
def node_client(request, node_runtime, tmp_path, monkeypatch):
    """Exercise Node found by command name and by its private executable path."""
    monkeypatch.setenv("NODE_OPTIONS", "--use-openssl-ca")
    monkeypatch.delenv("NODE_EXTRA_CA_CERTS", raising=False)
    monkeypatch.delenv("NODE_TLS_REJECT_UNAUTHORIZED", raising=False)
    if request.param == "embedded":
        return str(node_runtime)
    (tmp_path / "node").symlink_to(node_runtime)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    return "node"


@pytest.fixture(scope="module")
def localhost_tls_server(tmp_path_factory):
    """Serve a generated localhost certificate without using external networking."""
    root = tmp_path_factory.mktemp("node-tls")
    config = root / "openssl.cnf"
    config.write_text(
        "[req]\ndistinguished_name=dn\nx509_extensions=server\nprompt=no\n"
        "[dn]\nCN=127.0.0.1\n[server]\nsubjectAltName=IP:127.0.0.1\n"
        "basicConstraints=critical,CA:TRUE\n"
    )
    certificate, key = root / "server.pem", root / "server.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-config",
            str(config),
            "-keyout",
            str(key),
            "-out",
            str(certificate),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    key.chmod(0o600)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"127.0.0.1:{server.server_port}", certificate
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_node_rejects_untrusted_certificate(node_client, localhost_tls_server):
    """The actual Node HTTPS snippet must report its certificate failure code."""
    host, _ = localhost_tls_server
    assert trust.command_probe(node_client, host) == "certificate-rejected"


def test_node_accepts_trusted_certificate(node_client, localhost_tls_server, monkeypatch):
    """The same certificate succeeds only when its CA is explicitly trusted."""
    host, certificate = localhost_tls_server
    monkeypatch.setenv("NODE_EXTRA_CA_CERTS", str(certificate))
    assert trust.command_probe(node_client, host) == "accepted"


def test_node_refused_connection_is_inconclusive(node_client):
    """A listening failure has a different outcome from certificate rejection."""
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        host = f"127.0.0.1:{reserved.getsockname()[1]}"
        assert trust.command_probe(node_client, host).startswith("inconclusive")
