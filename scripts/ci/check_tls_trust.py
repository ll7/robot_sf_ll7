#!/usr/bin/env python3
"""Fail CI unless real TLS clients reject both disclosed eMudhra chains.

Network errors are inconclusive failures. A trusted positive control distinguishes
an offline runner from one that actually rejects the compromised certificates.
"""

from __future__ import annotations

import concurrent.futures
import glob
import importlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import sysconfig

BAD_HOSTS = (
    "dv-g3b-emudhra.we-have-a-problem.com",
    "dv-g3b-aaa-emudhra.we-have-a-problem.com",
)
GOOD_HOST = "github.com"
TIMEOUT = 15


def python_probe(host: str, cafile: str | None) -> str:
    """Distinguish certificate rejection from transport failure using SSL types."""
    try:
        context = ssl.create_default_context(cafile=cafile)
        import http.client

        connection = http.client.HTTPSConnection(host, timeout=TIMEOUT, context=context)
        try:
            connection.connect()
        finally:
            connection.close()
        return "accepted"
    except ssl.SSLCertVerificationError:
        return "certificate-rejected"
    except (OSError, ValueError) as error:
        return f"inconclusive: {type(error).__name__}: {error}"


def command_probe(client: str, host: str) -> str:
    """Use client-specific certificate error codes, never generic failure codes."""
    url = f"https://{host}/"
    if client == "curl":
        command = [
            client,
            "--silent",
            "--show-error",
            "--output",
            os.devnull,
            "--max-time",
            str(TIMEOUT),
            url,
        ]
    elif client == "uv":
        command = [
            client,
            "--no-config",
            "pip",
            "compile",
            "--no-cache",
            "--no-deps",
            "--no-build",
            "--python",
            sys.executable,
            "--index-url",
            url,
            "-",
        ]
    else:
        code = """const https = require('https');
const req = https.get(process.argv[1], res => {res.destroy(); process.exit(0);});
req.setTimeout(12000, () => {req.destroy(); process.exit(3);});
req.on('error', e => {
  const certErrors = new Set(['CERT_REVOKED','CERT_SIGNATURE_FAILURE',
    'UNABLE_TO_GET_ISSUER_CERT_LOCALLY','UNABLE_TO_VERIFY_LEAF_SIGNATURE',
    'CERT_UNTRUSTED','SELF_SIGNED_CERT_IN_CHAIN','DEPTH_ZERO_SELF_SIGNED_CERT',
    'CERT_HAS_EXPIRED','ERR_TLS_CERT_ALTNAME_INVALID']);
  console.error(e.code); process.exit(certErrors.has(e.code) ? 60 : 3);
});"""
        command = [client, "-e", code, url]
    try:
        result = subprocess.run(
            command,
            input="certifi==2026.10.19\n",
            text=True,
            capture_output=True,
            timeout=TIMEOUT + 5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"inconclusive: {type(error).__name__}"
    if result.returncode == 0:
        return "accepted"
    if client == "uv":
        # uv/rustls exposes typed certificate failures in its error chain.
        if any(
            marker in result.stderr
            for marker in (
                "UnknownIssuer",
                "CertRevoked",
                "CertificateRevoked",
                "InvalidCertificate",
                "invalid peer certificate",
                "certificate verify failed",
            )
        ):
            return "certificate-rejected"
    elif result.returncode == 60:
        return "certificate-rejected"
    return f"inconclusive: exit {result.returncode}: {result.stderr[-500:].strip()}"


def main() -> int:
    """Print all client checks and reject any unexpected or inconclusive result."""
    probes = [("python-default", lambda host: python_probe(host, None))]
    for module in ("certifi", "pip._vendor.certifi"):
        try:
            cafile = importlib.import_module(module).where()
        except ImportError:
            continue
        probes.append((module, lambda host, path=cafile: python_probe(host, path)))
    clients = [name for name in ("curl", "node", "uv") if shutil.which(name)]
    clients += glob.glob("/opt/robot-sf-runner/externals/node[0-9][0-9]/bin/node")
    driver = "node.exe" if os.name == "nt" else "node"
    clients += glob.glob(f"{sysconfig.get_path('purelib')}/playwright/driver/{driver}")
    for client in clients:
        probes.append((client, lambda host, name=client: command_probe(name, host)))
    jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for name, probe in probes:
            # uv's positive control must be a package index, rather than a web page.
            good = "pypi.org/simple" if name == "uv" else GOOD_HOST
            for host in (*BAD_HOSTS, good):
                expected = "certificate-rejected" if host in BAD_HOSTS else "accepted"
                jobs.append((name, host, expected, pool.submit(probe, host)))
        rows = [
            {"client": name, "host": host, "expected": expected, "result": job.result()}
            for name, host, expected, job in jobs
        ]
    ok = all(row["result"] == row["expected"] for row in rows)
    print(json.dumps({"ok": ok, "checks": rows}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
