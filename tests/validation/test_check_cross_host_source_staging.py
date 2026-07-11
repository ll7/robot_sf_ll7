"""Tests for the queued-worker cross-host source-staging preflight."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/validation/check_cross_host_source_staging.py"

_SPEC = importlib.util.spec_from_file_location("_cross_host_source_staging", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _manifest() -> dict:
    return {
        "schema_version": _MODULE.SCHEMA_VERSION,
        "sources": [
            {"relative_path": "reports/one.csv", "sha256": _digest("one")},
            {"relative_path": "reports/two.csv", "sha256": _digest("two")},
        ],
    }


def _runner(stdout: str, returncode: int = 0, stderr: str = ""):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr=stderr)

    return run


def test_matching_remote_digests_allow_dispatch() -> None:
    """Every reachable, readable source with its pinned digest opens the gate."""
    manifest = _manifest()
    root = Path("/srv/staging")
    stdout = "\n".join(
        f"ok\t{source['sha256']}\t{root / source['relative_path']}"
        for source in manifest["sources"]
    )

    report = _MODULE.check_worker_staging(manifest, "worker.example", root, _runner(stdout))

    assert report["status"] == "ready"
    assert report["verified_source_count"] == 2
    assert report["dispatch_allowed"] is True


def test_missing_and_changed_remote_sources_block_dispatch() -> None:
    """Missing and checksum-drifted sources each fail closed."""
    manifest = _manifest()
    root = Path("/srv/staging")
    stdout = "\n".join(
        [
            f"missing\t{root / 'reports/one.csv'}",
            f"ok\t{_digest('changed')}\t{root / 'reports/two.csv'}",
        ]
    )

    report = _MODULE.check_worker_staging(manifest, "worker.example", root, _runner(stdout))

    assert report["status"] == "blocked"
    assert report["dispatch_allowed"] is False
    assert report["blockers"] == ["source_checksum_mismatch", "source_missing"]


def test_unreachable_worker_blocks_without_trusting_partial_output() -> None:
    """A failed SSH probe cannot be mistaken for partial source verification."""
    report = _MODULE.check_worker_staging(
        _manifest(),
        "worker.example",
        Path("/srv/staging"),
        _runner("ok\t" + _digest("one"), returncode=255, stderr="No route to host"),
    )

    assert report["status"] == "blocked"
    assert report["verified_source_count"] == 0
    assert report["blockers"] == ["worker_unreachable_or_probe_failed"]


@pytest.mark.parametrize(
    "relative_path",
    ["/absolute.csv", "reports/../escape.csv", "reports\\windows.csv", "reports/line\nbreak.csv"],
)
def test_manifest_rejects_nonportable_or_escaping_paths(relative_path: str) -> None:
    """Tracked manifests cannot encode platform-specific or escaping paths."""
    manifest = _manifest()
    manifest["sources"][0]["relative_path"] = relative_path

    with pytest.raises(_MODULE.ContractError):
        _MODULE.parse_manifest(manifest)


def test_probe_command_quotes_paths_without_shell_expansion() -> None:
    """Remote path text is safely quoted before it reaches the worker shell."""
    command = _MODULE.build_remote_probe(["/srv/staging/reports/$(unsafe) file.csv"])

    assert "'" in command
    assert "$(unsafe)" in command


def test_cli_reports_malformed_manifest_without_attempting_ssh(tmp_path: Path) -> None:
    """Malformed private inputs fail locally before any worker contact occurs."""
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps({"schema_version": "wrong", "sources": []}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--worker-host",
            "worker.example",
            "--staging-root",
            "/srv/staging",
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    payload = json.loads(completed.stdout)
    assert payload["status"] == "malformed"
