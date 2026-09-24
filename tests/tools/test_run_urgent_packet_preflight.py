"""Focused fixture tests for the bounded urgent-packet batch preflight runner (#8847)."""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.tools import run_urgent_packet_preflight as tool

NOW = datetime(2026, 9, 12, 12, 0, 0, tzinfo=UTC)
REGISTRY_ID = "fixture-urgent-packets"

_OK = """
import json
print(json.dumps({"status": "ok", "argv": __import__("sys").argv[1:]}))
"""
_BLOCKED = """
import json, sys
print(json.dumps({"status": "blocked"}))
sys.exit(1)
"""
_FAIL = "import sys\nsys.exit(3)\n"
_MALFORMED = "print('not json at all')\n"
_TIMEOUT = "import time\ntime.sleep(30)\n"
_TAIL = """
import json
print("2026-09-12 00:00:00 | DEBUG | runtime:reg - registered sensor")
print(json.dumps({"status": "ok", "value": 1}, indent=2))
"""
_BIG = """
print("x" * 70000)
print('{"status": "ok"}')
"""
_SHARED = 'import json\nprint(json.dumps({"status": "ok"}))\n'


def _write_script(root: Path, name: str, body: str) -> Path:
    path = root / "scripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(
    packet_id: str,
    issue: int,
    script: Path | None,
    *,
    extra_argv: list[str] | None = None,
    contract: str = "json_object",
    timeout: float = 10.0,
    packet_sha256: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"packet_id": packet_id, "issue": issue}
    if script is not None:
        entry.update(
            preflight_argv=[
                sys.executable,
                str(script.relative_to(script.parents[1])),
                *(extra_argv or []),
            ],
            timeout_seconds=timeout,
            output_contract=contract,
            preflight_sha256=_sha256(script),
        )
    if packet_sha256 is not None:
        entry["packet_sha256"] = packet_sha256
    entry.update(extra)
    return entry


def _registry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": tool.REGISTRY_SCHEMA, "registry_id": REGISTRY_ID, "entries": entries}


def _run(root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    return tool.run_preflight(_registry(entries), repo_root=root, now=NOW)


def _rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["packet_id"]: row for row in report["rows"]}


def test_mixed_registry_classifies_sorts_and_counts(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    fail = _write_script(tmp_path, "fail.py", _FAIL)
    blocked = _write_script(tmp_path, "blocked.py", _BLOCKED)
    report = _run(
        tmp_path,
        [
            _entry("z-ready", 20, ok),
            _entry("a-failed", 10, fail),
            _entry("b-blocked", 15, blocked),
            _entry("c-unsupported", 30, None, unsupported_reason="requires_external_services"),
            _entry("d-resource", 31, ok, resource_projection={"path": "missing.json"}),
            _entry("e-stale", 32, ok, expires_at="2020-01-01T00:00:00Z"),
        ],
    )
    rows = _rows(report)
    assert [row["packet_id"] for row in report["rows"]] == [
        "a-failed",
        "b-blocked",
        "z-ready",
        "c-unsupported",
        "d-resource",
        "e-stale",
    ]
    assert rows["a-failed"]["classification"] == tool.FAILED
    assert rows["a-failed"]["first_blocker"] == "exit_status_3"
    assert rows["b-blocked"]["classification"] == tool.BLOCKED
    assert rows["b-blocked"]["first_blocker"] == "status_token_blocked"
    assert rows["z-ready"]["classification"] == tool.READY
    assert rows["z-ready"]["evidence"]["exit_status"] == 0
    assert rows["c-unsupported"]["classification"] == tool.UNSUPPORTED
    assert rows["d-resource"]["classification"] == tool.RESOURCE
    assert rows["e-stale"]["classification"] == tool.STALE
    assert report["summary"]["classification_counts"] == {
        tool.READY: 1,
        tool.FAILED: 1,
        tool.BLOCKED: 1,
        tool.STALE: 1,
        tool.DUPLICATE: 0,
        tool.RESOURCE: 1,
        tool.UNSUPPORTED: 1,
    }
    assert report["summary"]["action_required_count"] == 5
    assert all(row["compute_authority"] == "not_granted" for row in report["rows"])
    assert all(row["scientific_status"] == "not_evaluated" for row in report["rows"])


def test_shared_check_runs_once_and_preserves_per_packet_evidence(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    shared = _write_script(tmp_path, "shared.py", _SHARED)
    check = {
        "check_id": "staging-bundle",
        "argv": [sys.executable, "scripts/shared.py"],
        "timeout_seconds": 10,
        "output_contract": "json_object",
    }
    report = _run(
        tmp_path,
        [
            _entry("packet-a", 1, ok, packet_sha256="a" * 64, shared_check=check),
            _entry("packet-b", 1, ok, packet_sha256="b" * 64, shared_check=check),
        ],
    )
    assert report["summary"]["shared_check_count"] == 1
    assert report["shared_checks"][0]["status"] == "passed"
    assert report["shared_checks"][0]["check_id"] == "staging-bundle"
    for row in report["rows"]:
        assert row["classification"] == tool.READY
        assert row["shared_check_id"] == "staging-bundle"
        assert row["evidence"]["shared_check_status"] == "passed"
        assert row["evidence"]["exit_status"] == 0
    assert shared.is_file()


def test_failed_shared_check_blocks_only_its_packets(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    _write_script(tmp_path, "fail.py", _FAIL)
    check = {
        "check_id": "shared-fail",
        "argv": [sys.executable, "scripts/fail.py"],
        "timeout_seconds": 10,
        "output_contract": "json_object",
    }
    report = _run(
        tmp_path,
        [
            _entry("blocked-a", 1, ok, shared_check=check),
            _entry("free-b", 2, ok),
        ],
    )
    rows = _rows(report)
    assert rows["blocked-a"]["classification"] == tool.BLOCKED
    assert rows["blocked-a"]["first_blocker"] == "shared_check_not_passed:shared-fail"
    assert rows["free-b"]["classification"] == tool.READY
    assert report["shared_checks"][0]["status"] == "failed"


def test_conflicting_shared_check_registration_is_unsupported(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    shared = _write_script(tmp_path, "shared.py", _SHARED)
    first = {
        "check_id": "same-id",
        "argv": [sys.executable, "scripts/shared.py"],
        "timeout_seconds": 10,
        "output_contract": "json_object",
    }
    second = {**first, "argv": [sys.executable, "scripts/shared.py", "extra"]}
    report = _run(
        tmp_path,
        [
            _entry("a-first", 1, ok, shared_check=first),
            _entry("b-second", 1, ok, shared_check=second),
        ],
    )
    rows = _rows(report)
    assert rows["a-first"]["classification"] == tool.READY
    assert rows["b-second"]["classification"] == tool.UNSUPPORTED
    assert rows["b-second"]["first_blocker"] == "conflicting_shared_check"
    assert shared.is_file()


def test_identical_registrations_are_duplicate_active(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    packet = "a" * 64
    report = _run(
        tmp_path,
        [
            _entry("dup-b", 7, ok, packet_sha256=packet),
            _entry("dup-a", 7, ok, packet_sha256=packet),
        ],
    )
    rows = _rows(report)
    assert rows["dup-a"]["classification"] == tool.READY
    assert rows["dup-b"]["classification"] == tool.DUPLICATE
    assert rows["dup-b"]["duplicate_of"] == "dup-a"
    assert rows["dup-b"]["executed"] is False


def test_timeout_and_malformed_and_oversized_output_fail_closed(tmp_path: Path) -> None:
    timeout = _write_script(tmp_path, "timeout.py", _TIMEOUT)
    malformed = _write_script(tmp_path, "malformed.py", _MALFORMED)
    big = _write_script(tmp_path, "big.py", _BIG)
    report = _run(
        tmp_path,
        [
            _entry("timeout", 1, timeout, timeout=0.3),
            _entry("malformed", 2, malformed),
            _entry("oversized", 3, big),
        ],
    )
    rows = _rows(report)
    assert rows["timeout"]["classification"] == tool.FAILED
    assert rows["timeout"]["first_blocker"] == "timeout"
    assert rows["timeout"]["evidence"]["timed_out"] is True
    assert rows["malformed"]["classification"] == tool.FAILED
    assert rows["malformed"]["first_blocker"] == "malformed_output"
    assert rows["oversized"]["classification"] == tool.FAILED
    assert rows["oversized"]["first_blocker"] == "output_limit_exceeded"


def test_json_tail_object_accepts_log_prefixed_status(tmp_path: Path) -> None:
    tail = _write_script(tmp_path, "tail.py", _TAIL)
    report = _run(tmp_path, [_entry("tail", 1, tail, contract="json_tail_object")])
    row = _rows(report)["tail"]
    assert row["classification"] == tool.READY
    assert row["evidence"]["status_token"] == "ok"
    assert row["evidence"]["output_digest"]


def test_shell_tokens_are_rejected_without_side_effects(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    report = _run(
        tmp_path,
        [
            _entry("inject-inline", 1, ok, extra_argv=["a; touch pwned"]),
            _entry("inject-operator", 2, ok, extra_argv=["|"]),
        ],
    )
    rows = _rows(report)
    assert rows["inject-inline"]["classification"] == tool.UNSUPPORTED
    assert rows["inject-inline"]["first_blocker"] == "shell_metacharacter_in_argv"
    assert rows["inject-operator"]["classification"] == tool.UNSUPPORTED
    assert not (tmp_path / "pwned").exists()


def test_path_escape_missing_validator_and_unsupported_executable(tmp_path: Path) -> None:
    report = _run(
        tmp_path,
        [
            {
                "packet_id": "escape",
                "issue": 1,
                "preflight_argv": [sys.executable, "../outside.py"],
                "timeout_seconds": 10,
                "output_contract": "json_object",
                "preflight_sha256": "0" * 64,
            },
            {
                "packet_id": "missing",
                "issue": 2,
                "preflight_argv": [sys.executable, "scripts/absent.py"],
                "timeout_seconds": 10,
                "output_contract": "json_object",
                "preflight_sha256": "0" * 64,
            },
            {
                "packet_id": "shell",
                "issue": 3,
                "preflight_argv": ["bash", "-c", "true"],
                "timeout_seconds": 10,
                "output_contract": "json_object",
                "preflight_sha256": "0" * 64,
            },
        ],
    )
    rows = _rows(report)
    assert rows["escape"]["classification"] == tool.UNSUPPORTED
    assert rows["escape"]["first_blocker"] == "path_escape_in_argv"
    assert rows["missing"]["first_blocker"] == "missing_validator"
    assert rows["shell"]["first_blocker"] == "unsupported_executable"


def test_source_drift_projection_digest_and_registry_identity_fail_closed(
    tmp_path: Path,
) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    projection = tmp_path / "projection.json"
    projection.write_text('{"free": 1}', encoding="utf-8")
    report = _run(
        tmp_path,
        [
            _entry("drift", 1, ok, preflight_sha256="f" * 64),
            _entry(
                "projection-mismatch",
                2,
                ok,
                resource_projection={"path": "projection.json", "sha256": "e" * 64},
            ),
            _entry(
                "projection-ok",
                3,
                ok,
                resource_projection={
                    "path": "projection.json",
                    "sha256": _sha256(projection),
                },
            ),
        ],
    )
    rows = _rows(report)
    assert rows["drift"]["first_blocker"] == "preflight_source_drift"
    assert rows["projection-mismatch"]["first_blocker"] == "resource_projection_digest_mismatch"
    assert rows["projection-ok"]["classification"] == tool.READY


def test_registry_contract_errors_are_malformed(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    with pytest.raises(tool.RegistryError):
        tool.parse_registry({"schema": "wrong", "entries": []}, tmp_path)
    with pytest.raises(tool.RegistryError):
        tool.parse_registry(
            {"schema": tool.REGISTRY_SCHEMA, "registry_id": REGISTRY_ID, "entries": [], "x": 1},
            tmp_path,
        )
    with pytest.raises(tool.RegistryError):
        tool.run_preflight(
            _registry([_entry("dup", 1, ok), _entry("dup", 2, ok)]), repo_root=tmp_path, now=NOW
        )


def test_invalid_argv_sha_and_contract_are_unsupported(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    entry = _entry("bad-sha", 1, ok)
    entry["preflight_sha256"] = "not-a-digest"
    report = _run(tmp_path, [entry])
    assert _rows(report)["bad-sha"]["first_blocker"] == "missing_preflight_sha256"
    unknown = _entry("unknown-field", 2, ok)
    unknown["surprise"] = True
    report = _run(tmp_path, [unknown])
    assert _rows(report)["unknown-field"]["first_blocker"] == "unknown_entry_field"


def test_report_is_deterministic_apart_from_durations(tmp_path: Path) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    fail = _write_script(tmp_path, "fail.py", _FAIL)
    entries = [_entry("a", 1, ok), _entry("b", 2, fail, extra_argv=["--x"])]
    first = _run(tmp_path, entries)
    second = _run(tmp_path, entries)
    for report in (first, second):
        for row in report["rows"]:
            row["evidence"]["duration_seconds"] = None
        for shared in report["shared_checks"]:
            shared["duration_seconds"] = None
    assert tool.render_report_json(first) == tool.render_report_json(second)
    assert first["registry_sha256"] == second["registry_sha256"]


def test_checked_in_registry_is_well_formed() -> None:
    path = Path(tool.__file__).with_name("urgent_packet_registry.v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    registry_id, entries, findings = tool.parse_registry(payload, Path(tool.__file__).parents[2])
    assert registry_id == "compute-window-urgent-packets"
    assert len(entries) == 5
    assert findings == []
    unsupported = [entry for entry in entries if entry.unsupported_reason]
    assert len(unsupported) == 1
    assert all(entry.problem is None for entry in entries)


def test_cli_json_text_and_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ok = _write_script(tmp_path, "ok.py", _OK)
    fail = _write_script(tmp_path, "fail.py", _FAIL)
    mixed = tmp_path / "mixed.json"
    mixed.write_text(
        json.dumps(_registry([_entry("ok", 1, ok), _entry("fail", 2, fail)])), encoding="utf-8"
    )
    assert tool.main(["--check", "--registry", str(mixed), "--repo-root", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["action_required_count"] == 1

    ready = tmp_path / "ready.json"
    ready.write_text(json.dumps(_registry([_entry("ok", 1, ok)])), encoding="utf-8")
    assert tool.main(["--check", "--registry", str(ready), "--repo-root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert (
        tool.main(
            ["--check", "--registry", str(ready), "--repo-root", str(tmp_path), "--format", "text"]
        )
        == 0
    )
    assert "1 ready" in capsys.readouterr().out

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    assert tool.main(["--check", "--registry", str(malformed), "--repo-root", str(tmp_path)]) == 2
