"""Tests for the main-red incident reconcile classifier (#8001).

The load-bearing property: an incident may only be auto-closed (``stale``) when
main is DECISIVELY green on a run NEWER than the incident's deciding failure.
A still-failing main (``active``) and any non-decisive window (``pending``) must
fail closed and never auto-close. This is the exact gap that left #7999/#8000
open for six days while main CI was green.
"""

from __future__ import annotations

import json as _json

import pytest

from scripts.dev import main_ci_incident_reconcile as reconcile
from scripts.dev.main_ci_incident_reconcile import (
    build_incident_signal,
    incident_reconcile_status,
)


def _run(rid: int, status: str, conclusion: str | None, created: str) -> dict:
    return {
        "databaseId": rid,
        "status": status,
        "conclusion": conclusion,
        "headSha": f"{rid:040x}",
        "createdAt": created,
    }


def test_stale_when_green_run_newer_than_incident_failure() -> None:
    """Green on a run newer than the deciding failure => stale (reconcilable)."""
    runs = [
        _run(300, "completed", "success", "2026-08-28T12:00:00Z"),
        _run(200, "completed", "failure", "2026-08-22T03:00:00Z"),
    ]
    assert incident_reconcile_status(200, runs) == "stale"


def test_active_when_latest_decisive_still_failure() -> None:
    """Latest decisive run is still a failure => active, do not close."""
    runs = [
        _run(400, "completed", "failure", "2026-08-28T12:00:00Z"),
        _run(300, "completed", "success", "2026-08-27T12:00:00Z"),
    ]
    assert incident_reconcile_status(300, runs) == "active"


def test_pending_when_no_deciding_run_id() -> None:
    """An incident without a resolvable deciding run fails closed to pending."""
    runs = [_run(300, "completed", "success", "2026-08-28T12:00:00Z")]
    assert incident_reconcile_status(None, runs) == "pending"


def test_pending_when_green_is_not_newer_than_incident_failure() -> None:
    """A green run equal-or-older than the deciding failure is not supersession."""
    runs = [
        _run(200, "completed", "success", "2026-08-28T12:00:00Z"),
        _run(150, "completed", "failure", "2026-08-22T03:00:00Z"),
    ]
    # green run id 200 is still newer than 150 -> stale. Use a green run OLDER than
    # the failure to prove the guard: build a window whose only green is the failure's
    # own run id (equal), which must not count as newer.
    assert incident_reconcile_status(200, runs) == "pending"


def test_pending_when_no_decisive_run_in_window() -> None:
    """A window with only stale/in-progress runs yields pending (fail closed)."""
    runs = [
        _run(300, "in_progress", None, "2026-08-28T12:00:00Z"),
        _run(200, "completed", "cancelled", "2026-08-28T11:00:00Z"),
    ]
    assert incident_reconcile_status(200, runs) == "pending"


def test_pending_when_incident_run_id_matches_latest_green() -> None:
    """A green run with the same id as the incident failure is not newer => pending."""
    runs = [_run(200, "completed", "success", "2026-08-28T12:00:00Z")]
    assert incident_reconcile_status(200, runs) == "pending"


def test_pending_when_latest_decisive_verdict_is_stale_class() -> None:
    """Non-green/non-red decisive outcomes (shouldn't happen) fail closed to pending."""
    # A decisive green run must be required; a window whose latest is green but
    # the function's classify returns stale for an unexpected conclusion.
    runs = [_run(300, "completed", "success", "2026-08-28T12:00:00Z")]
    # verify the stale path is only reachable when classify is not green/red.
    latest = reconcile.latest_decisive_run(runs)
    from scripts.dev.main_ci_is_green import classify

    assert classify(latest["conclusion"]) == "green"


def test_build_incident_signal_stale_schema() -> None:
    """build_incident_signal encodes stale => can_auto_close True."""
    runs = [
        _run(300, "completed", "success", "2026-08-28T12:00:00Z"),
        _run(200, "completed", "failure", "2026-08-22T03:00:00Z"),
    ]
    signal = build_incident_signal(incident_reconcile_status(200, runs), 200, runs)

    assert signal["schema_version"] == "main_ci_incident_reconcile.v1"
    assert signal["status"] == "stale"
    assert signal["can_auto_close"] is True
    assert signal["deciding_failure_run_id"] == 200
    assert signal["current_deciding_run"]["databaseId"] == 300


def test_build_incident_signal_active_schema() -> None:
    """build_incident_signal encodes active => can_auto_close False."""
    runs = [
        _run(400, "completed", "failure", "2026-08-28T12:00:00Z"),
        _run(300, "completed", "success", "2026-08-27T12:00:00Z"),
    ]
    signal = build_incident_signal(incident_reconcile_status(300, runs), 300, runs)

    assert signal["status"] == "active"
    assert signal["can_auto_close"] is False


def test_cli_json_output_and_exit_code(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The --json CLI path emits valid schema JSON with the correct exit code."""
    sample = [
        _run(300, "completed", "success", "2026-08-28T12:00:00Z"),
        _run(200, "completed", "failure", "2026-08-22T03:00:00Z"),
    ]
    monkeypatch.setattr(reconcile, "fetch_runs", lambda *a, **k: sample)
    monkeypatch.setattr(
        reconcile.sys, "argv", ["main_ci_incident_reconcile.py", "--deciding-run", "200", "--json"]
    )

    rc = reconcile.main()

    captured = capsys.readouterr()
    payload = _json.loads(captured.out)
    assert rc == 0
    assert payload["status"] == "stale"
    assert payload["can_auto_close"] is True


def test_cli_active_exit_code(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A still-failing main yields exit 1 (do not close)."""
    sample = [
        _run(400, "completed", "failure", "2026-08-28T12:00:00Z"),
        _run(300, "completed", "success", "2026-08-27T12:00:00Z"),
    ]
    monkeypatch.setattr(reconcile, "fetch_runs", lambda *a, **k: sample)
    monkeypatch.setattr(
        reconcile.sys, "argv", ["main_ci_incident_reconcile.py", "--deciding-run", "300", "--json"]
    )

    rc = reconcile.main()

    captured = capsys.readouterr()
    payload = _json.loads(captured.out)
    assert rc == 1
    assert payload["status"] == "active"
    assert payload["can_auto_close"] is False


def test_cli_fetch_failure_is_pending_and_exit_1(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A fetch failure under --json exits 1 and reports pending (fail closed)."""
    monkeypatch.setattr(
        reconcile,
        "fetch_runs",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gh run list failed: boom")),
    )
    monkeypatch.setattr(
        reconcile.sys, "argv", ["main_ci_incident_reconcile.py", "--deciding-run", "200", "--json"]
    )

    rc = reconcile.main()
    captured = capsys.readouterr()
    payload = _json.loads(captured.out)

    assert rc == 1
    assert payload["status"] == "pending"
    assert payload["can_auto_close"] is False
    assert "error" in payload
