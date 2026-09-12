"""Tests for bounded PR-readiness termination receipts."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

from scripts.dev import pr_ready_termination
from scripts.dev.pr_ready_termination import (
    TerminationContext,
    _process_group_exists,
    _process_group_liveness,
    build_receipt,
    write_receipt,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_receipt_is_bounded_and_credential_free() -> None:
    """Receipt context is bounded and never includes command lines or environments."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="core_lane",
            lane="core",
            last_progress="x" * 1000,
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="process_group_cleanup_unverified",
            mode="interim",
            controller_pid=os.getpid(),
            child_pid=os.getpid(),
            child_process_group_id=os.getpgrp(),
        )
    )

    assert receipt["schema"] == "pr_ready_termination.v1"
    assert receipt["signal"] == {"name": "SIGTERM", "number": 15, "exit_code": 143}
    assert len(receipt["last_progress"]["message"]) == 200
    assert receipt["security"] == {
        "command_line_included": False,
        "environment_included": False,
    }
    assert "command" not in receipt
    assert "environment" not in receipt
    assert receipt["resources"]["host"]["cpu_count"] is not None


def test_write_receipt_is_private_and_does_not_overwrite(tmp_path: Path) -> None:
    """Receipts are private files and an existing path is never replaced."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="preflight",
            lane="none",
            last_progress="preflight",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="no_child_active",
            mode="interim",
        )
    )
    output = tmp_path / "nested" / "termination.json"

    assert write_receipt(receipt, output) == output
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_receipt(receipt, output)


def test_unverified_cleanup_with_missing_pgid_is_not_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing PGID cannot upgrade an unverified live child to verified cleanup."""
    monkeypatch.setattr(pr_ready_termination, "_process_group_exists", lambda _: None)
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="core_lane",
            lane="core",
            last_progress="core readiness lane running",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="process_group_cleanup_unverified",
            mode="interim",
            child_pid=1234,
            child_process_group_id=None,
            child_registration_state="registered",
        )
    )

    assert receipt["cleanup"]["verified"] is False


def test_verified_group_cleanup_is_downgraded_when_group_still_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contradictory live PGID cannot retain a verified group-cleanup claim."""
    for process_group_exists in (True, None):
        monkeypatch.setattr(
            pr_ready_termination,
            "_process_group_exists",
            lambda _, exists=process_group_exists: exists,
        )
        receipt = build_receipt(
            TerminationContext(
                signal_number=15,
                phase="core_lane",
                lane="core",
                last_progress="core readiness lane running",
                last_progress_at_utc="2026-09-03T06:00:00Z",
                cleanup_status="process_group_killed_and_verified",
                mode="interim",
                child_pid=1234,
                child_process_group_id=1234,
                child_registration_state="registered",
            )
        )

        assert receipt["process"]["child_process_group_exists"] is process_group_exists
        assert receipt["cleanup"] == {
            "status": "process_group_cleanup_unverified",
            "verified": False,
        }


def test_no_child_cleanup_is_verified_without_process_identifiers() -> None:
    """No active child remains a verified cleanup result without a PGID."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="preflight",
            lane="none",
            last_progress="preflight",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="no_child_active",
            mode="interim",
            child_registration_state="not_started",
        )
    )

    assert receipt["cleanup"]["verified"] is True


@pytest.mark.parametrize(
    ("child_pid", "child_process_group_id"),
    [("not-a-pid", None), (None, "not-a-pgid"), (0, None)],
)
def test_no_child_cleanup_rejects_invalid_process_identifiers(
    child_pid: object, child_process_group_id: object
) -> None:
    """Explicit but invalid identifiers cannot be treated as an absent child."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="preflight",
            lane="none",
            last_progress="preflight",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="no_child_active",
            mode="interim",
            child_pid=child_pid,
            child_process_group_id=child_process_group_id,
            child_registration_state="not_started",
        )
    )

    assert receipt["cleanup"] == {
        "status": "process_group_cleanup_unverified",
        "verified": False,
    }


def test_direct_cleanup_preserves_foreground_fallback_without_pgid() -> None:
    """A registered direct child can verify its own cleanup when no PGID is available."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="core_lane",
            lane="core",
            last_progress="core readiness lane running",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="direct_process_terminated_and_verified",
            mode="interim",
            child_pid=1234,
            child_process_group_id=None,
            child_registration_state="registered",
        )
    )

    assert receipt["cleanup"] == {
        "status": "direct_process_terminated_and_verified",
        "verified": True,
    }


def test_direct_cleanup_rejects_invalid_pgid(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid supplied PGID cannot be confused with the documented foreground fallback."""
    monkeypatch.setattr(pr_ready_termination, "_process_group_exists", lambda _: None)
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="core_lane",
            lane="core",
            last_progress="core readiness lane running",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="direct_process_terminated_and_verified",
            mode="interim",
            child_pid=1234,
            child_process_group_id="not-a-pgid",
            child_registration_state="registered",
        )
    )

    assert receipt["cleanup"] == {
        "status": "process_group_cleanup_unverified",
        "verified": False,
    }


@pytest.mark.parametrize("registration_state", ["registering", "registered", "unknown"])
def test_no_child_cleanup_requires_explicit_not_started_state(
    registration_state: str,
) -> None:
    """An incomplete or unknown lifecycle state cannot masquerade as no active child."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="core_lane",
            lane="core",
            last_progress="starting core readiness lane",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="no_child_active",
            mode="interim",
            child_registration_state=registration_state,
        )
    )

    assert receipt["cleanup"] == {
        "status": "process_group_cleanup_unverified",
        "verified": False,
    }
    assert receipt["process"]["child_registration_state"] == registration_state


@pytest.mark.parametrize(
    "cleanup_status",
    [
        "direct_process_terminated_and_verified",
        "process_group_terminated_and_verified",
    ],
)
def test_verified_cleanup_requires_registered_child_identifiers(cleanup_status: str) -> None:
    """Verified cleanup is rejected when registration has not completed or IDs are absent."""
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="core_lane",
            lane="core",
            last_progress="starting core readiness lane",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status=cleanup_status,
            mode="interim",
            child_registration_state="registering",
        )
    )

    assert receipt["cleanup"]["verified"] is False
    assert receipt["cleanup"]["status"] == "process_group_cleanup_unverified"


def test_process_group_liveness_returns_none_for_none() -> None:
    """A missing PGID produces None liveness rather than defaulting to absent."""
    assert _process_group_liveness(None) is None


def test_process_group_liveness_absent_for_nonexistent_pgid() -> None:
    """A nonexistent process group is truthfully classified as absent."""
    assert _process_group_liveness(9999999) == "absent"


def test_process_group_liveness_live_for_active_process_group() -> None:
    """An active process group with live processes is classified as live."""
    assert _process_group_liveness(os.getpgrp()) == "live"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_process_group_liveness_zombie_only_for_unreaped_group() -> None:
    """An exited unreaped process group leader is recognized as zombie_only."""
    pipe_r, pipe_w = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:
        os.close(pipe_r)
        os.setsid()
        os.write(pipe_w, b"ready\n")
        os.close(pipe_w)
        os._exit(0)

    os.close(pipe_w)
    try:
        os.read(pipe_r, 6)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            liveness = _process_group_liveness(child_pid)
            if liveness == "zombie_only":
                break
            time.sleep(0.01)
        assert _process_group_exists(child_pid) is True
        assert _process_group_liveness(child_pid) == "zombie_only"
    finally:
        os.close(pipe_r)
        os.waitpid(child_pid, 0)

    assert _process_group_exists(child_pid) is False
    assert _process_group_liveness(child_pid) == "absent"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups required")
def test_process_group_liveness_live_for_mixed_group() -> None:
    """A process group with both live and zombie members is classified as live."""
    ready_r, ready_w = os.pipe()
    release_r, release_w = os.pipe()

    leader_pid = os.fork()
    if leader_pid == 0:
        os.close(ready_r)
        os.close(release_w)
        os.setsid()
        zombie_child = os.fork()
        if zombie_child == 0:
            os.close(release_r)
            os.close(ready_w)
            os._exit(0)
        os.write(ready_w, b"ready\n")
        os.close(ready_w)
        os.read(release_r, 1)
        os.close(release_r)
        os.waitpid(zombie_child, 0)
        os._exit(0)

    os.close(ready_w)
    os.close(release_r)
    try:
        os.read(ready_r, 6)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if _process_group_liveness(leader_pid) == "live":
                break
            time.sleep(0.01)
        assert _process_group_exists(leader_pid) is True
        assert _process_group_liveness(leader_pid) == "live"
    finally:
        os.close(ready_r)
        try:
            os.write(release_w, b"x")
        except OSError:
            pass
        os.close(release_w)
        os.waitpid(leader_pid, 0)


def test_verified_group_cleanup_accepts_zombie_only_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group with child_process_group_exists=True but zombie_only liveness retains verified cleanup."""
    monkeypatch.setattr(pr_ready_termination, "_process_group_exists", lambda _: True)
    monkeypatch.setattr(pr_ready_termination, "_process_group_liveness", lambda _: "zombie_only")
    receipt = build_receipt(
        TerminationContext(
            signal_number=15,
            phase="core_lane",
            lane="core",
            last_progress="core readiness lane running",
            last_progress_at_utc="2026-09-03T06:00:00Z",
            cleanup_status="process_group_terminated_and_verified",
            mode="interim",
            child_pid=1234,
            child_process_group_id=1234,
            child_registration_state="registered",
        )
    )

    assert receipt["process"]["child_process_group_exists"] is True
    assert receipt["process"]["child_process_group_liveness"] == "zombie_only"
    assert receipt["cleanup"] == {
        "status": "process_group_terminated_and_verified",
        "verified": True,
    }


def test_verified_group_cleanup_rejects_live_or_unknown_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A group with live or unknown liveness is downgraded to unverified cleanup."""
    for liveness in ("live", "unknown"):
        monkeypatch.setattr(pr_ready_termination, "_process_group_exists", lambda _: True)
        monkeypatch.setattr(
            pr_ready_termination, "_process_group_liveness", lambda _, live=liveness: live
        )
        receipt = build_receipt(
            TerminationContext(
                signal_number=15,
                phase="core_lane",
                lane="core",
                last_progress="core readiness lane running",
                last_progress_at_utc="2026-09-03T06:00:00Z",
                cleanup_status="process_group_terminated_and_verified",
                mode="interim",
                child_pid=1234,
                child_process_group_id=1234,
                child_registration_state="registered",
            )
        )

        assert receipt["process"]["child_process_group_exists"] is True
        assert receipt["process"]["child_process_group_liveness"] == liveness
        assert receipt["cleanup"] == {
            "status": "process_group_cleanup_unverified",
            "verified": False,
        }
