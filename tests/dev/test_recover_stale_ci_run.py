"""Tests for the guarded setup-starvation recovery script (deterministic, no live GitHub)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from scripts.dev import recover_stale_ci_run as recovery

if TYPE_CHECKING:
    from pathlib import Path

FULL_SHA = "a6640d7141e8f7c3b2a5d9049f1c6e3a8b7d5f2e"
DRIFT_SHA = "b2e5f8a1c4d7e0b3a6c9f2e5d8b1a4c7e3f6a9d2"
RUN_ID = 841501
JOB_ID = 841502
PR = 8415


def _starvation_item(
    *,
    run_id: int = RUN_ID,
    exact_head_sha_matches: bool = True,
    phase: str = "setup",
    setup_starvation: bool = True,
    run_status: str = "in_progress",
    job_status: str = "in_progress",
    run_head_sha: str = FULL_SHA,
) -> dict[str, Any]:
    """Build one setup-starvation lifecycle item."""
    return {
        "name": "fast-feedback (2)",
        "phase": phase,
        "status": "in_progress",
        "age_seconds": 940,
        "age_source": "step_started_at",
        "stale": True,
        "setup_starvation": setup_starvation,
        "step_name": "Set up CI Python environment",
        "run_id": run_id,
        "job_id": JOB_ID,
        "run_status": run_status,
        "job_status": job_status,
        "run_head_sha": run_head_sha,
        "exact_head_sha_matches": exact_head_sha_matches,
    }


def _payload(
    *,
    head_sha: str = FULL_SHA,
    item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal ok CI payload with one setup-starvation run."""
    if item is None:
        item = _starvation_item(run_head_sha=head_sha)
    return {
        "status": "ok",
        "pr": PR,
        "state": "OPEN",
        "head_sha": head_sha,
        "checks": {
            "overall": "pending",
            "pending_reason": "setup_starvation",
            "diagnostic": "actions_gate_setup_starvation",
            "setup_starvation": True,
            "setup_starvation_items": [item],
            "actions_lifecycle": {
                "items": [item],
                "by_phase": {"setup": 1},
                "stale_count": 1,
                "setup_starvation_count": 1,
                "warning_threshold_seconds": 900,
            },
        },
    }


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[dict[str, Any]],
    *,
    gh_result: MagicMock | None = None,
) -> tuple[list[dict[str, Any]], MagicMock]:
    """Patch the recovery script's CI read and gh invocation with deterministic fakes."""
    responses = list(payloads)
    fetch_calls: list[dict[str, Any]] = []

    def fake_fetch(pr_number: str, backoff: float = 0.0, **kwargs: Any) -> dict[str, Any]:
        fetch_calls.append({"pr_number": pr_number, "backoff": backoff, **kwargs})
        index = min(len(fetch_calls) - 1, len(responses) - 1)
        return json.loads(json.dumps(responses[index]))

    gh = MagicMock(return_value=gh_result or MagicMock(returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(recovery, "_fetch_ci_status", fake_fetch)
    monkeypatch.setattr(recovery, "_gh", gh)
    return fetch_calls, gh


def _base_args(*extra: str) -> list[str]:
    """Return the required CLI arguments plus any extra flags."""
    return [
        "--pr",
        str(PR),
        "--run-id",
        str(RUN_ID),
        "--expected-head-sha",
        FULL_SHA,
        "--json",
        *extra,
    ]


def test_plan_only_emits_no_mutation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The default report-only path plans without touching GitHub."""
    fetch_calls, gh = _patch_runtime(monkeypatch, [_payload()])

    rc = recovery.main(_base_args())

    assert rc == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["schema"] == "ci_setup_recovery_receipt.v1"
    assert receipt["decision"] == "plan"
    assert receipt["reason_codes"] == []
    assert receipt["mutation"] == {
        "status": "none",
        "command": None,
        "exit_code": None,
        "stderr_excerpt": "",
    }
    assert receipt["observed_head_sha"] == FULL_SHA
    assert receipt["route_evidence_only"] is True
    assert receipt["implementation_evidence"] is False
    assert receipt["setup_starvation_evidence"]["matched"] is True
    assert receipt["setup_starvation_evidence"]["phase"] == "setup"
    assert len(fetch_calls) == 1
    gh.assert_not_called()


def test_apply_matching_starvation_requests_rerun_once(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--apply re-reads under the guard and requests exactly one rerun."""
    fetch_calls, gh = _patch_runtime(monkeypatch, [_payload(), _payload()])

    rc = recovery.main(_base_args("--apply", "--reason", "operator confirmed stalled setup"))

    assert rc == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["decision"] == "apply"
    assert receipt["mutation"]["status"] == "rerun_requested"
    assert receipt["mutation"]["command"] == ["gh", "run", "rerun", str(RUN_ID)]
    assert receipt["mutation"]["exit_code"] == 0
    assert len(fetch_calls) == 2
    gh.assert_called_once()
    assert gh.call_args.args[0] == ["run", "rerun", str(RUN_ID)]


def test_head_drift_refuses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A live PR head that differs from --expected-head-sha refuses without mutation."""
    _, gh = _patch_runtime(monkeypatch, [_payload(head_sha=DRIFT_SHA)])

    rc = recovery.main(_base_args("--apply", "--reason", "operator approved"))

    assert rc == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["decision"] == "refuse"
    assert "pr_head_mismatch" in receipt["reason_codes"]
    assert receipt["observed_head_sha"] == DRIFT_SHA
    assert receipt["mutation"]["status"] == "none"
    gh.assert_not_called()


def test_run_not_in_setup_starvation_items_refuses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A requested run absent from the starvation evidence refuses without mutation."""
    other_run = _starvation_item(run_id=RUN_ID + 1)
    _, gh = _patch_runtime(monkeypatch, [_payload(item=other_run)])

    rc = recovery.main(_base_args("--apply", "--reason", "operator approved"))

    assert rc == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["decision"] == "refuse"
    assert "run_not_in_setup_starvation_items" in receipt["reason_codes"]
    assert receipt["setup_starvation_evidence"]["matched"] is False
    gh.assert_not_called()


def test_apply_gh_failure_records_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A nonzero gh exit records a failed mutation and exits nonzero."""
    gh_result = MagicMock(returncode=2, stdout="", stderr="simulated rerun failure")
    _, gh = _patch_runtime(monkeypatch, [_payload(), _payload()], gh_result=gh_result)

    rc = recovery.main(_base_args("--apply", "--reason", "operator approved"))

    assert rc == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["decision"] == "apply"
    assert receipt["mutation"]["status"] == "failed"
    assert receipt["mutation"]["exit_code"] == 2
    assert receipt["mutation"]["stderr_excerpt"] == "simulated rerun failure"
    gh.assert_called_once()


def test_apply_without_reason_refuses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--apply without an explicit operator reason refuses before any GitHub read."""
    fetch_calls, gh = _patch_runtime(monkeypatch, [_payload()])

    rc = recovery.main(_base_args("--apply"))

    assert rc == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["decision"] == "refuse"
    assert "apply_reason_missing" in receipt["reason_codes"]
    assert fetch_calls == []
    gh.assert_not_called()


def test_recheck_head_drift_refuses_before_mutation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A head change between plan and locked re-read refuses without mutation."""
    fetch_calls, gh = _patch_runtime(monkeypatch, [_payload(), _payload(head_sha=DRIFT_SHA)])

    rc = recovery.main(_base_args("--apply", "--reason", "operator approved"))

    assert rc == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["decision"] == "refuse"
    assert "pr_head_mismatch" in receipt["reason_codes"]
    assert len(fetch_calls) == 2
    gh.assert_not_called()


def test_invalid_expected_head_sha_refuses(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A short or malformed expected head refuses before any GitHub read."""
    fetch_calls, gh = _patch_runtime(monkeypatch, [_payload()])

    rc = recovery.main(
        [
            "--pr",
            str(PR),
            "--run-id",
            str(RUN_ID),
            "--expected-head-sha",
            FULL_SHA[:8],
            "--json",
        ]
    )

    assert rc == 1
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["decision"] == "refuse"
    assert receipt["reason_codes"] == ["invalid_expected_head_sha"]
    assert fetch_calls == []
    gh.assert_not_called()


def test_output_writes_receipt_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--output writes the same receipt that is summarized on stdout."""
    _patch_runtime(monkeypatch, [_payload()])
    output_path = tmp_path / "receipt.json"

    rc = recovery.main([*_base_args()[:-1], "--output", str(output_path)])

    assert rc == 0
    assert "decision: plan" in capsys.readouterr().out
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    assert receipt["decision"] == "plan"
    assert receipt["schema"] == "ci_setup_recovery_receipt.v1"
