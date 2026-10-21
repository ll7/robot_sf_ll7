"""Contract tests for hosted reproducibility check-run reconciliation."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "dev"
    / "reconcile_reproducibility_check_run.py"
)
SPEC = importlib.util.spec_from_file_location("reconcile_reproducibility_check_run", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


REPOSITORY = "ll7/robot_sf_ll7"
RUN_ID = 12345
RUN_ATTEMPT = 2
HEAD_SHA = "a" * 40
JOB_ID = 98765


def _job(**overrides: Any) -> dict[str, Any]:
    job = {
        "id": JOB_ID,
        "name": "reproducibility-check",
        "run_id": RUN_ID,
        "run_attempt": RUN_ATTEMPT,
        "head_sha": HEAD_SHA,
        "check_run_url": (f"https://api.github.com/repos/{REPOSITORY}/check-runs/{JOB_ID}"),
        "status": "completed",
        "conclusion": "success",
    }
    job.update(overrides)
    return job


def _check_run(**overrides: Any) -> dict[str, Any]:
    check_run = {
        "id": JOB_ID,
        "name": "reproducibility-check",
        "head_sha": HEAD_SHA,
        "status": "completed",
        "conclusion": "success",
        "app": {"slug": "github-actions"},
    }
    check_run.update(overrides)
    return check_run


class FakeGitHub:
    """In-memory GitHub API double for reconciliation contract tests."""

    def __init__(self, jobs: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
        """Seed the fake API with job and sequential check-run responses."""

        self.jobs = jobs
        self.checks = checks
        self.patch_payloads: list[dict[str, Any]] = []

    def list_jobs(self, _repository: str, _run_id: int) -> list[dict[str, Any]]:
        return self.jobs

    def get_check_run(self, _repository: str, _check_run_id: int) -> dict[str, Any]:
        if not self.checks:
            raise AssertionError("unexpected check-run read")
        return self.checks.pop(0)

    def complete_check_run(
        self,
        _repository: str,
        _check_run_id: int,
        *,
        output: dict[str, str],
    ) -> dict[str, Any]:
        self.patch_payloads.append(output)
        return _check_run(status="completed", conclusion="success")


def _reconcile(client: FakeGitHub) -> dict[str, Any]:
    return MODULE.reconcile_successful_job(
        client,
        repository=REPOSITORY,
        run_id=RUN_ID,
        run_attempt=RUN_ATTEMPT,
        head_sha=HEAD_SHA,
    )


def test_completed_success_check_run_is_left_unchanged() -> None:
    client = FakeGitHub([_job()], [_check_run()])

    result = _reconcile(client)

    assert result["status"] == "healthy"
    assert result["action"] == "none"
    assert client.patch_payloads == []


def test_pending_check_run_is_patched_and_read_back() -> None:
    client = FakeGitHub(
        [_job()],
        [_check_run(status="in_progress", conclusion=None), _check_run()],
    )

    result = _reconcile(client)

    assert result["status"] == "reconciled"
    assert result["action"] == "patched_and_verified"
    assert len(client.patch_payloads) == 1
    assert "exact GitHub Actions job" in client.patch_payloads[0]["summary"]


def test_terminal_non_success_check_run_fails_closed() -> None:
    client = FakeGitHub([_job()], [_check_run(conclusion="failure")])

    try:
        _reconcile(client)
    except MODULE.ReconciliationError as exc:
        assert "terminal non-success" in str(exc)
    else:
        raise AssertionError("a terminal failing check-run must not be repaired")
    assert client.patch_payloads == []


def test_mismatched_head_sha_fails_closed() -> None:
    client = FakeGitHub([_job()], [_check_run(head_sha="b" * 40)])

    try:
        _reconcile(client)
    except MODULE.ReconciliationError as exc:
        assert "head SHA" in str(exc)
    else:
        raise AssertionError("a check-run for another SHA must not be repaired")
    assert client.patch_payloads == []


def test_cli_skips_failed_upstream_job_without_api_calls(tmp_path: Path, monkeypatch: Any) -> None:
    output = tmp_path / "reconciliation.json"

    def fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the API must not be queried for a failed upstream job")

    monkeypatch.setattr(MODULE, "_run_gh", fail_if_called)
    result = MODULE.main(
        [
            "--repository",
            REPOSITORY,
            "--run-id",
            str(RUN_ID),
            "--run-attempt",
            str(RUN_ATTEMPT),
            "--head-sha",
            HEAD_SHA,
            "--job-result",
            "failure",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "not_applicable"
    assert report["fail_closed"] is True


def _mock_transport(monkeypatch, responses):
    calls, sleeps = [], []
    remaining = iter(responses)

    def run(args, **kwargs):
        calls.append((args, kwargs))
        response = next(remaining)
        if isinstance(response, subprocess.TimeoutExpired):
            raise response
        return response

    monkeypatch.setattr(MODULE.subprocess, "run", run)
    monkeypatch.setattr(MODULE.time, "sleep", sleeps.append)
    return calls, sleeps


@pytest.mark.parametrize("status", [502, 503, 504])
def test_transient_read_failure_retries_before_decoding(monkeypatch, status):
    """A single GitHub gateway failure cannot red an otherwise healthy read."""
    calls, sleeps = _mock_transport(
        monkeypatch,
        [
            subprocess.CompletedProcess([], 1, "", f"gh: Server Error (HTTP {status})"),
            subprocess.CompletedProcess([], 0, '{"jobs": []}', ""),
        ],
    )
    assert MODULE._run_gh(["api", "repos/example/repo/actions/runs/1/jobs"]) == {"jobs": []}
    assert len(calls) == 2 and sleeps == [1]
    assert all(kwargs["timeout"] == 30 for _, kwargs in calls)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_read_failure_fails_immediately(monkeypatch, status):
    """Authentication, authorization and identity failures do not consume retries."""
    calls, sleeps = _mock_transport(
        monkeypatch,
        [
            subprocess.CompletedProcess([], 1, "", f"gh: API error (HTTP {status})"),
        ],
    )
    with pytest.raises(MODULE.ReconciliationError):
        MODULE._run_gh(["api", "repos/example/repo/actions/runs/1/jobs"])
    assert len(calls) == 1 and sleeps == []


def test_persistent_gateway_failure_exhausts_and_fails_closed(monkeypatch):
    """Read retries have a fixed budget and cannot report success on exhaustion."""
    failure = subprocess.CompletedProcess([], 1, "", "gh: Server Error (HTTP 502)")
    calls, sleeps = _mock_transport(monkeypatch, [failure] * 3)
    with pytest.raises(MODULE.ReconciliationError):
        MODULE._run_gh(["api", "repos/example/repo/actions/runs/1/jobs"])
    assert len(calls) == 3 and sleeps == [1, 2]


def test_patch_gateway_failure_is_not_blindly_repeated(monkeypatch):
    """An ambiguous mutation response cannot trigger another success-stamping PATCH."""
    calls, sleeps = _mock_transport(
        monkeypatch,
        [
            subprocess.CompletedProcess([], 1, "", "gh: Server Error (HTTP 502)"),
        ],
    )
    with pytest.raises(MODULE.ReconciliationError):
        MODULE._run_gh(["api", "--method", "PATCH", "repos/example/repo/check-runs/1"], "{}")
    assert len(calls) == 1 and sleeps == []


def test_successful_invalid_json_fails_closed_without_retry(monkeypatch):
    """A successful but malformed API body is not accepted as hosted-state evidence."""
    calls, sleeps = _mock_transport(
        monkeypatch,
        [
            subprocess.CompletedProcess([], 0, "not json", ""),
        ],
    )
    with pytest.raises(MODULE.ReconciliationError, match="invalid JSON"):
        MODULE._run_gh(["api", "repos/example/repo/check-runs/1"])
    assert len(calls) == 1 and sleeps == []


def test_read_timeout_is_bounded_and_retried(monkeypatch):
    """A timed-out read can recover within the same finite transport budget."""
    calls, sleeps = _mock_transport(
        monkeypatch,
        [
            subprocess.TimeoutExpired(["gh"], 30),
            subprocess.CompletedProcess([], 0, '{"id": 1}', ""),
        ],
    )
    assert MODULE._run_gh(["api", "repos/example/repo/check-runs/1"]) == {"id": 1}
    assert len(calls) == 2 and sleeps == [1]


def test_patch_timeout_is_not_repeated(monkeypatch):
    """A timed-out update remains ambiguous and fails closed after one request."""
    calls, sleeps = _mock_transport(monkeypatch, [subprocess.TimeoutExpired(["gh"], 30)])
    with pytest.raises(MODULE.ReconciliationError, match="timed out"):
        MODULE._run_gh(["api", "--method", "PATCH", "repos/example/repo/check-runs/1"], "{}")
    assert len(calls) == 1 and sleeps == []


def test_direct_ci_entrypoint_works_without_project_import_path(tmp_path):
    """Plain isolated Python can run CI's standalone reconciliation entrypoint."""
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT_PATH),
            "--job-result",
            "failure",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["status"] == "not_applicable"
