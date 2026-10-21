"""Contract tests for hosted reproducibility check-run reconciliation."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
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


class PropagatingGitHub(FakeGitHub):
    """Return successive job snapshots while recording bounded polling."""

    def __init__(self, snapshots: list[dict[str, Any]], checks: list[dict[str, Any]]) -> None:
        """Keep the last job snapshot stable once propagation stops."""

        super().__init__([], checks)
        self.snapshots = snapshots
        self.job_reads = 0

    def list_jobs(self, _repository: str, _run_id: int) -> list[dict[str, Any]]:
        self.job_reads += 1
        if len(self.snapshots) > 1:
            return [self.snapshots.pop(0)]
        return self.snapshots


@pytest.fixture
def pending_api_job() -> dict[str, Any]:
    """Model the stale queued/null Actions job observed in the incident."""

    return _job(status="queued", conclusion=None)


def test_queued_job_propagates_to_success_without_premature_patch(
    monkeypatch: Any, pending_api_job: dict[str, Any]
) -> None:

    sleeps = []
    client = PropagatingGitHub(
        [pending_api_job, _job(status="in_progress", conclusion=None), _job()],
        [_check_run(status="queued", conclusion=None), _check_run()],
    )

    def sleep(delay: float) -> None:
        assert client.patch_payloads == []
        sleeps.append(delay)

    monkeypatch.setattr(time, "sleep", sleep)
    result = _reconcile(client)

    assert result["status"] == "reconciled"
    assert client.job_reads == 3
    assert len(sleeps) == 2
    assert all(delay > 0 for delay in sleeps)
    assert len(client.patch_payloads) == 1


def test_persistently_pending_job_reports_pending_without_mutation(
    monkeypatch: Any, pending_api_job: dict[str, Any]
) -> None:

    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    client = PropagatingGitHub([pending_api_job], [])

    result = _reconcile(client)

    assert result["status"] == "pending"
    assert result["action"] == "none"
    assert result["job"] == {"status": "queued", "conclusion": None}
    assert result["fail_closed"] is True
    assert 1 < client.job_reads <= 10
    assert len(sleeps) == client.job_reads - 1
    assert sum(sleeps) <= 60
    assert client.patch_payloads == []


def test_pending_job_followed_by_terminal_failure_still_fails_closed(monkeypatch: Any) -> None:
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    client = PropagatingGitHub(
        [_job(status="queued", conclusion=None), _job(conclusion="failure")], []
    )
    try:
        _reconcile(client)
    except MODULE.ReconciliationError as exc:
        assert "not completed successfully" in str(exc)
    else:
        raise AssertionError("terminal failure after pending must not be repaired")
    assert client.job_reads == 2
    assert client.patch_payloads == []


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"id": JOB_ID + 1}, "identity changed"),
        (
            {"check_run_url": f"https://api.github.com/repos/{REPOSITORY}/check-runs/321"},
            "identity changed",
        ),
        ({"head_sha": "b" * 40}, "found 0"),
        ({"run_attempt": RUN_ATTEMPT + 1}, "found 0"),
        ({"run_id": RUN_ID + 1}, "found 0"),
        ({"name": "other-job"}, "found 0"),
    ],
)
def test_job_identity_change_during_polling_fails_closed(
    monkeypatch: Any, overrides: dict[str, Any], error: str
) -> None:
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    client = PropagatingGitHub([_job(status="queued", conclusion=None), _job(**overrides)], [])
    try:
        _reconcile(client)
    except MODULE.ReconciliationError as exc:
        assert error in str(exc)
    else:
        raise AssertionError("a replaced job must not be repaired")
    assert client.patch_payloads == []


@pytest.mark.parametrize(
    ("status", "conclusion"),
    [
        ("completed", "failure"),
        ("completed", "cancelled"),
        ("queued", "failure"),
        ("unknown", None),
    ],
)
def test_non_success_job_is_never_retried_or_patched(
    monkeypatch: Any, status: str, conclusion: str | None
) -> None:
    def fail_if_sleeping(_delay: float) -> None:
        raise AssertionError("only pending jobs without a conclusion may be retried")

    monkeypatch.setattr(time, "sleep", fail_if_sleeping)
    client = PropagatingGitHub([_job(status=status, conclusion=conclusion)], [])
    with pytest.raises(MODULE.ReconciliationError, match="not completed successfully"):
        _reconcile(client)
    assert client.job_reads == 1
    assert client.patch_payloads == []


def test_cli_reports_pending_as_successful_diagnostic(
    tmp_path: Path, monkeypatch: Any, pending_api_job: dict[str, Any]
) -> None:
    monkeypatch.setattr(time, "sleep", lambda _delay: None)

    def pending_jobs_only(args: list[str], _input: str | None) -> dict[str, Any]:
        assert args == ["api", f"repos/{REPOSITORY}/actions/runs/{RUN_ID}/jobs?per_page=100"]
        return {"jobs": [pending_api_job]}

    client = MODULE.GitHubActionsClient(runner=pending_jobs_only)
    monkeypatch.setattr(MODULE, "GitHubActionsClient", lambda: client)
    output = tmp_path / "reconciliation.json"
    assert (
        MODULE.main(
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
                "success",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pending"
    assert report["action"] == "none"
