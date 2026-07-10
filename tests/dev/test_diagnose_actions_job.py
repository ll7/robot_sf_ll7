"""Tests for the missing-GitHub-Actions-log annotation fallback."""

from __future__ import annotations

import json
import subprocess

from scripts.dev import diagnose_actions_job


def _result(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """Build a compact mocked ``gh`` result."""
    return subprocess.CompletedProcess(
        args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_annotations_path_requires_a_github_check_run_url() -> None:
    """Only GitHub API check-run URLs may become annotation endpoints."""
    assert (
        diagnose_actions_job._annotations_path(
            "https://api.github.com/repos/ll7/robot_sf_ll7/check-runs/123",
        )
        == "repos/ll7/robot_sf_ll7/check-runs/123/annotations?per_page=100"
    )
    assert diagnose_actions_job._annotations_path("https://example.test/check-runs/123") is None
    assert diagnose_actions_job._annotations_path(None) is None


def test_main_prints_normal_logs_without_requesting_annotations(monkeypatch, capsys) -> None:
    """A usable normal log remains the preferred diagnostic output."""
    calls: list[list[str]] = []
    results = iter(
        [
            _result(0, json.dumps({"run_id": 456, "check_run_url": "unused"})),
            _result(0, "unit test output\n"),
        ]
    )

    def fake_gh(args: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return next(results)

    monkeypatch.setattr(diagnose_actions_job, "_gh", fake_gh)

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 0
    assert capsys.readouterr().out == "unit test output\n"
    assert len(calls) == 2
    assert calls[1][:3] == ["run", "view", "456"]


def test_main_falls_back_to_check_run_annotations_when_logs_are_absent(monkeypatch, capsys) -> None:
    """An unavailable job log should expose GitHub's retained error annotation."""
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(0, json.dumps([[{"message": "No space left on device"}]]) + "\n"),
        ]
    )

    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 0
    captured = capsys.readouterr()
    assert "No space left on device" in captured.out
    assert "Normal log retrieval unavailable" in captured.err
    assert "Falling back to check-run annotations." in captured.err


def test_main_fails_closed_when_annotation_fallback_is_unavailable(monkeypatch, capsys) -> None:
    """Missing logs are not treated as diagnosed when annotations also fail."""
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(1, stderr="HTTP 403: Forbidden"),
        ]
    )

    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 1
    assert "Could not recover check-run annotations: HTTP 403: Forbidden" in capsys.readouterr().err


def test_main_fails_closed_when_annotations_are_empty(monkeypatch, capsys) -> None:
    """An empty annotations list is not a successful diagnosis."""
    results = iter(
        [
            _result(
                0,
                json.dumps(
                    {
                        "run_id": 456,
                        "check_run_url": "https://api.github.com/repos/owner/repo/check-runs/789",
                    }
                ),
            ),
            _result(1, stderr="HTTP 404: Not Found"),
            _result(0, json.dumps([[]]) + "\n"),
        ]
    )

    monkeypatch.setattr(diagnose_actions_job, "_gh", lambda _args: next(results))

    assert diagnose_actions_job.main(["123", "--repo", "owner/repo"]) == 1
    assert "the endpoint returned no annotations" in capsys.readouterr().err
