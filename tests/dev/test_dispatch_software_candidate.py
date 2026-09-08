"""Offline contract tests for exact-source software-candidate dispatch."""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev import dispatch_software_candidate as dispatch

SOURCE_SHA = "a" * 40


def _result(stdout: str = "", *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["gh", "api"], returncode, stdout, "")


def test_dispatch_binds_source_and_deletes_ref_after_terminal_run(monkeypatch) -> None:
    calls: list[tuple[str, str, object]] = []
    run = {
        "id": 123,
        "head_branch": "automation/software-candidate/aaaaaaaaaaaa-deadbeef00",
        "head_sha": SOURCE_SHA,
        "event": "workflow_dispatch",
        "status": "completed",
        "conclusion": "failure",
        "html_url": "https://github.com/ll7/robot_sf_ll7/actions/runs/123",
    }

    monkeypatch.setattr(dispatch.secrets, "token_hex", lambda _size: "deadbeef00")

    def fake_run_gh_api(path, payload=None, *, method=None, **_kwargs):
        calls.append((method or "GET", path, payload))
        if method is None:
            if "/commits/" in path:
                return _result(json.dumps({"sha": SOURCE_SHA}))
            if "/actions/workflows/" in path:
                return _result(json.dumps({"workflow_runs": [run]}))
            if "/actions/runs/" in path:
                return _result(json.dumps(run))
        if path.endswith("/git/refs"):
            return _result(json.dumps({"object": {"sha": SOURCE_SHA}}))
        return _result()

    monkeypatch.setattr(dispatch, "run_gh_api", fake_run_gh_api)
    receipt = dispatch.dispatch(
        repo="ll7/robot_sf_ll7",
        workflow="software-candidate.yml",
        source_sha=SOURCE_SHA,
        discovery_timeout=1,
        wait=True,
        run_timeout=1,
        poll_seconds=1,
    )

    assert receipt["requested_source_sha"] == SOURCE_SHA
    assert receipt["observed_source_sha"] == SOURCE_SHA
    assert receipt["temporary_ref_deleted"] is True
    assert calls[0] == ("GET", f"repos/ll7/robot_sf_ll7/commits/{SOURCE_SHA}", None)
    assert calls[1][0:2] == ("POST", "repos/ll7/robot_sf_ll7/git/refs")
    assert calls[1][2] == {
        "ref": "refs/heads/automation/software-candidate/aaaaaaaaaaaa-deadbeef00",
        "sha": SOURCE_SHA,
    }
    assert calls[2][0:2] == (
        "POST",
        "repos/ll7/robot_sf_ll7/actions/workflows/software-candidate.yml/dispatches",
    )
    assert calls[2][2] == {
        "ref": "automation/software-candidate/aaaaaaaaaaaa-deadbeef00",
        "inputs": {"requested_source_sha": SOURCE_SHA},
    }
    assert calls[-1] == (
        "DELETE",
        "repos/ll7/robot_sf_ll7/git/refs/heads/automation/software-candidate/aaaaaaaaaaaa-deadbeef00",
        None,
    )


def test_no_wait_keeps_ref_for_runner_checkout(monkeypatch) -> None:
    monkeypatch.setattr(dispatch.secrets, "token_hex", lambda _size: "deadbeef00")

    def fake_run_gh_api(path, payload=None, *, method=None, **_kwargs):
        if method is None and "/commits/" in path:
            return _result(json.dumps({"sha": SOURCE_SHA}))
        if method is None and "/actions/workflows/" in path:
            return _result(
                json.dumps(
                    {
                        "workflow_runs": [
                            {
                                "id": 123,
                                "head_branch": "automation/software-candidate/aaaaaaaaaaaa-deadbeef00",
                                "head_sha": SOURCE_SHA,
                                "event": "workflow_dispatch",
                                "status": "queued",
                            }
                        ]
                    }
                )
            )
        if path.endswith("/git/refs"):
            return _result(json.dumps({"object": {"sha": SOURCE_SHA}}))
        return _result()

    calls: list[tuple[str, str]] = []

    def recording_run_gh_api(path, payload=None, *, method=None, **kwargs):
        calls.append((method or "GET", path))
        return fake_run_gh_api(path, payload, method=method, **kwargs)

    monkeypatch.setattr(dispatch, "run_gh_api", recording_run_gh_api)
    receipt = dispatch.dispatch(
        repo="ll7/robot_sf_ll7",
        workflow="software-candidate.yml",
        source_sha=SOURCE_SHA,
        discovery_timeout=1,
        wait=False,
        run_timeout=1,
        poll_seconds=1,
    )

    assert receipt["temporary_ref_deleted"] is False
    assert not any(method == "DELETE" for method, _path in calls)


@pytest.mark.parametrize("source_sha", ["A" * 40, "a" * 39, "not-a-sha"])
def test_dispatch_rejects_noncanonical_source_before_api_write(
    monkeypatch, source_sha: str
) -> None:
    monkeypatch.setattr(
        dispatch,
        "run_gh_api",
        lambda *_args, **_kwargs: pytest.fail("invalid input must not call GitHub"),
    )
    with pytest.raises(dispatch.DispatchError, match="exact lowercase 40-hex"):
        dispatch.dispatch(
            repo="ll7/robot_sf_ll7",
            workflow="software-candidate.yml",
            source_sha=source_sha,
            discovery_timeout=1,
            wait=False,
            run_timeout=1,
            poll_seconds=1,
        )
