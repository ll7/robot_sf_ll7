"""Tests for the main-CI green/red signal used by the red-main merge hold (#5385).

The load-bearing property: an IN-PROGRESS run must never decide green or red —
only the most recent *completed* run does. This is the exact bug that made the
escalation guard silent on 2026-07-11 (it counted an in-progress newest run),
so it gets an explicit test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.dev import main_ci_is_green
from scripts.dev.main_ci_is_green import (
    MainCiRunFetchError,
    MainCiRunWindow,
    build_signal,
    classify,
    decide,
    decide_verified_main_ci_signal,
    dispatch_decision,
    dispatch_gate_decision,
    fetch_dispatch_retry_matrix_admitted,
    fetch_dispatch_run_window,
    fetch_run_window,
    fetch_runs,
    latest_completed_run,
    wait_for_dispatch_gate,
)


def _run(
    rid: int,
    status: str,
    conclusion: str | None,
    created: str,
    *,
    event: str = "push",
    head_sha: str | None = None,
) -> dict:
    return {
        "databaseId": rid,
        "status": status,
        "conclusion": conclusion,
        "headSha": head_sha or f"{rid:040x}",
        "createdAt": created,
        "event": event,
    }


def test_green_when_latest_completed_succeeded() -> None:
    """Green when the newest completed run succeeded."""
    runs = [
        _run(3, "completed", "success", "2026-07-12T12:00:00Z"),
        _run(2, "completed", "failure", "2026-07-12T11:00:00Z"),
    ]
    is_green, run = decide(runs)
    assert is_green is True
    assert run["databaseId"] == 3


def test_red_when_latest_completed_failed() -> None:
    """Red when the newest completed run failed."""
    runs = [
        _run(3, "completed", "failure", "2026-07-12T12:00:00Z"),
        _run(2, "completed", "success", "2026-07-12T11:00:00Z"),
    ]
    is_green, run = decide(runs)
    assert is_green is False
    assert run["databaseId"] == 3


def test_in_progress_newest_is_ignored_completed_green_wins() -> None:
    """An in-progress newest run is ignored; latest completed green wins."""
    # The newest run is still RUNNING; the decision must come from the latest
    # *completed* run (green), not the in-progress one. Regression guard for the
    # 2026-07-11 escalation-guard silent miss.
    runs = [
        _run(4, "in_progress", None, "2026-07-12T12:30:00Z"),
        _run(3, "queued", None, "2026-07-12T12:20:00Z"),
        _run(2, "completed", "success", "2026-07-12T12:00:00Z"),
    ]
    is_green, run = decide(runs)
    assert is_green is True
    assert run["databaseId"] == 2


def test_in_progress_newest_does_not_mask_a_red_completed() -> None:
    """An in-progress newest run does not hide a red completed run."""
    runs = [
        _run(4, "in_progress", None, "2026-07-12T12:30:00Z"),
        _run(2, "completed", "failure", "2026-07-12T12:00:00Z"),
    ]
    is_green, run = decide(runs)
    assert is_green is False
    assert run["databaseId"] == 2


def test_verified_signal_refuses_a_completed_run_for_an_older_main_head() -> None:
    """A stale listing cannot reuse an old green run for today's main SHA."""
    old_success = _run(
        7,
        "completed",
        "success",
        "2026-09-24T12:00:00Z",
        head_sha="a" * 40,
    )

    is_green, run = decide_verified_main_ci_signal(
        [old_success],
        expected_head_sha="b" * 40,
        matrix_admission_lookup=lambda _run_id: True,
    )

    assert is_green is False
    assert run is None


def test_fetch_main_head_sha_reads_and_validates_the_main_ref() -> None:
    """The branch-tip reader binds signals to the exact live main commit."""
    expected_sha = "a" * 40

    def fake_runner(path: str, *_args: object, **_kwargs: object):
        assert path == "repos/ll7/robot_sf_ll7/git/ref/heads/main"
        return subprocess.CompletedProcess(
            ["gh"],
            0,
            json.dumps({"ref": "refs/heads/main", "object": {"sha": expected_sha}}),
            "",
        )

    assert main_ci_is_green.fetch_main_head_sha(runner=fake_runner) == expected_sha


@pytest.mark.parametrize(
    "payload",
    [
        {"ref": "refs/heads/other", "object": {"sha": "a" * 40}},
        {"ref": "refs/heads/main", "object": {"sha": "bad-sha"}},
        {"ref": "refs/heads/main", "object": None},
    ],
)
def test_fetch_main_head_sha_fails_closed_on_malformed_ref(payload: dict) -> None:
    """A malformed or mismatched branch ref cannot authorize green."""

    def fake_runner(_path: str, *_args: object, **_kwargs: object):
        return subprocess.CompletedProcess(["gh"], 0, json.dumps(payload), "")

    with pytest.raises(MainCiRunFetchError, match="main branch ref|commit SHA"):
        main_ci_is_green.fetch_main_head_sha(runner=fake_runner)


def test_stale_only_window_is_not_green_and_has_no_deciding_run() -> None:
    """A window of only stale completed runs yields no decisive verdict (fail closed)."""
    for stale in ("cancelled", "timed_out", "startup_failure", "skipped", "neutral", None):
        is_green, run = decide([_run(1, "completed", stale, "2026-07-12T12:00:00Z")])
        assert is_green is False, stale
        assert run is None, stale  # stale is skipped -> no deciding run


def test_gate_only_dispatch_failure_uses_same_head_matrix_verdict() -> None:
    """A gate-only failure is not red when an older same-SHA run has a verdict."""
    sha = "a" * 40
    runs = [
        _run(
            11,
            "completed",
            "failure",
            "2026-09-24T13:35:04Z",
            event="workflow_dispatch",
            head_sha=sha,
        ),
        _run(10, "completed", "success", "2026-09-24T13:00:00Z", head_sha=sha),
    ]

    is_green, deciding_run = decide_verified_main_ci_signal(
        runs,
        matrix_admission_lookup=lambda run_id: run_id != 11,
    )

    assert is_green is True
    assert deciding_run is not None
    assert deciding_run["databaseId"] == 10


def test_gate_only_failure_does_not_reuse_a_different_head_verdict() -> None:
    """A failed gate-only run on a new SHA cannot inherit an older SHA's green."""
    runs = [
        _run(
            11,
            "completed",
            "failure",
            "2026-09-24T13:35:04Z",
            event="workflow_dispatch",
            head_sha="b" * 40,
        ),
        _run(10, "completed", "success", "2026-09-24T13:00:00Z", head_sha="a" * 40),
    ]

    is_green, deciding_run = decide_verified_main_ci_signal(
        runs,
        matrix_admission_lookup=lambda _run_id: False,
    )

    assert is_green is False
    assert deciding_run is None


def test_all_skipped_dispatch_success_is_not_green_for_a_new_head() -> None:
    """A gate-only success with explicit all-skipped matrix cannot claim green."""
    runs = [
        _run(
            11,
            "completed",
            "success",
            "2026-09-24T13:35:04Z",
            event="workflow_dispatch",
            head_sha="b" * 40,
        ),
        _run(10, "completed", "success", "2026-09-24T13:00:00Z", head_sha="a" * 40),
    ]

    is_green, deciding_run = decide_verified_main_ci_signal(
        runs,
        matrix_admission_lookup=lambda _run_id: False,
    )

    assert is_green is False
    assert deciding_run is None


def test_admitted_workflow_dispatch_matrix_failure_remains_red() -> None:
    """A manual run is decisive red when its compatibility matrix was admitted."""
    manual_failure = _run(
        11,
        "completed",
        "failure",
        "2026-09-24T13:35:04Z",
        event="workflow_dispatch",
        head_sha="b" * 40,
    )

    is_green, deciding_run = decide_verified_main_ci_signal(
        [manual_failure],
        matrix_admission_lookup=lambda _run_id: True,
    )

    assert is_green is False
    assert deciding_run is manual_failure
    assert deciding_run["fullMatrixAdmitted"] is True


def test_manual_run_admission_lookup_error_propagates_fail_closed() -> None:
    """Unreadable admission evidence must become stale, not a guessed verdict."""
    manual_failure = _run(
        11,
        "completed",
        "failure",
        "2026-09-24T13:35:04Z",
        event="workflow_dispatch",
        head_sha="b" * 40,
    )

    with pytest.raises(MainCiRunFetchError, match="job API unavailable"):
        decide_verified_main_ci_signal(
            [manual_failure],
            matrix_admission_lookup=lambda _run_id: (_ for _ in ()).throw(
                MainCiRunFetchError("job API unavailable")
            ),
        )


def test_json_signal_does_not_count_all_skipped_dispatch_as_green(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """An all-skipped manual success is stale in the actual machine signal."""
    manual_success = _run(
        11,
        "completed",
        "success",
        "2026-09-24T13:35:04Z",
        event="workflow_dispatch",
        head_sha="b" * 40,
    )
    monkeypatch.setattr(main_ci_is_green, "fetch_runs", lambda *a, **k: [manual_success])
    monkeypatch.setattr(main_ci_is_green, "fetch_main_head_sha", lambda *_a, **_k: "b" * 40)

    def fake_rest_runner(path: str, *_args: object, **_kwargs: object):
        assert path == "repos/ll7/robot_sf_ll7/actions/runs/11/jobs?per_page=100&page=1"
        jobs = {
            "jobs": [
                {
                    "name": "compat-matrix (ubuntu-latest, 3.11)",
                    "status": "completed",
                    "conclusion": "skipped",
                }
            ]
        }
        return subprocess.CompletedProcess(["gh"], 0, json.dumps(jobs), "")

    monkeypatch.setattr(main_ci_is_green, "_default_rest_runner", fake_rest_runner)
    monkeypatch.setattr(main_ci_is_green.sys, "argv", ["main_ci_is_green.py", "--json"])

    rc = main_ci_is_green.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["status"] == "stale"
    assert payload["is_green"] is False
    assert payload["deciding_run"] is None


@pytest.mark.parametrize(
    "job_payload",
    [
        None,
        {"jobs": []},
        {
            "jobs": [
                {
                    "name": "compat-matrix (ubuntu-latest, 3.11)",
                    "status": "completed",
                    "conclusion": None,
                }
            ]
        },
    ],
    ids=["transport-error", "missing-matrix", "malformed-matrix"],
)
def test_json_signal_matrix_admission_failure_is_stale(
    monkeypatch: pytest.MonkeyPatch, capsys, job_payload: dict | None
) -> None:
    """Transport or malformed matrix evidence cannot turn a manual run green."""
    manual_success = _run(
        11,
        "completed",
        "success",
        "2026-09-24T13:35:04Z",
        event="workflow_dispatch",
        head_sha="b" * 40,
    )
    monkeypatch.setattr(main_ci_is_green, "fetch_runs", lambda *a, **k: [manual_success])
    monkeypatch.setattr(main_ci_is_green, "fetch_main_head_sha", lambda *_a, **_k: "b" * 40)

    def fake_rest_runner(path: str, *_args: object, **_kwargs: object):
        assert path == "repos/ll7/robot_sf_ll7/actions/runs/11/jobs?per_page=100&page=1"
        if job_payload is None:
            return subprocess.CompletedProcess(["gh"], 1, "", "jobs API unavailable")
        return subprocess.CompletedProcess(["gh"], 0, json.dumps(job_payload), "")

    monkeypatch.setattr(main_ci_is_green, "_default_rest_runner", fake_rest_runner)
    monkeypatch.setattr(main_ci_is_green.sys, "argv", ["main_ci_is_green.py", "--json"])

    rc = main_ci_is_green.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["status"] == "stale"
    assert payload["is_green"] is False
    assert payload["deciding_run"] is None
    assert "error" in payload


def test_cancelled_newest_is_skipped_and_older_green_decides() -> None:
    """A cancelled (superseded) newest run must not block; the older green decides.

    This is the exact freeze that stranded ~8 gate-vetted PRs on 2026-07-13:
    rapid merges superseded each other into cancelled runs, and cancelled was
    mis-read as red. Cancelled carries no verdict -> skip to the latest decisive.
    """
    runs = [
        _run(3, "completed", "cancelled", "2026-07-13T03:00:00Z"),
        _run(2, "completed", "success", "2026-07-13T02:00:00Z"),
    ]
    is_green, run = decide(runs)
    assert is_green is True
    assert run["databaseId"] == 2


def test_cancelled_newest_does_not_hide_an_older_red() -> None:
    """Skipping cancelled must not skip past a real failure to an even older green."""
    runs = [
        _run(3, "completed", "cancelled", "2026-07-13T03:00:00Z"),
        _run(2, "completed", "failure", "2026-07-13T02:00:00Z"),
        _run(1, "completed", "success", "2026-07-13T01:00:00Z"),
    ]
    is_green, run = decide(runs)
    assert is_green is False
    assert run["databaseId"] == 2  # the failure is the latest decisive verdict


def test_classify_buckets() -> None:
    """success->green, failure->red, everything else->stale."""
    assert classify("success") == "green"
    assert classify("failure") == "red"
    for stale in ("cancelled", "timed_out", "skipped", "neutral", "startup_failure", None):
        assert classify(stale) == "stale", stale


def test_no_completed_runs_is_not_green() -> None:
    """No completed run means not green (and None deciding run)."""
    runs = [_run(4, "in_progress", None, "2026-07-12T12:30:00Z")]
    is_green, run = decide(runs)
    assert is_green is False
    assert run is None

    assert latest_completed_run([]) is None


def test_dispatch_policy_observes_queued_and_in_progress_same_head() -> None:
    """The watcher must not replace a queued or active decisive run."""
    for status in ("queued", "in_progress"):
        decision = dispatch_decision(
            "a" * 40,
            [{"headSha": "a" * 40, "status": status, "conclusion": None}],
        )
        assert decision["action"] == "observe"
        assert decision["reason"] == "same_head_run_active"


def test_dispatch_policy_handles_success_failure_and_moved_head() -> None:
    """Success/failure/moved-head fixtures are deterministic and idempotent."""
    success = dispatch_decision(
        "a" * 40,
        [
            {
                "headSha": "a" * 40,
                "event": "push",
                "status": "completed",
                "conclusion": "success",
            }
        ],
    )
    assert success["action"] == "observe"
    failed = dispatch_decision(
        "a" * 40,
        [
            {
                "headSha": "a" * 40,
                "event": "push",
                "status": "completed",
                "conclusion": "failure",
            }
        ],
        retry_failed=True,
    )
    assert failed["action"] == "dispatch"
    deduped = dispatch_decision(
        "a" * 40,
        [
            {
                "headSha": "a" * 40,
                "event": "push",
                "status": "completed",
                "conclusion": "failure",
            }
        ],
        retry_failed=True,
        retry_receipt_seen=True,
    )
    assert deduped["action"] == "observe"
    moved = dispatch_decision(
        "b" * 40,
        [{"headSha": "a" * 40, "status": "completed", "conclusion": "success"}],
    )
    assert moved["action"] == "dispatch"


def test_dispatch_gate_waits_for_recorded_queued_aggregate_with_running_jobs() -> None:
    """A queued aggregate remains the owner when some of its jobs already run (#9340)."""
    sha = "a" * 40
    decision = dispatch_gate_decision(
        sha,
        35730131902,
        [
            {
                "databaseId": 35729609257,
                "headSha": sha,
                "status": "queued",
                "conclusion": None,
                "createdAt": "2026-09-22T12:50:16Z",
                "jobs": [{"name": "compat-matrix (macos-latest, 3.11)", "status": "in_progress"}],
            },
            {
                "databaseId": 35730131902,
                "headSha": sha,
                "status": "queued",
                "conclusion": None,
                "createdAt": "2026-09-22T12:55:17Z",
            },
        ],
    )

    assert decision == {
        "action": "wait",
        "reason": "older_same_head_run_active",
        "head_sha": sha,
        "owner_run_id": 35729609257,
        "current_run_id": 35730131902,
    }


def test_dispatch_gate_waits_for_older_same_head_waiting_run() -> None:
    """GitHub's waiting workflow-run status keeps an older run as matrix owner."""
    sha = "g" * 40
    decision = dispatch_gate_decision(
        sha,
        35730131902,
        [
            {
                "databaseId": 35729609257,
                "headSha": sha,
                "status": "waiting",
                "conclusion": None,
            },
            {
                "databaseId": 35730131902,
                "headSha": sha,
                "status": "queued",
                "conclusion": None,
            },
        ],
    )

    assert decision["action"] == "wait"
    assert decision["action"] != "run_full_ci"
    assert decision["reason"] == "older_same_head_run_active"
    assert decision["owner_run_id"] == 35729609257
    assert decision["current_run_id"] == 35730131902


def test_dispatch_gate_observes_decisive_result_without_unlocking_matrix() -> None:
    """A follower may mirror exact-head evidence but cannot launch duplicate jobs."""
    sha = "b" * 40
    success = dispatch_gate_decision(
        sha,
        20,
        [
            {
                "databaseId": 10,
                "headSha": sha,
                "event": "push",
                "status": "completed",
                "conclusion": "success",
            }
        ],
    )
    failure = dispatch_gate_decision(
        sha,
        21,
        [
            {
                "databaseId": 10,
                "headSha": sha,
                "event": "push",
                "status": "completed",
                "conclusion": "failure",
            }
        ],
    )

    assert success["action"] == "observe_success"
    assert failure["action"] == "observe_failure"


@pytest.mark.parametrize("event", ["push", "pull_request", "merge_group"])
def test_dispatch_gate_accepts_only_explicit_full_matrix_events(event: str) -> None:
    """Configured non-manual triggers carry full-matrix evidence."""
    sha = "b" * 40
    decision = dispatch_gate_decision(
        sha,
        20,
        [
            {
                "databaseId": 10,
                "headSha": sha,
                "event": event,
                "status": "completed",
                "conclusion": "success",
            }
        ],
    )

    assert decision["action"] == "observe_success"


@pytest.mark.parametrize("event", [None, "", "schedule", "unknown-event"])
def test_dispatch_gate_does_not_trust_missing_or_unknown_trigger(event: str | None) -> None:
    """An incomplete REST event cannot suppress the full compatibility matrix."""
    sha = "b" * 40
    run = {
        "databaseId": 10,
        "headSha": sha,
        "status": "completed",
        "conclusion": "success",
    }
    run["event"] = event
    decision = dispatch_gate_decision(sha, 20, [run])

    assert decision["action"] == "run_full_ci"
    assert decision["reason"] == "no_same_head_decisive_run"


def test_dispatch_gate_does_not_trust_missing_trigger() -> None:
    """A missing REST event is not equivalent to a full-matrix trigger."""
    sha = "b" * 40
    decision = dispatch_gate_decision(
        sha,
        20,
        [
            {
                "databaseId": 10,
                "headSha": sha,
                "status": "completed",
                "conclusion": "success",
            }
        ],
    )

    assert decision["action"] == "run_full_ci"
    assert decision["reason"] == "no_same_head_decisive_run"


@pytest.mark.parametrize("conclusion", ["success", "failure"])
def test_dispatch_gate_does_not_treat_unadmitted_manual_result_as_decisive(
    conclusion: str,
) -> None:
    """An aggregate-only manual conclusion cannot own or suppress the matrix."""
    sha = "b" * 40
    decision = dispatch_gate_decision(
        sha,
        20,
        [
            {
                "databaseId": 10,
                "headSha": sha,
                "event": "workflow_dispatch",
                "status": "completed",
                "conclusion": conclusion,
                "fullMatrixAdmitted": False,
            }
        ],
    )

    assert decision["action"] == "run_full_ci"
    assert decision["reason"] == "no_same_head_decisive_run"


def test_dispatch_gate_failed_retry_receipt_is_idempotent() -> None:
    """A completed explicit retry is a receipt that prevents another retry."""
    sha = "c" * 40
    failed_retry = {
        "databaseId": 11,
        "headSha": sha,
        "status": "completed",
        "conclusion": "failure",
        "event": "workflow_dispatch",
        "displayTitle": "CI manual recovery retry_failed=true",
        "fullMatrixAdmitted": True,
    }

    decision = dispatch_gate_decision(sha, 20, [failed_retry], retry_failed=True)

    assert decision["action"] == "observe_failure"
    assert decision["retry_receipt_seen"] is True


def test_dispatch_gate_keeps_retry_eligible_when_prior_retry_matrix_was_skipped() -> None:
    """An all-skipped direct matrix does not consume the retry receipt."""
    sha = "c" * 40
    failed_retry = {
        "databaseId": 11,
        "headSha": sha,
        "status": "completed",
        "conclusion": "failure",
        "event": "workflow_dispatch",
        "displayTitle": "CI manual recovery retry_failed=true",
        "fullMatrixAdmitted": False,
    }

    decision = dispatch_gate_decision(sha, 20, [failed_retry], retry_failed=True)

    assert decision["action"] == "run_full_ci"
    assert decision["reason"] == "no_same_head_decisive_run"
    assert decision["retry_receipt_seen"] is False


def test_dispatch_gate_allows_first_explicit_failed_retry() -> None:
    """One explicit failed retry owns the matrix before a receipt exists."""
    sha = "e" * 40
    decision = dispatch_gate_decision(
        sha,
        20,
        [
            {
                "databaseId": 11,
                "headSha": sha,
                "status": "completed",
                "conclusion": "failure",
                "event": "push",
            }
        ],
        retry_failed=True,
    )

    assert decision["action"] == "run_full_ci"
    assert decision["reason"] == "explicit_failed_retry"
    assert decision["retry_receipt_seen"] is False


def test_dispatch_gate_follower_takes_ownership_after_stale_owner() -> None:
    """A follower starts full CI only after its older owner becomes stale."""
    sha = "d" * 40
    windows = iter(
        [
            [
                {
                    "databaseId": 10,
                    "headSha": sha,
                    "status": "queued",
                    "conclusion": None,
                }
            ],
            [
                {
                    "databaseId": 10,
                    "headSha": sha,
                    "status": "completed",
                    "conclusion": "cancelled",
                }
            ],
        ]
    )

    decision = wait_for_dispatch_gate(
        sha,
        20,
        target_ref_type="branch",
        poll_seconds=1,
        max_wait_seconds=10,
        fetcher=lambda: next(windows),
        sleeper=lambda _seconds: None,
    )

    assert decision["action"] == "run_full_ci"
    assert decision["reason"] == "no_same_head_decisive_run"


def test_dispatch_gate_timeout_is_fail_closed() -> None:
    """An owner that exceeds the wait budget never unlocks duplicate work."""
    sha = "f" * 40
    decision = wait_for_dispatch_gate(
        sha,
        20,
        target_ref_type="branch",
        poll_seconds=1,
        max_wait_seconds=0,
        fetcher=lambda: [
            {
                "databaseId": 10,
                "headSha": sha,
                "status": "in_progress",
                "conclusion": None,
            }
        ],
    )

    assert decision["action"] == "timeout"
    assert decision["owner_run_id"] == 10


def test_fetch_dispatch_window_preserves_event_and_retry_receipt_title() -> None:
    """The ownership reader retains fields needed for retry deduplication."""

    def fake_runner(path: str, payload: object = None, **_kwargs: object):
        assert payload is None
        if path.endswith("actions/workflows?per_page=100&page=1"):
            body = {"workflows": [{"id": 77, "name": "CI", "path": ".github/workflows/ci.yml"}]}
        else:
            assert path.endswith("actions/workflows/77/runs?branch=main&per_page=100&page=1")
            body = {
                "workflow_runs": [
                    {
                        "id": 11,
                        "status": "completed",
                        "conclusion": "failure",
                        "head_sha": "a" * 40,
                        "created_at": "2026-09-22T12:50:16Z",
                        "event": "workflow_dispatch",
                        "display_title": "CI manual recovery retry_failed=true",
                    }
                ]
            }
        return subprocess.CompletedProcess(["gh"], 0, json.dumps(body), "")

    runs = fetch_dispatch_run_window(runner=fake_runner)

    assert runs[0]["event"] == "workflow_dispatch"
    assert runs[0]["displayTitle"].endswith("retry_failed=true")


@pytest.mark.parametrize("active_status", ["queued", "in_progress"])
def test_fetch_dispatch_window_filters_exact_head_and_finds_active_owner(
    active_status: str,
) -> None:
    """Exact-head filtering finds queued or running ownership without scanning unrelated runs."""
    sha = "9" * 40
    older_active = _actions_run(
        {
            "databaseId": 11,
            "status": active_status,
            "conclusion": None,
            "headSha": sha,
            "createdAt": "2026-09-22T12:00:00Z",
        }
    )
    fake = _FakeRunREST([[older_active]])

    runs = fetch_dispatch_run_window(
        target_sha=sha,
        current_run_id=20,
        runner=fake,
    )
    decision = dispatch_gate_decision(sha, 20, runs)

    assert decision["action"] == "wait"
    assert decision["reason"] == "older_same_head_run_active"
    assert decision["owner_run_id"] == 11
    assert any(
        f"runs?branch=main&head_sha={sha}&per_page=100&page=1" in call for call in fake.calls
    )
    assert not any("&page=2" in call for call in fake.calls)
    assert not any("/jobs?" in call for call in fake.calls)


def test_fetch_dispatch_window_paginates_filtered_head_until_active_owner() -> None:
    """Pagination still finds an owner behind a full page for the exact SHA."""
    sha = "9" * 40
    completed_page = [
        _actions_run(
            {
                "databaseId": 1000 + index,
                "status": "completed",
                "conclusion": "cancelled",
                "headSha": sha,
                "createdAt": f"2026-09-23T00:{index:02d}:00Z",
            }
        )
        for index in range(100)
    ]
    older_active = _actions_run(
        {
            "databaseId": 11,
            "status": "in_progress",
            "conclusion": None,
            "headSha": sha,
            "createdAt": "2026-09-22T12:00:00Z",
        }
    )
    fake = _FakeRunREST([completed_page, [older_active]])

    runs = fetch_dispatch_run_window(
        target_sha=sha,
        current_run_id=20,
        runner=fake,
    )
    decision = dispatch_gate_decision(sha, 20, runs)

    assert decision["action"] == "wait"
    assert decision["owner_run_id"] == 11
    assert any(f"head_sha={sha}&per_page=100&page=2" in call for call in fake.calls)
    assert not any("/jobs?" in call for call in fake.calls)


def test_fetch_dispatch_window_fails_closed_when_page_budget_is_exhausted() -> None:
    """A full final page is not evidence that no older owner exists."""
    full_page = [
        _actions_run(
            {
                "databaseId": 1000 + index,
                "status": "completed",
                "conclusion": "cancelled",
                "headSha": "9" * 40,
                "createdAt": f"2026-09-23T00:{index:02d}:00Z",
            }
        )
        for index in range(100)
    ]
    fake = _FakeRunREST([full_page, full_page])

    with pytest.raises(MainCiRunFetchError, match="2-page budget"):
        fetch_dispatch_run_window(
            target_sha="9" * 40,
            current_run_id=20,
            max_pages=2,
            runner=fake,
        )

    assert any(call.endswith("page=2") for call in fake.calls)


@pytest.mark.parametrize(
    ("run_conclusion", "job_conclusion", "expected_admission", "expected_action"),
    [
        ("failure", "skipped", False, "run_full_ci"),
        ("failure", "failure", True, "observe_failure"),
        ("success", "skipped", False, "run_full_ci"),
        ("success", "success", True, "observe_success"),
    ],
)
def test_retry_receipt_requires_non_skipped_compatibility_job(
    run_conclusion: str,
    job_conclusion: str,
    expected_admission: bool,
    expected_action: str,
) -> None:
    """The dispatch gate requires compat-matrix admission for manual verdicts."""
    sha = "a" * 40

    def fake_runner(path: str, payload: object = None, **_kwargs: object):
        assert payload is None
        if path.endswith("actions/workflows?per_page=100&page=1"):
            body = {"workflows": [{"id": 77, "name": "CI", "path": ".github/workflows/ci.yml"}]}
        elif f"/actions/workflows/77/runs?branch=main&head_sha={sha}&per_page=100&page=1" in path:
            body = {
                "workflow_runs": [
                    {
                        "id": 11,
                        "status": "completed",
                        "conclusion": run_conclusion,
                        "head_sha": sha,
                        "created_at": "2026-09-22T12:50:16Z",
                        "event": "workflow_dispatch",
                        "display_title": "CI manual recovery retry_failed=true",
                    }
                ]
            }
        elif path == "repos/ll7/robot_sf_ll7/actions/runs/11/jobs?per_page=100&page=1":
            body = {
                "jobs": [
                    {
                        "name": "compat-matrix (ubuntu-latest, 3.11)",
                        "status": "completed",
                        "conclusion": job_conclusion,
                    }
                ]
            }
        else:
            raise AssertionError(f"unexpected REST path: {path}")
        return subprocess.CompletedProcess(["gh"], 0, json.dumps(body), "")

    runs = fetch_dispatch_run_window(
        target_sha=sha,
        current_run_id=20,
        runner=fake_runner,
    )
    decision = dispatch_gate_decision(sha, 20, runs, retry_failed=True)

    assert runs[0]["fullMatrixAdmitted"] is expected_admission
    assert decision["action"] == expected_action
    assert decision["retry_receipt_seen"] is expected_admission


@pytest.mark.parametrize(
    "jobs",
    [
        [],
        [{"name": "fast-feedback (1)", "status": "completed", "conclusion": "success"}],
    ],
    ids=["empty-page", "unrelated-job-only"],
)
def test_retry_matrix_without_recognized_compat_job_is_unknown(jobs: list[dict]) -> None:
    """A complete jobs response without compat-matrix is not proof of a skip."""

    def fake_runner(path: str, payload: object = None, **_kwargs: object):
        assert payload is None
        assert path == "repos/ll7/robot_sf_ll7/actions/runs/11/jobs?per_page=100&page=1"
        return subprocess.CompletedProcess(["gh"], 0, json.dumps({"jobs": jobs}), "")

    with pytest.raises(MainCiRunFetchError, match="no recognized compat-matrix job"):
        fetch_dispatch_retry_matrix_admitted(
            repo="ll7/robot_sf_ll7",
            run_id=11,
            runner=fake_runner,
        )


def test_retry_matrix_paginates_to_explicit_all_skipped_job() -> None:
    """A later recognized all-skipped matrix remains known-not-admitted."""
    unrelated_page = [
        {"name": f"fast-feedback ({index})", "status": "completed", "conclusion": "success"}
        for index in range(100)
    ]
    calls: list[str] = []
    pages = [
        unrelated_page,
        [
            {
                "name": "compat-matrix (ubuntu-latest, 3.11)",
                "status": "completed",
                "conclusion": "skipped",
            }
        ],
    ]

    def fake_runner(path: str, payload: object = None, **_kwargs: object):
        assert payload is None
        calls.append(path)
        page = int(path.rsplit("page=", 1)[1])
        return subprocess.CompletedProcess(["gh"], 0, json.dumps({"jobs": pages[page - 1]}), "")

    admitted = fetch_dispatch_retry_matrix_admitted(
        repo="ll7/robot_sf_ll7",
        run_id=11,
        runner=fake_runner,
        max_pages=2,
    )

    assert admitted is False
    assert calls[-1].endswith("page=2")


def test_retry_matrix_page_budget_without_compat_job_fails_closed() -> None:
    """A full bounded page without a recognized matrix cannot mean skipped."""
    full_page = [
        {"name": f"fast-feedback ({index})", "status": "completed", "conclusion": "success"}
        for index in range(100)
    ]

    def fake_runner(path: str, payload: object = None, **_kwargs: object):
        assert payload is None
        assert path.endswith("/jobs?per_page=100&page=1")
        return subprocess.CompletedProcess(["gh"], 0, json.dumps({"jobs": full_page}), "")

    with pytest.raises(MainCiRunFetchError, match="no recognized compat-matrix job"):
        fetch_dispatch_retry_matrix_admitted(
            repo="ll7/robot_sf_ll7",
            run_id=11,
            runner=fake_runner,
            max_pages=1,
        )


def test_retry_matrix_admission_lookup_failure_fails_closed() -> None:
    """An unreadable receipt cannot silently authorize another retry."""
    sha = "a" * 40

    def fake_runner(path: str, payload: object = None, **_kwargs: object):
        assert payload is None
        if path.endswith("actions/workflows?per_page=100&page=1"):
            body = {"workflows": [{"id": 77, "name": "CI", "path": ".github/workflows/ci.yml"}]}
            return subprocess.CompletedProcess(["gh"], 0, json.dumps(body), "")
        if f"/actions/workflows/77/runs?branch=main&head_sha={sha}&per_page=100&page=1" in path:
            body = {
                "workflow_runs": [
                    {
                        "id": 11,
                        "status": "completed",
                        "conclusion": "failure",
                        "head_sha": sha,
                        "created_at": "2026-09-22T12:50:16Z",
                        "event": "workflow_dispatch",
                        "display_title": "CI manual recovery retry_failed=true",
                    }
                ]
            }
            return subprocess.CompletedProcess(["gh"], 0, json.dumps(body), "")
        assert path.startswith("repos/ll7/robot_sf_ll7/actions/runs/11/jobs?")
        return subprocess.CompletedProcess(["gh"], 1, "", "API unavailable")

    with pytest.raises(MainCiRunFetchError, match="API unavailable"):
        fetch_dispatch_run_window(
            target_sha=sha,
            current_run_id=20,
            runner=fake_runner,
        )


def test_off_main_dispatch_window_observes_active_same_head_run() -> None:
    """An active run on the selected feature branch participates in ownership election."""
    sha = "f" * 40

    def fake_runner(path: str, payload: object = None, **_kwargs: object):
        assert payload is None
        if path.endswith("actions/workflows?per_page=100&page=1"):
            body = {"workflows": [{"id": 77, "name": "CI", "path": ".github/workflows/ci.yml"}]}
        else:
            assert path.endswith(
                "actions/workflows/77/runs?branch=release%2F9340-test&per_page=100&page=1"
            )
            body = {
                "workflow_runs": [
                    {
                        "id": 11,
                        "status": "in_progress",
                        "conclusion": None,
                        "head_sha": sha,
                        "created_at": "2026-09-23T12:00:00Z",
                        "event": "workflow_dispatch",
                    }
                ]
            }
        return subprocess.CompletedProcess(["gh"], 0, json.dumps(body), "")

    runs = fetch_dispatch_run_window(target_branch="release/9340-test", runner=fake_runner)
    decision = dispatch_gate_decision(sha, 20, runs)

    assert decision["action"] == "wait"
    assert decision["owner_run_id"] == 11


@pytest.mark.parametrize("target_branch", ["", " refs/heads/main", "main ", None])
def test_dispatch_window_rejects_missing_or_non_branch_ref(target_branch: str | None) -> None:
    """Missing or full-ref branch input fails closed before querying the API."""
    with pytest.raises(MainCiRunFetchError, match="target_branch"):
        fetch_dispatch_run_window(target_branch=target_branch)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("action", "expected_rc", "expected_output"),
    [
        ("run_full_ci", 0, "run_full_ci=true\n"),
        ("observe_success", 0, "run_full_ci=false\n"),
        ("observe_failure", 1, "run_full_ci=false\n"),
    ],
)
def test_dispatch_gate_cli_writes_boolean_job_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
    action: str,
    expected_rc: int,
    expected_output: str,
) -> None:
    """The workflow CLI maps each decision to a stable job output and exit status."""
    output = tmp_path / "github-output"
    monkeypatch.setattr(
        main_ci_is_green,
        "wait_for_dispatch_gate",
        lambda *_args, **_kwargs: {
            "action": action,
            "reason": "fixture",
            "head_sha": "a" * 40,
            "current_run_id": 20,
        },
    )

    rc = main_ci_is_green.main(
        [
            "--dispatch-gate",
            "--target-sha",
            "a" * 40,
            "--target-branch",
            "feature/9340-test",
            "--target-ref-type",
            "branch",
            "--current-run-id",
            "20",
            "--github-output",
            str(output),
        ]
    )

    assert rc == expected_rc
    assert output.read_text(encoding="utf-8") == expected_output
    assert json.loads(capsys.readouterr().out)["action"] == action


def test_dispatch_gate_cli_fails_closed_on_unreadable_run_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An API failure writes false and cannot unlock the expensive matrix."""
    output = tmp_path / "github-output"

    def fail(*_args: object, **_kwargs: object) -> dict:
        raise MainCiRunFetchError("fixture API failure")

    monkeypatch.setattr(main_ci_is_green, "wait_for_dispatch_gate", fail)

    rc = main_ci_is_green.main(
        [
            "--dispatch-gate",
            "--target-sha",
            "a" * 40,
            "--target-branch",
            "feature/9340-test",
            "--target-ref-type",
            "branch",
            "--current-run-id",
            "20",
            "--github-output",
            str(output),
        ]
    )

    assert rc == 1
    assert output.read_text(encoding="utf-8") == "run_full_ci=false\n"
    assert json.loads(capsys.readouterr().out)["action"] == "error"


def test_dispatch_gate_cli_fails_closed_on_tag_ref(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tag workflow_dispatch cannot unlock branch-scoped matrix work."""
    output = tmp_path / "github-output"

    rc = main_ci_is_green.main(
        [
            "--dispatch-gate",
            "--target-sha",
            "a" * 40,
            "--target-branch",
            "v0.0.7",
            "--target-ref-type",
            "tag",
            "--current-run-id",
            "20",
            "--github-output",
            str(output),
        ]
    )

    result = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert output.read_text(encoding="utf-8") == "run_full_ci=false\n"
    assert result["action"] == "error"
    assert "branch refs only" in result["reason"]


def test_unsorted_input_still_picks_newest_completed() -> None:
    """decide() sorts by createdAt, so unsorted input still resolves correctly."""
    # gh returns newest-first, but decide() sorts defensively — prove it.
    runs = [
        _run(1, "completed", "failure", "2026-07-12T10:00:00Z"),
        _run(3, "completed", "success", "2026-07-12T12:00:00Z"),
        _run(2, "completed", "failure", "2026-07-12T11:00:00Z"),
    ]
    is_green, run = decide(runs)
    assert is_green is True
    assert run["databaseId"] == 3


def test_non_mapping_entries_are_ignored() -> None:
    """Malformed list entries cannot prevent a conservative decision."""
    is_green, run = decide(
        ["not a run", None, _run(1, "completed", "success", "2026-07-12T12:00:00Z")]
    )
    assert is_green is True
    assert run is not None
    assert run["databaseId"] == 1


@pytest.mark.parametrize("payload", ["null", '{"message": "bad credentials"}'])
def test_fetch_runs_rejects_non_list_json(monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    """Unexpected API JSON reaches main's clean fail-closed path."""
    monkeypatch.setattr(
        main_ci_is_green,
        "_gh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["gh"], returncode=0, stdout=payload, stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="Unexpected JSON response type"):
        fetch_runs()


@pytest.mark.parametrize(
    ("missing_field", "message"),
    [("event", "event"), ("headSha", "head SHA"), ("createdAt", "createdAt")],
)
def test_fetch_runs_requires_signal_identity_fields(
    monkeypatch: pytest.MonkeyPatch, missing_field: str, message: str
) -> None:
    """Missing run identity fields cannot let unverified data claim green."""
    payload = [_run(1, "completed", "success", "2026-09-24T12:00:00Z")]
    payload[0].pop(missing_field)
    monkeypatch.setattr(
        main_ci_is_green,
        "_gh",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["gh"], returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match=f"no usable {message}"):
        fetch_runs()


@pytest.mark.parametrize(
    ("workflow", "expected_selector"),
    [
        (main_ci_is_green.DEFAULT_WORKFLOW, main_ci_is_green.DEFAULT_WORKFLOW_FILE),
        ("CodeQL", "CodeQL"),
    ],
)
def test_fetch_runs_uses_stable_default_workflow_selector(
    monkeypatch: pytest.MonkeyPatch, workflow: str, expected_selector: str
) -> None:
    """Default CI queries by workflow file; other requested selectors stay intact."""
    observed: dict[str, list[str]] = {}

    def fake_gh(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["args"] = args
        return subprocess.CompletedProcess(args=["gh", *args], returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(main_ci_is_green, "_gh", fake_gh)

    assert fetch_runs(workflow=workflow) == []
    args = observed["args"]
    selector_index = args.index("--workflow") + 1
    assert args[selector_index] == expected_selector
    limit_index = args.index("--limit") + 1
    assert args[limit_index] == str(main_ci_is_green.DEFAULT_MAIN_CI_RUN_LIMIT)


def test_default_workflow_selector_keeps_json_report_label(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Default CLI selection uses the file path but retains the public JSON label."""
    observed: dict[str, list[str]] = {}

    def fake_gh(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["args"] = args
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=0,
            stdout=json.dumps([_run(7, "completed", "success", "2026-07-12T12:00:00Z")]),
            stderr="",
        )

    monkeypatch.setattr(
        main_ci_is_green,
        "_gh",
        fake_gh,
    )
    monkeypatch.setattr(
        main_ci_is_green,
        "fetch_main_head_sha",
        lambda *_a, **_k: f"{7:040x}",
    )
    monkeypatch.setattr(main_ci_is_green.sys, "argv", ["main_ci_is_green.py", "--json"])

    assert main_ci_is_green.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["workflow"] == "CI"
    args = observed["args"]
    selector_index = args.index("--workflow") + 1
    assert args[selector_index] == main_ci_is_green.DEFAULT_WORKFLOW_FILE


def test_gh_oserror_becomes_a_failed_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing GitHub CLI reports not-green without a traceback."""
    monkeypatch.setattr(
        main_ci_is_green.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing gh")),
    )

    proc = main_ci_is_green._gh(["run", "list"])

    assert proc.returncode == 127
    assert "not executable" in proc.stderr


def test_build_signal_green_schema() -> None:
    """build_signal encodes a green decisive run with the machine-readable schema."""
    runs = [_run(3, "completed", "success", "2026-07-12T12:00:00Z")]
    is_green, run = decide(runs)
    signal = build_signal(is_green, run)

    assert signal["schema_version"] == "main_ci_is_green.v1"
    assert signal["is_green"] is True
    assert signal["status"] == "green"
    assert signal["deciding_run"]["databaseId"] == 3
    assert signal["deciding_run"]["conclusion"] == "success"
    assert signal["deciding_run"]["status"] == "completed"


def test_build_signal_red_schema() -> None:
    """build_signal encodes a red decisive run; is_green False, status red."""
    runs = [_run(2, "completed", "failure", "2026-07-12T12:00:00Z")]
    is_green, run = decide(runs)
    signal = build_signal(is_green, run)

    assert signal["is_green"] is False
    assert signal["status"] == "red"
    assert signal["deciding_run"]["conclusion"] == "failure"


def test_build_signal_stale_when_no_deciding_run() -> None:
    """A stale-only window yields status=stale, deciding_run=None, is_green False."""
    is_green, run = decide([_run(1, "completed", "cancelled", "2026-07-12T12:00:00Z")])
    signal = build_signal(is_green, run)

    assert is_green is False
    assert signal["is_green"] is False
    assert signal["status"] == "stale"
    assert signal["deciding_run"] is None


def test_json_output_matches_schema_and_exit_code(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The --json CLI path emits valid schema JSON and preserves the exit code.

    This is the exact gate contract from issue #5571 that previously failed with
    an argparse error: ``uv run python scripts/dev/main_ci_is_green.py --json``.
    """
    import json as _json

    sample = [_run(7, "completed", "success", "2026-07-12T12:00:00Z")]
    monkeypatch.setattr(main_ci_is_green, "fetch_runs", lambda *a, **k: sample)
    monkeypatch.setattr(main_ci_is_green, "fetch_main_head_sha", lambda *_a, **_k: f"{7:040x}")
    monkeypatch.setattr(main_ci_is_green.sys, "argv", ["main_ci_is_green.py", "--json"])

    rc = main_ci_is_green.main()

    captured = capsys.readouterr()
    payload = _json.loads(captured.out)
    assert rc == 0
    assert payload["is_green"] is True
    assert payload["status"] == "green"
    assert payload["schema_version"] == "main_ci_is_green.v1"
    assert payload["deciding_run"]["databaseId"] == 7


def test_json_signal_is_stale_if_main_moves_during_read(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A moving main ref cannot yield a green signal from a split snapshot."""
    first_head = "a" * 40
    second_head = "b" * 40
    monkeypatch.setattr(
        main_ci_is_green,
        "fetch_runs",
        lambda *_a, **_k: [
            _run(7, "completed", "success", "2026-09-24T12:00:00Z", head_sha=first_head)
        ],
    )
    heads = iter([first_head, second_head])
    monkeypatch.setattr(
        main_ci_is_green,
        "fetch_main_head_sha",
        lambda *_a, **_k: next(heads),
    )
    monkeypatch.setattr(main_ci_is_green.sys, "argv", ["main_ci_is_green.py", "--json"])

    rc = main_ci_is_green.main()
    payload = json.loads(capsys.readouterr().out)

    assert rc == 1
    assert payload["status"] == "stale"
    assert payload["deciding_run"] is None
    assert "advanced while" in payload["error"]


def test_signal_rechecks_main_after_manual_matrix_admission_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A branch advance during job-proof lookup invalidates the signal."""
    first_head = "a" * 40
    second_head = "b" * 40
    head_reads = iter([first_head, second_head])
    matrix_lookups: list[int] = []
    monkeypatch.setattr(
        main_ci_is_green,
        "fetch_main_head_sha",
        lambda *_a, **_k: next(head_reads),
    )

    def fake_fetch_runs(*_args: object, **kwargs: object) -> list[dict]:
        assert kwargs["head_sha"] == first_head
        return [
            _run(
                7,
                "completed",
                "success",
                "2026-09-24T12:00:00Z",
                event="workflow_dispatch",
                head_sha=first_head,
            )
        ]

    def fake_matrix_admission(**kwargs: object) -> bool:
        matrix_lookups.append(int(kwargs["run_id"]))
        return True

    monkeypatch.setattr(main_ci_is_green, "fetch_runs", fake_fetch_runs)
    monkeypatch.setattr(
        main_ci_is_green,
        "fetch_dispatch_retry_matrix_admitted",
        fake_matrix_admission,
    )

    with pytest.raises(MainCiRunFetchError, match="main advanced while"):
        main_ci_is_green._read_stable_main_ci_signal(
            main_ci_is_green.DEFAULT_REPO,
            main_ci_is_green.DEFAULT_WORKFLOW,
            5,
        )

    assert matrix_lookups == [7]


def test_json_fetch_failure_is_machine_readable_stale(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A fetch failure under --json still exits 1 but emits a stale JSON signal."""
    import json as _json

    monkeypatch.setattr(
        main_ci_is_green,
        "fetch_runs",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gh run list failed: boom")),
    )
    monkeypatch.setattr(main_ci_is_green, "fetch_main_head_sha", lambda *_a, **_k: "a" * 40)
    monkeypatch.setattr(main_ci_is_green.sys, "argv", ["main_ci_is_green.py", "--json"])

    rc = main_ci_is_green.main()
    captured = capsys.readouterr()
    payload = _json.loads(captured.out)

    assert rc == 1
    assert payload["status"] == "stale"
    assert payload["is_green"] is False
    assert payload["deciding_run"] is None
    assert "error" in payload


class _FakeRunREST:
    """REST fake for workflow resolution and paginated Actions run pages."""

    def __init__(self, pages: list[list[dict]]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def __call__(
        self,
        path: str,
        payload: object | None = None,
        *,
        method: str | None = None,
        extra_args: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        assert payload is None
        assert method is None
        assert extra_args is None
        self.calls.append(path)
        if path.endswith("actions/workflows?per_page=100&page=1"):
            inventory = {
                "workflows": [{"id": 77, "name": "CI", "path": ".github/workflows/ci.yml"}]
            }
            return subprocess.CompletedProcess(["gh"], 0, json.dumps(inventory), "")
        prefix = f"repos/{main_ci_is_green.DEFAULT_REPO}/actions/workflows/77/runs?branch=main&"
        if path.startswith(prefix):
            page = int(path.rsplit("&page=", 1)[1])
            return subprocess.CompletedProcess(
                ["gh"], 0, json.dumps({"workflow_runs": self.pages[page - 1]}), ""
            )
        raise AssertionError(f"unexpected REST path: {path}")


def _actions_run(run: dict) -> dict:
    """Convert a classifier-shaped run into an Actions REST row."""
    return {
        "id": run["databaseId"],
        "status": run["status"],
        "conclusion": run["conclusion"],
        "head_sha": run["headSha"],
        "created_at": run["createdAt"],
        "event": run.get("event", "push"),
    }


def test_fetch_run_window_skips_cancelled_flood_and_stops_at_decisive() -> None:
    """The paginated reader reaches the decisive verdict behind a cancelled page."""
    cancelled_page = [
        _actions_run(_run(1000 + index, "completed", "cancelled", f"2026-09-04T00:{index:02d}:00Z"))
        for index in range(100)
    ]
    decisive_page = [
        _actions_run(_run(300, "completed", "success", "2026-09-03T23:00:00Z")),
        _actions_run(_run(200, "completed", "failure", "2026-09-03T22:00:00Z")),
    ]
    fake = _FakeRunREST([cancelled_page, decisive_page])

    window = fetch_run_window(max_pages=3, stop_after_decisive=1, runner=fake)

    assert isinstance(window, MainCiRunWindow)
    assert window.window_exhausted is False
    assert [run["databaseId"] for run in window.runs[-2:]] == [300, 200]
    assert any(call.endswith("page=2") for call in fake.calls)


def test_fetch_run_window_reports_budget_exhaustion_without_a_verdict() -> None:
    """A cancelled-only window exhausting its budget is explicit, never red."""
    cancelled_pages = [
        [
            _actions_run(
                _run(
                    2000 + page * 100 + index,
                    "completed",
                    "cancelled",
                    f"2026-09-0{page}T00:{index:02d}:00Z",
                )
            )
            for index in range(100)
        ]
        for page in range(1, 3)
    ]
    fake = _FakeRunREST(cancelled_pages)

    window = fetch_run_window(max_pages=2, stop_after_decisive=1, runner=fake)

    assert window.window_exhausted is True
    assert len(window.runs) == 200
    assert all(run["conclusion"] == "cancelled" for run in window.runs)
    assert latest_completed_run(window.runs) is None


def test_fetch_run_window_malformed_page_fails_closed() -> None:
    """A non-object page payload raises instead of being guessed at."""

    def malformed_runner(path: str, payload: object = None, **kwargs: object):
        if path.endswith("actions/workflows?per_page=100&page=1"):
            inventory = {
                "workflows": [{"id": 77, "name": "CI", "path": ".github/workflows/ci.yml"}]
            }
            return subprocess.CompletedProcess(["gh"], 0, json.dumps(inventory), "")
        return subprocess.CompletedProcess(["gh"], 0, json.dumps([]), "")

    with pytest.raises(MainCiRunFetchError, match="non-object payload"):
        fetch_run_window(max_pages=1, runner=malformed_runner)


def test_raw_fetch_runs_keeps_single_bounded_limit_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raw window stays bounded without a potentially stale server filter."""
    captured: dict = {}

    def fake_gh(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        payload = [
            _run(2, "in_progress", None, "2026-09-02T00:00:00Z"),
            _run(1, "completed", "success", "2026-09-01T00:00:00Z"),
        ]
        return subprocess.CompletedProcess(["gh", *args], 0, json.dumps(payload), "")

    monkeypatch.setattr(main_ci_is_green, "_gh", fake_gh)

    expected_head = "f" * 40
    runs = fetch_runs("owner/repo", "CI", 7, head_sha=expected_head)

    assert [run["databaseId"] for run in runs] == [2, 1]
    assert "--limit" in captured["args"]
    assert captured["args"][captured["args"].index("--limit") + 1] == "7"
    assert "--commit" in captured["args"]
    assert captured["args"][captured["args"].index("--commit") + 1] == expected_head
    assert "--status" not in captured["args"]
    json_fields = captured["args"][captured["args"].index("--json") + 1]
    assert {"event", "status"} <= set(json_fields.split(","))


def test_direct_file_execution_imports_without_installed_package() -> None:
    """The CI dispatch step runs `python scripts/dev/main_ci_is_green.py` as a
    file path (issue #9676): sys.path[0] is scripts/dev, so the module must
    bootstrap the repo root itself instead of relying on ambient sys.path."""
    repo_root = main_ci_is_green.__file__
    assert repo_root is not None
    root = Path(repo_root).resolve().parents[2]
    script = root / "scripts" / "dev" / "main_ci_is_green.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Is main CI green" in proc.stdout
