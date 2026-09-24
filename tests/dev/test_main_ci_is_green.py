"""Tests for the main-CI green/red signal used by the red-main merge hold (#5385).

The load-bearing property: an IN-PROGRESS run must never decide green or red —
only the most recent *completed* run does. This is the exact bug that made the
escalation guard silent on 2026-07-11 (it counted an in-progress newest run),
so it gets an explicit test.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from scripts.dev import main_ci_is_green
from scripts.dev.main_ci_is_green import (
    MainCiRunFetchError,
    MainCiRunWindow,
    build_signal,
    classify,
    decide,
    dispatch_decision,
    dispatch_gate_decision,
    fetch_dispatch_run_window,
    fetch_run_window,
    fetch_runs,
    latest_completed_run,
    wait_for_dispatch_gate,
)


def _run(rid: int, status: str, conclusion: str | None, created: str) -> dict:
    return {
        "databaseId": rid,
        "status": status,
        "conclusion": conclusion,
        "headSha": f"{rid:040x}",
        "createdAt": created,
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


def test_stale_only_window_is_not_green_and_has_no_deciding_run() -> None:
    """A window of only stale completed runs yields no decisive verdict (fail closed)."""
    for stale in ("cancelled", "timed_out", "startup_failure", "skipped", "neutral", None):
        is_green, run = decide([_run(1, "completed", stale, "2026-07-12T12:00:00Z")])
        assert is_green is False, stale
        assert run is None, stale  # stale is skipped -> no deciding run


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
        [{"headSha": "a" * 40, "status": "completed", "conclusion": "success"}],
    )
    assert success["action"] == "observe"
    failed = dispatch_decision(
        "a" * 40,
        [{"headSha": "a" * 40, "status": "completed", "conclusion": "failure"}],
        retry_failed=True,
    )
    assert failed["action"] == "dispatch"
    deduped = dispatch_decision(
        "a" * 40,
        [{"headSha": "a" * 40, "status": "completed", "conclusion": "failure"}],
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
                "status": "completed",
                "conclusion": "failure",
            }
        ],
    )

    assert success["action"] == "observe_success"
    assert failure["action"] == "observe_failure"


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
    }

    decision = dispatch_gate_decision(sha, 20, [failed_retry], retry_failed=True)

    assert decision["action"] == "observe_failure"
    assert decision["retry_receipt_seen"] is True


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
            "--current-run-id",
            "20",
            "--github-output",
            str(output),
        ]
    )

    assert rc == 1
    assert output.read_text(encoding="utf-8") == "run_full_ci=false\n"
    assert json.loads(capsys.readouterr().out)["action"] == "error"


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
    monkeypatch.setattr(main_ci_is_green.sys, "argv", ["main_ci_is_green.py", "--json"])

    rc = main_ci_is_green.main()

    captured = capsys.readouterr()
    payload = _json.loads(captured.out)
    assert rc == 0
    assert payload["is_green"] is True
    assert payload["status"] == "green"
    assert payload["schema_version"] == "main_ci_is_green.v1"
    assert payload["deciding_run"]["databaseId"] == 7


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
        prefix = (
            f"repos/{main_ci_is_green.DEFAULT_REPO}"
            "/actions/workflows/77/runs?branch=main&per_page=100&page="
        )
        if path.startswith(prefix):
            page = int(path.removeprefix(prefix))
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
    """The legacy raw window still issues exactly one bounded gh run list call."""
    captured: dict = {}

    def fake_gh(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        payload = [_run(1, "completed", "success", "2026-09-01T00:00:00Z")]
        return subprocess.CompletedProcess(["gh", *args], 0, json.dumps(payload), "")

    monkeypatch.setattr(main_ci_is_green, "_gh", fake_gh)

    runs = fetch_runs("owner/repo", "CI", 7)

    assert [run["databaseId"] for run in runs] == [1]
    assert "--limit" in captured["args"]
    assert captured["args"][captured["args"].index("--limit") + 1] == "7"
    assert "--status" in captured["args"]
    assert captured["args"][captured["args"].index("--status") + 1] == "completed"
