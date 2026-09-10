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
    fetch_run_window,
    fetch_runs,
    latest_completed_run,
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
