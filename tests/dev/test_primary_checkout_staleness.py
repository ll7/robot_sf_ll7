"""Primary-checkout staleness detection and the preflight doctor step.

These tests build real Git repositories and observe the two public entry points
(``worktree_hygiene_snapshot.py --repo-status --json`` and
``check_skills.py --preflight``), so the expected values are derived from the
constructed repository with plain ``git`` invocations rather than from the
functions under test.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_SCRIPT = REPO_ROOT / "scripts" / "dev" / "worktree_hygiene_snapshot.py"
PREFLIGHT_SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_skills.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "commit", "--allow-empty", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _build_repo(tmp_path: Path, *, commits: int) -> tuple[Path, list[str]]:
    """Create a repo on ``main`` with ``commits`` commits and a matching origin/main ref."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "agent@example.invalid")
    _git(repo, "config", "user.name", "Agent")
    _git(repo, "checkout", "-q", "-b", "main")
    shas = [_commit(repo, f"commit {index}") for index in range(commits)]
    _git(repo, "update-ref", "refs/remotes/origin/main", shas[-1])
    return repo, shas


def _run_snapshot(repo: Path) -> dict:
    result = subprocess.run(
        [
            sys_executable(),
            str(SNAPSHOT_SCRIPT),
            "--repo-status",
            "--filter",
            "nonexistent",
            "--json",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def sys_executable() -> str:
    import sys

    return sys.executable


def _checkout_status(payload: dict) -> str | None:
    return (payload.get("primary_checkout") or {}).get("status")


def _run_preflight(repo: Path) -> tuple[int, dict]:
    result = subprocess.run(
        [sys_executable(), str(PREFLIGHT_SCRIPT), "--preflight", "goal-autopilot", "--json"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return result.returncode, payload


def test_detached_head_behind_origin_main_is_reported_stale(tmp_path: Path) -> None:
    """A detached HEAD missing commits from origin/main is named, not left to be found."""
    repo, shas = _build_repo(tmp_path, commits=3)
    _git(repo, "checkout", "-q", "--detach", shas[0])

    repo_status = _run_snapshot(repo)["repo_status"]

    # Derived independently of the code under test.
    assert _git(repo, "rev-parse", "HEAD") == shas[0]
    assert _git(repo, "rev-list", "--count", "HEAD..origin/main") == "2"
    assert repo_status.get("stale_primary_checkout") is True, (
        "a detached HEAD two commits behind origin/main must be reported stale; got "
        f"{repo_status.get('stale_primary_checkout')!r} in {repo_status!r}"
    )
    assert repo_status.get("head_attached") is False
    assert repo_status.get("head_behind_origin_main") == 2
    assert repo_status.get("head_sha") == shas[0]


def test_detached_head_current_with_origin_main_is_not_stale(tmp_path: Path) -> None:
    """A detached HEAD already at origin/main is current, not stale."""
    repo, shas = _build_repo(tmp_path, commits=2)
    _git(repo, "checkout", "-q", "--detach", shas[-1])

    repo_status = _run_snapshot(repo)["repo_status"]

    assert repo_status.get("stale_primary_checkout") is False, (
        "a detached HEAD already at origin/main is current; got "
        f"{repo_status.get('stale_primary_checkout')!r} in {repo_status!r}"
    )
    assert repo_status.get("head_attached") is False
    assert repo_status.get("head_behind_origin_main") == 0


def test_attached_branch_behind_origin_main_is_not_flagged_stale(tmp_path: Path) -> None:
    """The documented rule is detached *and* behind; an attached branch is not flagged."""
    repo, shas = _build_repo(tmp_path, commits=3)
    _git(repo, "checkout", "-q", "-b", "topic", shas[0])

    repo_status = _run_snapshot(repo)["repo_status"]

    assert repo_status.get("stale_primary_checkout") is False, (
        "the rule is detached AND behind; an attached branch two commits back is not stale; got "
        f"{repo_status.get('stale_primary_checkout')!r} in {repo_status!r}"
    )
    assert repo_status.get("head_attached") is True
    assert repo_status.get("head_behind_origin_main") == 2


def test_missing_origin_main_ref_does_not_report_healthy(tmp_path: Path) -> None:
    """An unresolvable origin/main is unknown, never silently reported as up to date."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo.parent, "init", "-q", str(repo))
    _git(repo, "config", "user.email", "agent@example.invalid")
    _git(repo, "config", "user.name", "Agent")
    _git(repo, "checkout", "-q", "-b", "main")
    _commit(repo, "only commit")
    _git(repo, "checkout", "-q", "--detach", "HEAD")

    repo_status = _run_snapshot(repo)["repo_status"]

    assert repo_status.get("origin_main_sha") is None
    assert repo_status.get("head_behind_origin_main") is None
    assert repo_status.get("stale_primary_checkout") is False, (
        "a detached HEAD with no resolvable origin/main cannot be proven current"
    )
    code, payload = _run_preflight(repo)
    assert _checkout_status(payload) == "unknown", (
        f"unresolvable origin/main must fail closed; got {_checkout_status(payload)!r}"
    )
    assert code == 1


def test_preflight_fails_closed_on_stale_primary_checkout(tmp_path: Path) -> None:
    """Preflight refuses to certify a lane from a detached, behind checkout."""
    repo, shas = _build_repo(tmp_path, commits=3)
    _git(repo, "checkout", "-q", "--detach", shas[0])

    code, payload = _run_preflight(repo)

    assert code == 1, (
        "preflight must refuse to certify a lane from a detached checkout 2 commits behind "
        f"origin/main; it exited {code} with {payload!r}"
    )
    assert _checkout_status(payload) == "stale", payload
    assert payload.get("summary", {}).get("missing") == 0, (
        "the failure must come from the checkout, not a missing tool"
    )
    assert payload.get("summary", {}).get("unrecognized") == 0
    assert payload.get("status") == "fail"
    assert "git checkout --detach origin/main" in payload.get("primary_checkout", {}).get(
        "detail", ""
    )


def test_preflight_passes_once_the_checkout_is_current(tmp_path: Path) -> None:
    """The same preflight passes on an attached current checkout, isolating the variable."""
    repo, _shas = _build_repo(tmp_path, commits=2)

    code, payload = _run_preflight(repo)

    assert code == 0, (
        "the same preflight must pass once the checkout is current, so the stale verdict is the "
        f"only variable; it exited {code} with {payload!r}"
    )
    assert _checkout_status(payload) == "ok", payload
    assert payload.get("summary", {}).get("missing") == 0
    assert payload.get("status") == "ok"


def _rendered_repo_status(repo: Path) -> str:
    """Render the human-readable fleet view from inside ``repo``.

    Uses ``build_snapshot`` so the assertion exercises the real path from Git state
    to rendered text, with no test-only construction of the dataclasses.
    """
    import os

    from scripts.dev.worktree_hygiene_snapshot import build_snapshot, format_human

    previous = Path.cwd()
    os.chdir(repo)
    try:
        snapshot = build_snapshot(
            filters=["no-such-worktree"],
            include_repo_status=True,
        )
    finally:
        os.chdir(previous)
    return format_human(snapshot)


def test_repo_status_markdown_names_the_repair_command(tmp_path: Path) -> None:
    """The human-readable fleet view carries the repair, not just a boolean."""
    repo, shas = _build_repo(tmp_path, commits=3)
    _git(repo, "checkout", "-q", "--detach", shas[0])

    rendered = _rendered_repo_status(repo)

    assert "stale_primary_checkout=True" in rendered, rendered
    assert "behind_origin_main=2" in rendered, rendered
    assert "git checkout --detach origin/main" in rendered, rendered


def test_repo_status_markdown_omits_the_repair_when_current(tmp_path: Path) -> None:
    """A current checkout must not be told to run a repair it does not need."""
    repo, _shas = _build_repo(tmp_path, commits=2)

    rendered = _rendered_repo_status(repo)

    assert "stale_primary_checkout=False" in rendered, rendered
    assert "git checkout --detach origin/main" not in rendered, rendered
