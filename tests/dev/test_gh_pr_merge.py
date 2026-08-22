"""Behavior tests for the guarded gh_pr_merge.sh wrapper (issue #7733)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GH_PR_MERGE = REPO_ROOT / "scripts" / "dev" / "gh_pr_merge.sh"

FULL_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"


def _fake_gh_bin(tmp_path: Path) -> Path:
    """Write a fake ``gh`` that dispatches on subcommand to scripted outputs.

    The fake is controlled through FAKE_GH_PLAN, a JSON mapping from a short
    key (``merge_ok``, ``worktree_conflict``, ``other_error``, ``repo_view``,
    ``pr_view``, ``rest_merge_ok``, ``rest_merge_error``, ``branch_delete_ok``,
    ``branch_delete_fail``) to ``{"stdout": ..., "stderr": ..., "exit": n}``.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_gh = bin_dir / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "plan = json.load(open(os.environ['FAKE_GH_PLAN'], encoding='utf-8'))\n"
        "args = sys.argv[1:]\n"
        "key = None\n"
        "if args[:3] == ['pr', 'merge', '1234']:\n"
        "    key = plan['merge_key']\n"
        "elif args[:2] == ['repo', 'view']:\n"
        "    key = 'repo_view'\n"
        "elif args[:3] == ['pr', 'view', '1234']:\n"
        "    key = 'pr_view'\n"
        "    if '--jq' in args:\n"
        "        field = args[args.index('--jq') + 1].lstrip('.')\n"
        "        data = json.loads(plan['pr_view']['stdout'])\n"
        "        print(data.get(field, ''))\n"
        "        sys.exit(0)\n"
        "elif args[:3] == ['api', '-X', 'PUT']:\n"
        "    key = plan['rest_key']\n"
        "elif args[:5] == ['api', '-X', 'DELETE', 'repos/o/r/git/refs/heads/branch']:\n"
        "    key = 'branch_delete_ok'\n"
        "else:\n"
        "    print(f'unexpected: {args}', file=sys.stderr)\n"
        "    sys.exit(99)\n"
        "resp = plan[key]\n"
        "if resp.get('stdout'):\n"
        "    print(resp['stdout'])\n"
        "if resp.get('stderr'):\n"
        "    print(resp['stderr'], file=sys.stderr)\n"
        "sys.exit(resp.get('exit', 0))\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    return bin_dir


def _run_wrapper(tmp_path: Path, plan: dict[str, object]) -> subprocess.CompletedProcess[str]:
    """Run gh_pr_merge.sh with a scripted fake gh on PATH."""
    plan_path = tmp_path / "fake_gh_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    env = os.environ.copy()
    env["FAKE_GH_PLAN"] = str(plan_path)
    env["PATH"] = str(_fake_gh_bin(tmp_path)) + os.pathsep + env.get("PATH", "")
    return subprocess.run(
        [
            str(GH_PR_MERGE),
            "1234",
            "--match-head-commit",
            FULL_SHA,
            "--repo",
            "o/r",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )


def test_merge_success_path_uses_native_gh(tmp_path: Path) -> None:
    """When gh pr merge succeeds the wrapper exits 0 with no REST call."""
    result = _run_wrapper(
        tmp_path,
        plan={"merge_key": "merge_ok", "merge_ok": {"stdout": "Merged", "exit": 0}},
    )
    assert result.returncode == 0, result.stderr


def test_worktree_conflict_falls_back_to_rest(tmp_path: Path) -> None:
    """The issue #7733 signature triggers the exact-head REST squash merge."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "worktree_conflict",
            "worktree_conflict": {
                "stderr": (
                    "failed to run git: fatal: 'main' is already used by "
                    "worktree at '/home/o/r.worktrees/other'"
                ),
                "exit": 1,
            },
            "repo_view": {"stdout": "o/r", "exit": 0},
            "pr_view": {"stdout": json.dumps({"headRefOid": FULL_SHA}), "exit": 0},
            "rest_key": "rest_merge_ok",
            "rest_merge_ok": {
                "stdout": json.dumps({"merged": True, "sha": "c0ffee" * 5, "message": "ok"}),
                "exit": 0,
            },
            "branch_delete_ok": {"exit": 0},
        },
    )
    assert result.returncode == 0, result.stderr
    assert "retrying through REST" in result.stderr
    assert "Merged via REST fallback" in result.stderr


def test_rest_fallback_tolerates_branch_delete_failure(tmp_path: Path) -> None:
    """A failed remote branch deletion after the squash is non-fatal."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "worktree_conflict",
            "worktree_conflict": {
                "stderr": "already used by worktree at '/x'",
                "exit": 1,
            },
            "repo_view": {"stdout": "o/r", "exit": 0},
            "pr_view": {
                "stdout": json.dumps({"headRefOid": FULL_SHA, "headRefName": "branch"}),
                "exit": 0,
            },
            "rest_key": "rest_merge_ok",
            "rest_merge_ok": {
                "stdout": json.dumps({"merged": True, "sha": "c0ffee" * 5, "message": "ok"}),
                "exit": 0,
            },
            "branch_delete_ok": {"exit": 1, "stderr": "ref does not exist"},
        },
    )
    assert result.returncode == 0, result.stderr
    assert "could not delete remote branch" in result.stderr


def test_non_worktree_failure_stays_fail_closed(tmp_path: Path) -> None:
    """Any other merge failure exits nonzero with the raw diagnostic."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "other_error",
            "other_error": {"stderr": "gh: SomethingElse failed", "exit": 1},
        },
    )
    assert result.returncode == 1
    assert "SomethingElse failed" in result.stderr
    assert "REST fallback" not in result.stderr


def test_rest_fallback_refuses_stale_head(tmp_path: Path) -> None:
    """If the live head moved past the expected binding, do not merge."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "worktree_conflict",
            "worktree_conflict": {
                "stderr": "fatal: 'main' is already used by worktree at '/x'",
                "exit": 1,
            },
            "repo_view": {"stdout": "o/r", "exit": 0},
            "pr_view": {"stdout": json.dumps({"headRefOid": "f" * 40}), "exit": 0},
        },
    )
    assert result.returncode == 2
    assert "refuses stale head" in result.stderr
