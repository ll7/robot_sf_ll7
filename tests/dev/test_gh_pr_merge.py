"""Behavior tests for the guarded gh_pr_merge.sh wrapper (issues #7733 and #8132)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GH_PR_MERGE = REPO_ROOT / "scripts" / "dev" / "gh_pr_merge.sh"

FULL_SHA = "a1b2c3d4e5f60718293a4b5c6d7e8f9001020304"


def _rest_preflight(
    *,
    head: str = FULL_SHA,
    state: str = "open",
    draft: bool = False,
    mergeable: bool = True,
    mergeable_state: str = "clean",
) -> str:
    return "\t".join(
        (
            head,
            "branch",
            state,
            str(draft).lower(),
            str(mergeable).lower(),
            mergeable_state,
        )
    )


def _fake_gh_bin(tmp_path: Path) -> Path:
    """Write a fake ``gh`` that dispatches on subcommand to scripted outputs.

    The fake is controlled through FAKE_GH_PLAN, a JSON mapping from a short
    key (``merge_ok``, ``worktree_conflict``, ``other_error``, ``repo_view``,
    ``pr_view``, ``rest_preflight``, ``rest_labels``, ``rest_labels_page_N``,
    ``rest_merge_ok``, ``rest_merge_error``, ``branch_delete_ok``,
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
        "elif args[:2] == ['api', 'repos/o/r/pulls/1234']:\n"
        "    key = 'rest_preflight'\n"
        "elif len(args) >= 2 and args[0] == 'api' and args[1].startswith(\n"
        "    'repos/o/r/issues/1234/labels?'\n"
        "):\n"
        "    page = args[1].rsplit('page=', 1)[-1]\n"
        "    key = f'rest_labels_page_{page}'\n"
        "    if key not in plan:\n"
        "        key = 'rest_labels'\n"
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


def _run_wrapper(
    tmp_path: Path,
    plan: dict[str, object],
    *,
    include_repo_arg: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run gh_pr_merge.sh with a scripted fake gh on PATH."""
    plan_path = tmp_path / "fake_gh_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    env = os.environ.copy()
    env["FAKE_GH_PLAN"] = str(plan_path)
    env["PATH"] = str(_fake_gh_bin(tmp_path)) + os.pathsep + env.get("PATH", "")
    args = [str(GH_PR_MERGE), "1234", "--match-head-commit", FULL_SHA]
    if include_repo_arg:
        args.extend(("--repo", "o/r"))
    return subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )


def _successful_quota_plan() -> dict[str, object]:
    return {
        "merge_key": "graphql_quota",
        "graphql_quota": {
            "stderr": "GraphQL: API rate limit already exceeded for user ID 123.",
            "exit": 1,
        },
        "rest_preflight": {"stdout": _rest_preflight(), "exit": 0},
        "rest_labels": {"stdout": "ok\t1\ttrue", "exit": 0},
        "rest_key": "rest_merge_ok",
        "rest_merge_ok": {
            "stdout": json.dumps({"merged": True, "sha": "c0ffee" * 5, "message": "ok"}),
            "exit": 0,
        },
        "branch_delete_ok": {"exit": 0},
    }


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


def test_graphql_quota_falls_back_after_rest_guard_reverification(tmp_path: Path) -> None:
    """Quota exhaustion retries only after the REST guard snapshot is merge-ready."""
    result = _run_wrapper(tmp_path, _successful_quota_plan())
    assert result.returncode == 0, result.stderr
    assert "GraphQL quota exhaustion" in result.stderr
    assert "re-verifying guarded state through REST" in result.stderr
    assert "Merged via REST fallback" in result.stderr


def test_quota_fallback_reads_merge_ready_from_second_label_page(tmp_path: Path) -> None:
    """The authority label remains visible when it is not on the first REST page."""
    plan = _successful_quota_plan()
    plan.pop("rest_labels")
    plan["rest_labels_page_1"] = {"stdout": "ok\t100\tfalse", "exit": 0}
    plan["rest_labels_page_2"] = {"stdout": "ok\t1\ttrue", "exit": 0}
    result = _run_wrapper(tmp_path, plan)
    assert result.returncode == 0, result.stderr
    assert "Merged via REST fallback" in result.stderr


def test_quota_fallback_resolves_repo_from_git_origin(tmp_path: Path) -> None:
    """Omitting --repo must not force GraphQL-backed repository discovery."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "git@github.com:o/r.git"],
        check=True,
    )
    result = _run_wrapper(
        tmp_path,
        _successful_quota_plan(),
        include_repo_arg=False,
        cwd=checkout,
    )
    assert result.returncode == 0, result.stderr
    assert "Merged via REST fallback" in result.stderr


def test_quota_fallback_rejects_non_github_origin(tmp_path: Path) -> None:
    """Repository discovery must not reinterpret a non-GitHub origin."""
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", "git@gitlab.com:o/r.git"],
        check=True,
    )
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "graphql_quota",
            "graphql_quota": {"stderr": "GraphQL: API quota exhausted", "exit": 1},
            "repo_view": {"stderr": "GraphQL: API quota exhausted", "exit": 1},
        },
        include_repo_arg=False,
        cwd=checkout,
    )
    assert result.returncode == 2
    assert "cannot resolve owner/name" in result.stderr


def test_quota_fallback_refuses_stale_head(tmp_path: Path) -> None:
    """A moved head fails before the REST compare-and-swap merge is attempted."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "graphql_quota",
            "graphql_quota": {"stderr": "GraphQL: API rate limit exceeded", "exit": 1},
            "rest_preflight": {"stdout": _rest_preflight(head="f" * 40), "exit": 0},
        },
    )
    assert result.returncode == 2
    assert "refuses stale head" in result.stderr


def test_quota_fallback_refuses_draft_pr(tmp_path: Path) -> None:
    """A live draft state cannot inherit stale merge-ready authority."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "graphql_quota",
            "graphql_quota": {"stderr": "GraphQL: API rate limit exceeded", "exit": 1},
            "rest_preflight": {"stdout": _rest_preflight(draft=True), "exit": 0},
        },
    )
    assert result.returncode == 2
    assert "refuses PR state" in result.stderr


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


@pytest.mark.parametrize(
    "diagnostic",
    (
        "GraphQL: Something went wrong resolving PullRequest.",
        "GraphQL: Repository quota metadata is unavailable.",
    ),
)
def test_non_quota_graphql_failure_stays_fail_closed(
    tmp_path: Path, diagnostic: str
) -> None:
    """Generic GraphQL and incidental quota text are not fallback eligible."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "graphql_error",
            "graphql_error": {"stderr": diagnostic, "exit": 1},
        },
    )
    assert result.returncode == 1
    assert diagnostic in result.stderr
    assert "re-verifying guarded state" not in result.stderr


@pytest.mark.parametrize(
    "diagnostic",
    (
        "GraphQL: Bad credentials; API rate limit already exceeded.",
        "GraphQL: Could not resolve to a Repository with the name 'o/missing'; quota unavailable.",
    ),
)
def test_auth_and_repository_failures_win_over_quota_markers(
    tmp_path: Path, diagnostic: str
) -> None:
    """Authority and repository failures remain fail-closed even with quota text."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "fail_closed",
            "fail_closed": {"stderr": diagnostic, "exit": 1},
        },
    )
    assert result.returncode == 1
    assert diagnostic in result.stderr
    assert "re-verifying guarded state" not in result.stderr


def test_quota_fallback_refuses_missing_merge_ready_label(tmp_path: Path) -> None:
    """The REST fallback must re-check live merge authority before merging."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "graphql_quota",
            "graphql_quota": {"stderr": "GraphQL: API rate limit exceeded", "exit": 1},
            "rest_preflight": {"stdout": _rest_preflight(), "exit": 0},
            "rest_labels": {"stdout": "ok\t0\tfalse", "exit": 0},
        },
    )
    assert result.returncode == 2
    assert "without the merge-ready label" in result.stderr


def test_quota_fallback_refuses_malformed_label_inventory(tmp_path: Path) -> None:
    """Malformed authority-label data cannot be treated as a complete inventory."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "graphql_quota",
            "graphql_quota": {"stderr": "GraphQL: API rate limit exceeded", "exit": 1},
            "rest_preflight": {"stdout": _rest_preflight(), "exit": 0},
            "rest_labels": {"stdout": "error\tmalformed-label-row\t", "exit": 0},
        },
    )
    assert result.returncode == 1
    assert "label page 1 was malformed" in result.stderr


def test_quota_fallback_refuses_non_clean_mergeability(tmp_path: Path) -> None:
    """Conflicting or otherwise non-clean REST mergeability fails closed."""
    result = _run_wrapper(
        tmp_path,
        {
            "merge_key": "graphql_quota",
            "graphql_quota": {"stderr": "GraphQL: API rate limit exceeded", "exit": 1},
            "rest_preflight": {
                "stdout": _rest_preflight(mergeable=False, mergeable_state="dirty"),
                "exit": 0,
            },
        },
    )
    assert result.returncode == 2
    assert "refuses non-clean mergeability" in result.stderr


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
            "pr_view": {"stdout": json.dumps({"headRefOid": "f" * 40}), "exit": 0},
        },
    )
    assert result.returncode == 2
    assert "refuses stale head" in result.stderr
