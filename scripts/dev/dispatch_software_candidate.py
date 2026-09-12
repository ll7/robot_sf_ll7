#!/usr/bin/env python3
"""Dispatch the software-candidate workflow from one exact commit.

GitHub Actions workflow dispatch accepts branch and tag refs, but not a raw
commit SHA.  This helper creates a short-lived branch at the requested commit,
dispatches the candidate workflow with an explicit SHA input, waits for the
run to become visible (and optionally complete), then removes the temporary
branch.  All GitHub API writes are made through the user's authenticated
``gh`` CLI; no token is read, echoed, or stored by this script.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.dev._gh_rest import parse_json, run_gh_api

SCHEMA = "robot_sf.software_candidate_dispatch.v1"
DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_WORKFLOW = "software-candidate.yml"
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
REPO_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
TERMINAL_STATUSES = frozenset({"completed"})


class DispatchError(ValueError):
    """Raised when exact-source dispatch cannot be proven."""


def _validate_inputs(repo: str, source_sha: str, workflow: str) -> None:
    if not REPO_RE.fullmatch(repo):
        raise DispatchError("repository must be an exact OWNER/REPO identity")
    if not SHA_RE.fullmatch(source_sha):
        raise DispatchError("source SHA must be one exact lowercase 40-hex commit identity")
    if not workflow or "/" in workflow or not re.fullmatch(r"[A-Za-z0-9_.-]+", workflow):
        raise DispatchError("workflow must be a workflow filename or numeric ID")


def _api_json(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Any:
    result = run_gh_api(path, payload, method=None if method == "GET" else method, timeout=timeout)
    value, error = parse_json(result, what=f"GitHub API {method} {path}")
    if error:
        raise DispatchError(error)
    return value


def _api_write(path: str, *, method: str, payload: dict[str, Any] | None = None) -> Any:
    result = run_gh_api(path, payload, method=method, timeout=45)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise DispatchError(f"GitHub API {method} {path} failed: {detail}")
    if not result.stdout.strip():
        return None
    value, error = parse_json(result, what=f"GitHub API {method} {path}")
    if error:
        raise DispatchError(error)
    return value


def _commit_exists(repo: str, source_sha: str) -> None:
    payload = _api_json(f"repos/{repo}/commits/{source_sha}")
    if not isinstance(payload, dict) or payload.get("sha") != source_sha:
        raise DispatchError("GitHub did not confirm the requested commit SHA")


def _branch_name(source_sha: str) -> str:
    # The random suffix prevents two operators from sharing or deleting one
    # another's temporary ref while the source SHA remains visibly bound.
    return f"automation/software-candidate/{source_sha[:12]}-{secrets.token_hex(5)}"


def _create_ref(repo: str, branch: str, source_sha: str) -> None:
    payload = _api_write(
        f"repos/{repo}/git/refs",
        method="POST",
        payload={"ref": f"refs/heads/{branch}", "sha": source_sha},
    )
    if not isinstance(payload, dict):
        raise DispatchError("GitHub did not return the created temporary ref")
    obj = payload.get("object")
    if not isinstance(obj, dict) or obj.get("sha") != source_sha:
        raise DispatchError("temporary ref did not resolve to the requested source SHA")


def _delete_ref(repo: str, branch: str) -> None:
    _api_write(f"repos/{repo}/git/refs/heads/{branch}", method="DELETE")


def _dispatch(repo: str, workflow: str, branch: str, source_sha: str) -> None:
    _api_write(
        f"repos/{repo}/actions/workflows/{workflow}/dispatches",
        method="POST",
        payload={"ref": branch, "inputs": {"requested_source_sha": source_sha}},
    )


def _run_candidates(repo: str, workflow: str, branch: str, source_sha: str) -> list[dict[str, Any]]:
    payload = _api_json(
        f"repos/{repo}/actions/workflows/{workflow}/runs?event=workflow_dispatch&branch={branch}&per_page=20"
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("workflow_runs"), list):
        raise DispatchError("workflow-runs response was not a valid object")
    runs = [run for run in payload["workflow_runs"] if isinstance(run, dict)]
    return [
        run
        for run in runs
        if run.get("head_branch") == branch
        and run.get("head_sha") == source_sha
        and run.get("event") == "workflow_dispatch"
    ]


def _run(repo: str, run_id: int) -> dict[str, Any]:
    payload = _api_json(f"repos/{repo}/actions/runs/{run_id}")
    if not isinstance(payload, dict):
        raise DispatchError("workflow-run response was not an object")
    return payload


def _await_run(
    repo: str,
    run: dict[str, Any],
    *,
    run_timeout: int,
    poll_seconds: int,
) -> dict[str, Any]:
    """Poll one exact workflow run until GitHub reports a terminal status."""
    run_id = run.get("id")
    if not isinstance(run_id, int) or run_id < 1:
        raise DispatchError("workflow run ID is invalid")
    deadline = time.monotonic() + run_timeout
    while run.get("status") not in TERMINAL_STATUSES and time.monotonic() < deadline:
        time.sleep(poll_seconds)
        run = _run(repo, run_id)
    if run.get("status") not in TERMINAL_STATUSES:
        raise DispatchError("workflow run did not reach a terminal state before timeout")
    return run


def _identity_payload(
    *,
    repo: str,
    workflow: str,
    branch: str,
    source_sha: str,
    run: dict[str, Any] | None,
    ref_deleted: bool,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "repository": repo,
        "workflow": workflow,
        "temporary_ref": f"refs/heads/{branch}",
        "requested_source_sha": source_sha,
        "observed_source_sha": (run or {}).get("head_sha"),
        "workflow_run": {
            "id": (run or {}).get("id"),
            "status": (run or {}).get("status"),
            "conclusion": (run or {}).get("conclusion"),
            "url": (run or {}).get("html_url"),
        },
        "temporary_ref_deleted": ref_deleted,
    }


def dispatch(
    *,
    repo: str,
    workflow: str,
    source_sha: str,
    discovery_timeout: int,
    wait: bool,
    run_timeout: int,
    poll_seconds: int,
) -> dict[str, Any]:
    """Create the exact ref, dispatch one run, and optionally await completion."""
    _validate_inputs(repo, source_sha, workflow)
    if discovery_timeout < 1 or run_timeout < 1 or poll_seconds < 1:
        raise DispatchError("timeouts and poll interval must be positive")
    _commit_exists(repo, source_sha)
    branch = _branch_name(source_sha)
    _create_ref(repo, branch, source_sha)
    run: dict[str, Any] | None = None
    receipt: dict[str, Any] | None = None
    try:
        _dispatch(repo, workflow, branch, source_sha)
        discovery_deadline = time.monotonic() + discovery_timeout
        while time.monotonic() < discovery_deadline:
            matches = _run_candidates(repo, workflow, branch, source_sha)
            if matches:
                run = max(matches, key=lambda item: int(item.get("id", 0)))
                break
            time.sleep(poll_seconds)
        if run is None:
            raise DispatchError("workflow dispatch was not observed at the exact source SHA")

        if wait:
            run = _await_run(
                repo,
                run,
                run_timeout=run_timeout,
                poll_seconds=poll_seconds,
            )
        receipt = _identity_payload(
            repo=repo,
            workflow=workflow,
            branch=branch,
            source_sha=source_sha,
            run=run,
            ref_deleted=False,
        )
        return receipt
    finally:
        # A running workflow may still need its event ref for checkout.  Keep
        # the ref for ``--no-wait`` callers and require them to delete it after
        # the run reaches a terminal state.
        if wait:
            try:
                _delete_ref(repo, branch)
                if receipt is not None:
                    receipt["temporary_ref_deleted"] = True
            except DispatchError as exc:
                # Preserve the original failure and make cleanup failure visible
                # to successful callers through the final receipt.
                if run is not None:
                    raise DispatchError(f"temporary ref cleanup failed: {exc}") from exc
                raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--discovery-timeout", type=int, default=120)
    parser.add_argument("--run-timeout", type=int, default=4 * 60 * 60)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Return after the exact run is observed; keep the ref until the run is terminal.",
    )
    parser.add_argument("--receipt", type=Path, help="Optional JSON receipt path outside the repo.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, dispatch one exact-source run, and print its receipt."""
    args = _parser().parse_args(argv)
    try:
        receipt = dispatch(
            repo=args.repo,
            workflow=args.workflow,
            source_sha=args.source_sha,
            discovery_timeout=args.discovery_timeout,
            wait=not args.no_wait,
            run_timeout=args.run_timeout,
            poll_seconds=args.poll_seconds,
        )
        if args.receipt is not None:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["observed_source_sha"] == receipt["requested_source_sha"] else 1
    except (DispatchError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
