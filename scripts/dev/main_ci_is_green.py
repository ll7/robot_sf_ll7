#!/usr/bin/env python3
"""Report whether ``main`` CI is green, based on the latest COMPLETED run.

Motivation (issue #5385). Three separate main-red incidents in 36h
(2026-07-11/12) shared one mechanism: merges kept landing while main CI was
red, and the already-failing required check masked the NEW breakage each merge
introduced, so recovery cost grew with every merge that landed inside the red
window. The cure is a merge hold: gates may review a PR while main is red but
must not merge (except the unbreak-main fix itself) until main is green again.

This helper is the deterministic green/red signal that hold consults. The one
rule that matters — learned the hard way when the escalation guard stayed
silent on 2026-07-11 — is that an IN-PROGRESS run must never count as evidence
either way: only the most recent *completed* run decides. The fetch uses
``--status completed`` so in-progress runs are excluded at the API, and the
pure decision function filters defensively on top so the rule is unit-tested.

Exit code: 0 == green (latest completed CI run on main concluded ``success``),
1 == not green (red, or no completed run to judge from). Prints the run id and
conclusion it decided from.

Not-green is not one thing (issue #5424). A merge hold must fail closed on any
non-``success`` signal, but the *reason* it holds is operationally different:

- ``red``   — a completed run evaluated main's code and it FAILED. Main
  regressed; the cure is an unbreak-main fix, and no PR may merge over it.
- ``stale`` — the deciding run was aborted / skipped / never really ran the
  checks (``cancelled``, ``timed_out``, ``startup_failure``, ...). It still
  holds the gate, but it is NOT evidence that main regressed — as issue #5424
  put it, "a baseline gate blocker, not evidence that the held PRs themselves
  regress main". The cure is a fresh CI run, not a code fix.

``classify()`` surfaces that distinction (and ``--json`` makes it machine
readable) without changing the fail-closed exit code: only ``success`` is green.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_WORKFLOW = "CI"

# A completed run's ``conclusion`` falls into exactly one bucket:
#   green  -> the run evaluated main and it passed (the only mergeable signal).
#   red    -> the run evaluated main and it FAILED: a real regression.
#   stale  -> the run was aborted / skipped / never ran the checks. Not a
#             verdict on main's code; holds the gate but calls for a re-run.
GREEN_CONCLUSIONS = frozenset({"success"})
RED_CONCLUSIONS = frozenset({"failure"})


def classify(conclusion: str | None) -> str:
    """Bucket a completed run's ``conclusion`` into ``green`` / ``red`` / ``stale``.

    Only ``success`` is ``green`` and only ``failure`` is a ``red`` regression
    verdict. Everything else a completed run can report — ``cancelled``,
    ``timed_out``, ``startup_failure``, ``skipped``, ``neutral``,
    ``action_required``, or an unknown/``None`` value — is ``stale``: the run
    did not deliver a clean verdict on main's code. Falling through to ``stale``
    (rather than ``red``) for unknown strings keeps the human framing honest —
    an aborted run is not a code regression — while the caller still fails
    closed, since ``stale`` is not ``green``.
    """
    text = str(conclusion)
    if text in GREEN_CONCLUSIONS:
        return "green"
    if text in RED_CONCLUSIONS:
        return "red"
    return "stale"


def _gh(args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a GitHub CLI command (mirrors scripts/dev/compact_ci_snapshot.py)."""
    try:
        return subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=["gh", *args], returncode=124, stdout="", stderr="gh timed out"
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=127,
            stdout="",
            stderr=f"gh not executable: {exc}",
        )


def latest_completed_run(runs: list[Any]) -> dict[str, Any] | None:
    """Return the newest run whose ``status`` is ``completed``.

    ``runs`` is the list ``gh run list --json`` returns (newest first). We
    sort by ``createdAt`` descending defensively rather than trusting order,
    then take the first ``completed`` entry. In-progress / queued runs are
    skipped: an unfinished run is not evidence of green OR red.
    """
    completed = [
        run for run in runs if isinstance(run, dict) and str(run.get("status")) == "completed"
    ]
    completed.sort(key=lambda r: str(r.get("createdAt", "")), reverse=True)
    return completed[0] if completed else None


def decide(runs: list[Any]) -> tuple[bool, dict[str, Any] | None]:
    """(is_green, deciding_run). Green iff the latest completed run succeeded."""
    run = latest_completed_run(runs)
    if run is None:
        return False, None
    return str(run.get("conclusion")) == "success", run


def fetch_runs(
    repo: str = DEFAULT_REPO, workflow: str = DEFAULT_WORKFLOW, limit: int = 5
) -> list[dict[str, Any]]:
    """Fetch recent completed main CI runs. ``--status completed`` is load-bearing."""
    proc = _gh(
        [
            "run",
            "list",
            "--repo",
            repo,
            "--branch",
            "main",
            "--workflow",
            workflow,
            "--status",
            "completed",
            "--limit",
            str(limit),
            "--json",
            "databaseId,status,conclusion,headSha,createdAt",
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gh run list failed: {proc.stderr.strip() or proc.returncode}")
    data = json.loads(proc.stdout or "[]")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected JSON response type: {type(data).__name__}")
    return data


def main(argv: list[str] | None = None) -> int:
    """CLI entry: exit 0 if main CI is green, 1 otherwise."""
    ap = argparse.ArgumentParser(description="Is main CI green (latest completed run)?")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="suppress the human line")
    ap.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit a machine-readable {is_green, reason, conclusion, run_id, head_sha} object",
    )
    args = ap.parse_args(argv)

    try:
        runs = fetch_runs(args.repo, args.workflow, args.limit)
    except (RuntimeError, json.JSONDecodeError) as exc:
        # Fail closed: an unreadable signal is treated as NOT green so a merge
        # hold errs toward holding, never toward merging on unknown state.
        if args.as_json:
            print(json.dumps({"is_green": False, "reason": "unknown", "error": str(exc)}))
        elif not args.quiet:
            print(f"main CI status UNKNOWN ({exc}) -> treated as not-green", file=sys.stderr)
        return 1

    is_green, run = decide(runs)
    # reason distinguishes the two ways a hold happens: a real ``red`` regression
    # versus a ``stale`` (cancelled/aborted) signal that only needs a re-run.
    reason = "no-run" if run is None else classify(run.get("conclusion"))

    if args.as_json:
        print(
            json.dumps(
                {
                    "is_green": is_green,
                    "reason": reason,
                    "conclusion": None if run is None else run.get("conclusion"),
                    "run_id": None if run is None else run.get("databaseId"),
                    "head_sha": None if run is None else run.get("headSha"),
                }
            )
        )
    elif not args.quiet:
        if run is None:
            print("main CI: no completed run found -> not green [no-run: nothing to judge from]")
        else:
            hint = {
                "green": "",
                "red": " [red: main regressed; needs an unbreak-main fix]",
                "stale": " [stale: run aborted/skipped, not a main regression; re-run CI]",
            }[reason]
            print(
                f"main CI: {run.get('conclusion')} "
                f"(run {run.get('databaseId')}, {str(run.get('headSha'))[:9]}) "
                f"-> {'GREEN' if is_green else 'NOT GREEN'}{hint}"
            )
    return 0 if is_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
