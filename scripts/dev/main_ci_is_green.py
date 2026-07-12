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
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

DEFAULT_REPO = "ll7/robot_sf_ll7"
DEFAULT_WORKFLOW = "CI"


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


def main() -> int:
    """CLI entry: exit 0 if main CI is green, 1 otherwise."""
    ap = argparse.ArgumentParser(description="Is main CI green (latest completed run)?")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="suppress the human line")
    args = ap.parse_args()

    try:
        runs = fetch_runs(args.repo, args.workflow, args.limit)
    except (RuntimeError, json.JSONDecodeError) as exc:
        # Fail closed: an unreadable signal is treated as NOT green so a merge
        # hold errs toward holding, never toward merging on unknown state.
        if not args.quiet:
            print(f"main CI status UNKNOWN ({exc}) -> treated as not-green", file=sys.stderr)
        return 1

    is_green, run = decide(runs)
    if not args.quiet:
        if run is None:
            print("main CI: no completed run found -> not green")
        else:
            print(
                f"main CI: {run.get('conclusion')} "
                f"(run {run.get('databaseId')}, {str(run.get('headSha'))[:9]}) "
                f"-> {'GREEN' if is_green else 'NOT GREEN'}"
            )
    return 0 if is_green else 1


if __name__ == "__main__":
    raise SystemExit(main())
