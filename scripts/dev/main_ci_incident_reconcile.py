#!/usr/bin/env python3
"""Classify a main-red incident as active / stale / pending for auto-reconcile (#8001).

Motivation (issue #8001). The deterministic maintenance cycle opens a P0
"main CI is red" incident on a deciding failing run. When main later returns to
green, that incident is never reconciled and lingers open as a stale P0 (#7999,
#8000 both cited the same 2026-08-22 failure `32548602945` for six days while
main CI was green). This helper is the reusable, in-repo classifier that tells
an automation whether an incident can be auto-closed:

- ``stale``  : main is decisively green on a run NEWER than the incident's
  deciding failure -> the failure is superseded; the incident can be reconciled.
- ``active`` : the latest decisive main run is still a failure -> the incident
  is live and must stay open.
- ``pending``: no decisive verdict, or the incident's deciding failure cannot be
  resolved against the run window -> fail closed (do not auto-close).

It reuses the decisive green/red/stale classification from
``main_ci_is_green.py`` (issue #5385) so the two share one source of truth: an
in-progress or cancelled (superseded) run never counts as a verdict either way.

Exit code: 0 == stale (reconcilable), 1 == active or pending (do not close).
The ``--json`` flag emits the machine-readable schema
``main_ci_incident_reconcile.v1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from scripts.dev.main_ci_is_green import (
    classify,
    decide,
    fetch_runs,
    latest_decisive_run,
)

INCIDENT_SCHEMA_VERSION = "main_ci_incident_reconcile.v1"


def incident_reconcile_status(deciding_failure_run_id: int | None, runs: list[Any]) -> str:
    """Classify a red-main incident for auto-reconcile.

    Args:
        deciding_failure_run_id: The GitHub run id the incident was opened on, or
            ``None`` when the incident does not carry a resolvable deciding run.
        runs: Recent completed main CI runs (see :func:`fetch_runs`).

    Returns:
        ``stale``, ``active``, or ``pending`` (see module docstring).
    """
    if deciding_failure_run_id is None:
        return "pending"
    latest = latest_decisive_run(runs)
    if latest is None:
        return "pending"
    verdict = classify(latest.get("conclusion"))
    if verdict == "red":
        return "active"
    if verdict != "green":
        return "pending"
    latest_id = latest.get("databaseId")
    if latest_id is None or int(latest_id) <= deciding_failure_run_id:
        return "pending"
    return "stale"


def build_incident_signal(
    status: str,
    deciding_failure_run_id: int | None,
    runs: list[Any],
    *,
    repo: str = "ll7/robot_sf_ll7",
    workflow: str = "CI",
) -> dict[str, Any]:
    """Build the machine-readable incident-reconcile payload (the ``--json`` contract)."""
    is_green, current_run = decide(runs)
    return {
        "schema_version": INCIDENT_SCHEMA_VERSION,
        "status": status,
        "is_green": is_green,
        "can_auto_close": status == "stale",
        "deciding_failure_run_id": deciding_failure_run_id,
        "current_deciding_run": (
            {
                "databaseId": current_run.get("databaseId"),
                "conclusion": current_run.get("conclusion"),
                "headSha": current_run.get("headSha"),
                "createdAt": current_run.get("createdAt"),
            }
            if current_run is not None
            else None
        ),
        "repo": repo,
        "workflow": workflow,
    }


def main() -> int:
    """CLI entry: exit 0 when the incident is stale (reconcilable), 1 otherwise."""
    ap = argparse.ArgumentParser(
        description="Classify a main-red incident as active/stale/pending for auto-reconcile"
    )
    ap.add_argument(
        "--deciding-run",
        type=int,
        required=True,
        help="GitHub run id the incident was opened on (the deciding failure).",
    )
    ap.add_argument("--repo", default="ll7/robot_sf_ll7")
    ap.add_argument("--workflow", default="CI")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--quiet", action="store_true", help="suppress the human line")
    ap.add_argument(
        "--json", dest="as_json", action="store_true", help="emit machine-readable JSON"
    )
    args = ap.parse_args()

    try:
        runs = fetch_runs(args.repo, args.workflow, args.limit)
    except (RuntimeError, json.JSONDecodeError) as exc:
        if args.as_json:
            print(
                json.dumps(
                    {
                        "schema_version": INCIDENT_SCHEMA_VERSION,
                        "status": "pending",
                        "is_green": False,
                        "can_auto_close": False,
                        "deciding_failure_run_id": args.deciding_run,
                        "current_deciding_run": None,
                        "repo": args.repo,
                        "workflow": args.workflow,
                        "error": f"fetch failed: {exc}",
                    }
                )
            )
        if not args.quiet:
            print(f"incident reconcile UNKNOWN ({exc}) -> treated as pending", file=sys.stderr)
        return 1

    status = incident_reconcile_status(args.deciding_run, runs)
    if args.as_json:
        print(
            json.dumps(
                build_incident_signal(
                    status, args.deciding_run, runs, repo=args.repo, workflow=args.workflow
                )
            )
        )
    elif not args.quiet:
        print(
            f"incident {args.deciding_run}: {status} -> {'reconcilable' if status == 'stale' else 'do not close'}"
        )
    return 0 if status == "stale" else 1


if __name__ == "__main__":
    raise SystemExit(main())
