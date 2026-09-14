#!/usr/bin/env python3
"""Recurring TTL prune for merged-and-clean linked worktrees (issue #9257).

The linked-worktree fleet grows between manual reclaims, and one-shot passes
cannot keep up. This helper is the recurring mechanism: it selects only
worktrees that the preservation-aware hygiene assessment already classifies as
``removeable``, that carry no active claims or preservation evidence, and whose
last activity is older than the configured TTL. Everything else is reported
with a stable skip reason.

Dry-run by default. With ``--apply`` the helper delegates each selected path to
``scripts/dev/stale_worktree_reaper.py --apply --path <path>`` so the reaper's
own fail-closed checks, lease serialization, and refusal semantics stay
authoritative; this helper never removes a worktree itself and never touches
``output/`` artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

DEFAULT_TTL_DAYS = 7
DEFAULT_LIMIT = 20
SCHEMA = "worktree_ttl_prune.v1"
SECONDS_PER_DAY = 86_400.0


def _problem(row: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "path": str(row.get("path") or ""),
        "branch": row.get("branch"),
        "head_sha": row.get("head_sha"),
        "skip_reason": reason,
    }


def worktree_activity_epoch(
    path: str, *, run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> float | None:
    """Return the newest activity timestamp for one worktree, or None when unknown.

    Uses the worktree directory's mtime and its admin ``HEAD`` file, so a
    worktree that only had files touched without git activity still ages out
    while an actively used one does not.
    """
    try:
        epoch = Path(path).stat().st_mtime
    except OSError:
        return None
    try:
        git_dir = run(
            ["git", "-C", path, "rev-parse", "--absolute-git-dir"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return epoch
    if git_dir.returncode != 0:
        return epoch
    head_file = Path(git_dir.stdout.strip()) / "HEAD"
    try:
        epoch = max(epoch, head_file.stat().st_mtime)
    except OSError:
        pass
    return epoch


def select_ttl_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    ttl_days: float,
    now: float,
    activity_lookup: Callable[[str], float | None] = worktree_activity_epoch,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split hygiene rows into TTL-eligible candidates and skipped rows.

    Selection is fail-closed: a row is eligible only when the hygiene decision
    is ``removeable``, no active claims or preservation evidence are present,
    and the activity epoch is known and older than the TTL.
    """
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            skipped.append({"path": "", "skip_reason": "malformed_row"})
            continue
        decision = str(row.get("decision") or "")
        if decision != "removeable":
            skipped.append(_problem(row, f"hygiene_decision:{decision or 'unknown'}"))
            continue
        if row.get("active_claims"):
            skipped.append(_problem(row, "active_claims"))
            continue
        if row.get("ignored_artifacts") or row.get("tracked_durable_paths"):
            skipped.append(_problem(row, "preservation_evidence"))
            continue
        path = str(row.get("path") or "")
        epoch = activity_lookup(path) if path else None
        if epoch is None:
            skipped.append(_problem(row, "activity_unavailable"))
            continue
        age_days = (now - epoch) / SECONDS_PER_DAY
        if age_days < ttl_days:
            skipped.append({**_problem(row, "within_ttl"), "age_days": round(age_days, 2)})
            continue
        eligible.append(
            {
                "path": path,
                "branch": row.get("branch"),
                "head_sha": row.get("head_sha"),
                "issue_numbers": row.get("issue_numbers") or [],
                "age_days": round(age_days, 2),
                "hygiene_reasons": row.get("reasons") or [],
            }
        )
    return eligible, skipped


def _read_hygiene_rows(
    hygiene_json: Path | None, *, worktree_budget: int | None
) -> tuple[list[dict[str, Any]], str | None]:
    """Read fleet rows from a prior hygiene snapshot or a fresh bounded scan."""
    if hygiene_json is not None:
        try:
            payload = json.loads(Path(hygiene_json).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [], f"could not read hygiene snapshot {hygiene_json}: {exc}"
    else:
        command = [
            sys.executable,
            str(Path(__file__).with_name("worktree_hygiene_snapshot.py")),
            "--json",
            "--retirement-plan",
        ]
        if worktree_budget is not None:
            command.extend(["--worktree-budget", str(worktree_budget)])
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=1800, check=False
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return [], f"hygiene scan failed: {exc}"
        if result.returncode != 0:
            return [], f"hygiene scan failed with exit {result.returncode}"
        try:
            payload = json.loads(result.stdout)
        except ValueError as exc:
            return [], f"hygiene scan returned invalid JSON: {exc}"
    rows = payload.get("worktrees")
    if not isinstance(rows, list):
        return [], "hygiene snapshot has no worktrees list"
    return [row for row in rows if isinstance(row, dict)], None


def _run_reaper(path: str, *, repo_root: Path) -> dict[str, Any]:
    """Delegate one removal to the fail-closed reaper and report its verdict."""
    command = [
        sys.executable,
        str(repo_root / "scripts" / "dev" / "stale_worktree_reaper.py"),
        "--apply",
        "--path",
        path,
        "--json",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"path": path, "status": "error", "error": str(exc)}
    return {
        "path": path,
        "status": "applied" if result.returncode == 0 else "refused",
        "returncode": result.returncode,
        "detail": (result.stdout or result.stderr).strip()[:400],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the TTL prune CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ttl-days",
        type=float,
        default=DEFAULT_TTL_DAYS,
        help=f"Minimum idle age in days (default: {DEFAULT_TTL_DAYS}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delegate each selected path to the reaper; default is dry-run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum candidates to report/apply (default: {DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--hygiene-json",
        type=Path,
        default=None,
        help="Reuse a prior worktree_hygiene_snapshot JSON instead of rescanning.",
    )
    parser.add_argument(
        "--worktree-budget",
        type=int,
        default=None,
        help="Pass through the hygiene scan's retirement-plan worktree budget.",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON result.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the TTL prune pass and return the process exit code."""
    args = build_parser().parse_args(argv)
    if args.ttl_days <= 0:
        print("--ttl-days must be positive", file=sys.stderr)
        return 2
    if args.limit < 1:
        print("--limit must be >= 1", file=sys.stderr)
        return 2
    rows, error = _read_hygiene_rows(args.hygiene_json, worktree_budget=args.worktree_budget)
    if error is not None:
        print(json.dumps({"schema": SCHEMA, "status": "error", "error": error}, sort_keys=True))
        return 1
    eligible, skipped = select_ttl_candidates(rows, ttl_days=args.ttl_days, now=time.time())
    eligible = eligible[: args.limit]
    results: list[dict[str, Any]] = []
    if args.apply:
        repo_root = Path(__file__).resolve().parents[2]
        for candidate in eligible:
            results.append(_run_reaper(candidate["path"], repo_root=repo_root))
    payload = {
        "schema": SCHEMA,
        "status": "ok",
        "mode": "apply" if args.apply else "dry_run",
        "ttl_days": args.ttl_days,
        "scanned_worktrees": len(rows),
        "eligible_count": len(eligible),
        "eligible": eligible,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "reaper_results": results,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
