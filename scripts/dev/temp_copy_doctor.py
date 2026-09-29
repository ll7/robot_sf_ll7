#!/usr/bin/env python3
"""Stale validation-copy doctor for /tmp (issue #9720).

Task-owned validation copies (``/tmp/robot-sf-pr<N>-*`` and siblings) are
created by ad-hoc agent commands and, when a run is killed before its cleanup
trap fires, silently consume the filesystem headroom the worktree capacity
gate depends on. This doctor is the safe, reviewable cleanup path:

- read-only by default; ``--apply`` is the only mode that deletes anything;
- reports count and bytes grouped by PR, with ownership (live process
  references) and age for every entry;
- classes every entry ``protected``, ``active``, ``registered_done``,
  ``unclassified``, or ``orphan_pr`` and never deletes ``protected``,
  ``active``, or ``unclassified`` data;
- deletion under ``--apply`` requires either a completed task-registry
  entry (``registered_done``) or an explicit ``--pr N --assume-merged``
  selection, always with ``--min-age-hours`` satisfied.

Report schema: ``temp_copy_doctor.v1``. Exit status: 0 when the report is
produced (including when reclaim candidates exist), 1 on operational errors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if TYPE_CHECKING:
    from collections.abc import Iterator

SCHEMA = "temp_copy_doctor.v1"
PROTECTED_NAMES = frozenset({"robot-sf-pr-ready-locks"})
ENTRY_PREFIX = "robot-sf-"
PR_PATTERN = re.compile(r"^robot-sf-pr(?P<pr>\d+)-")
ALT_PR_PATTERN = re.compile(r"^robot-sf-(?P<pr>\d+)-")
DEFAULT_MIN_AGE_HOURS = 24.0
MAX_REPORTED_REFERENCES = 8


def _iter_entries(tmp_root: Path) -> Iterator[Path]:
    try:
        with os.scandir(tmp_root) as scan:
            for entry in sorted(scan, key=lambda item: item.name):
                if entry.name.startswith(ENTRY_PREFIX):
                    yield Path(entry.path)
    except OSError as err:
        raise RuntimeError(f"cannot scan {tmp_root}: {err}") from err


def path_bytes(path: Path) -> int:
    """Return total bytes for *path* (files report their size)."""
    try:
        if path.is_symlink():
            return 0
        if path.is_file():
            return path.stat().st_size
        total = 0
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    continue
        return total
    except OSError:
        return 0


def parse_pr(name: str) -> int | None:
    """Return the PR number encoded in an entry name, when present."""
    match = PR_PATTERN.match(name) or ALT_PR_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group("pr"))


def _pid_references_target(pid: int, resolved: str) -> bool:
    """Return True when pid's cwd, exe, or an open fd names *resolved*."""
    base = f"/proc/{pid}"
    for probe in (f"{base}/cwd", f"{base}/exe"):
        try:
            if os.readlink(probe).startswith(resolved):
                return True
        except OSError:
            continue
    try:
        fds = os.listdir(f"{base}/fd")
    except OSError:
        return False
    for fd in fds:
        try:
            if os.readlink(f"{base}/fd/{fd}").startswith(resolved):
                return True
        except OSError:
            continue
    return False


def _proc_references(target: Path) -> list[int]:
    """Return PIDs whose cwd, exe, or an open fd resolves to *target*."""
    resolved = str(target)
    hits: list[int] = []
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return hits
    for pid in pids:
        if _pid_references_target(pid, resolved):
            hits.append(pid)
            if len(hits) >= MAX_REPORTED_REFERENCES:
                break
    return hits


def _registry_entries(registry_dir: Path) -> dict[str, dict[str, Any]]:
    """Load task-registry receipts keyed by absolute registered path."""
    by_path: dict[str, dict[str, Any]] = {}
    if not registry_dir.is_dir():
        return by_path
    for file in sorted(registry_dir.glob("*.json")):
        try:
            payload = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for item in payload.get("temporary_paths", []):
            raw = item.get("path")
            if isinstance(raw, str) and raw:
                by_path[raw] = {
                    "task": payload.get("task"),
                    "status": payload.get("status", "active"),
                    "receipt": file.name,
                }
    return by_path


def classify(
    path: Path,
    *,
    age_hours: float,
    owner_pids: list[int],
    registry: dict[str, dict[str, Any]],
    min_age_hours: float,
    pr: int | None,
) -> str:
    """Return the fail-closed classification for one entry."""
    if path.name in PROTECTED_NAMES or path.is_symlink():
        return "protected"
    if owner_pids:
        return "active"
    record = registry.get(str(path))
    if record is not None and record.get("status") == "completed":
        return "registered_done"
    if record is not None:
        return "active"
    if age_hours < min_age_hours:
        return "active"
    return "orphan_pr" if pr is not None else "unclassified"


def build_report(
    tmp_root: Path,
    *,
    registry_dir: Path,
    min_age_hours: float,
    now: float | None = None,
) -> dict[str, Any]:
    """Scan *tmp_root* and return the ``temp_copy_doctor.v1`` report."""
    current = now if now is not None else time.time()
    registry = _registry_entries(registry_dir)
    entries: list[dict[str, Any]] = []
    totals: dict[str, dict[str, int]] = {}
    pr_groups: dict[str, dict[str, int]] = {}
    for path in _iter_entries(tmp_root):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        age_hours = max(0.0, (current - mtime) / 3600.0)
        pr = parse_pr(path.name)
        owner_pids = _proc_references(path)
        classification = classify(
            path,
            age_hours=age_hours,
            owner_pids=owner_pids,
            registry=registry,
            min_age_hours=min_age_hours,
            pr=pr,
        )
        size = path_bytes(path)
        record = registry.get(str(path))
        row = {
            "path": str(path),
            "pr": pr,
            "bytes": size,
            "age_hours": round(age_hours, 2),
            "kind": "dir" if path.is_dir() else "file",
            "has_git": (path / ".git").exists() if path.is_dir() else False,
            "owner_pids": owner_pids,
            "classification": classification,
            "receipt": record.get("receipt") if record else None,
        }
        entries.append(row)
        totals.setdefault(classification, {"count": 0, "bytes": 0})
        totals[classification]["count"] += 1
        totals[classification]["bytes"] += size
        key = str(pr) if pr is not None else "no-pr"
        pr_groups.setdefault(key, {"count": 0, "bytes": 0, "reclaimable_bytes": 0})
        pr_groups[key]["count"] += 1
        pr_groups[key]["bytes"] += size
        if classification in {"registered_done", "orphan_pr"}:
            pr_groups[key]["reclaimable_bytes"] += size
    return {
        "schema": SCHEMA,
        "tmp_root": str(tmp_root),
        "min_age_hours": min_age_hours,
        "scanned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(current)),
        "totals_by_class": totals,
        "by_pr": pr_groups,
        "entries": entries,
    }


def deletable(entry: dict[str, Any], *, assume_merged_prs: set[int]) -> bool:
    """Return True when *entry* may be removed under ``--apply`` rules."""
    classification = entry.get("classification")
    if classification == "registered_done":
        return True
    pr = entry.get("pr")
    return classification == "orphan_pr" and isinstance(pr, int) and pr in assume_merged_prs


def apply_removals(
    report: dict[str, Any], *, assume_merged_prs: set[int], run=os.remove
) -> list[dict[str, Any]]:
    """Remove every deletable entry; return one receipt row per attempt."""
    outcomes: list[dict[str, Any]] = []
    for entry in report.get("entries", []):
        if not deletable(entry, assume_merged_prs=assume_merged_prs):
            continue
        path = Path(entry["path"])
        row = {"path": str(path), "bytes_freed": entry.get("bytes", 0), "ok": False}
        try:
            if not path.is_symlink() and str(path).startswith(str(report["tmp_root"])):
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    run(path)
                row["ok"] = True
        except OSError as err:
            row["error"] = str(err)
        outcomes.append(row)
    return outcomes


def _format_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TiB"


def render_human(report: dict[str, Any]) -> str:
    """Render a compact human summary of *report*."""
    lines = [
        f"temp_copy_doctor: {report['tmp_root']} "
        f"(min-age {report['min_age_hours']}h, scanned {report['scanned_at']})"
    ]
    for name, bucket in sorted(report["totals_by_class"].items()):
        lines.append(f"  {name:16s} {bucket['count']:5d} entries  {_format_bytes(bucket['bytes'])}")
    lines.append("  by PR:")
    for pr, bucket in sorted(
        report["by_pr"].items(), key=lambda item: item[1]["bytes"], reverse=True
    )[:15]:
        lines.append(
            f"    pr={pr:>8s}  {bucket['count']:4d} entries  "
            f"{_format_bytes(bucket['bytes'])}  "
            f"(reclaimable {_format_bytes(bucket['reclaimable_bytes'])})"
        )
    lines.append(
        "  dry-run: nothing removed; use --apply with --pr/--assume-merged "
        "or a completed task-registry receipt"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the doctor; return 0 when a report was produced, 1 on scan errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tmp-root",
        default=os.environ.get("TMPDIR", "/tmp"),
        help="directory to scan (default: $TMPDIR or /tmp)",
    )
    parser.add_argument(
        "--registry-dir",
        default="",
        help="task temp-registry receipts (default: <repo>/output/validation/tmp_registry)",
    )
    parser.add_argument("--min-age-hours", type=float, default=DEFAULT_MIN_AGE_HOURS)
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--apply", action="store_true", help="delete eligible entries")
    parser.add_argument("--pr", type=int, action="append", default=[])
    parser.add_argument(
        "--assume-merged",
        action="store_true",
        help="confirm the --pr selections are merged/closed and may be reclaimed",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    registry_dir = (
        Path(args.registry_dir)
        if args.registry_dir
        else repo_root / "output" / "validation" / "tmp_registry"
    )
    if args.assume_merged and not args.pr:
        parser.error("--assume-merged requires one or more --pr selections")
    try:
        report = build_report(
            Path(args.tmp_root).expanduser().resolve(),
            registry_dir=registry_dir,
            min_age_hours=args.min_age_hours,
        )
    except RuntimeError as err:
        print(f"temp_copy_doctor: {err}", file=sys.stderr)
        return 1

    if args.pr:
        selected = set(args.pr)
        report["entries"] = [entry for entry in report["entries"] if entry.get("pr") in selected]

    outcomes: list[dict[str, Any]] = []
    if args.apply:
        outcomes = apply_removals(
            report, assume_merged_prs=set(args.pr) if args.assume_merged else set()
        )
        report["removals"] = outcomes
        report["freed_bytes"] = sum(row["bytes_freed"] for row in outcomes if row.get("ok"))

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_human(report))
        for row in outcomes:
            state = "removed" if row.get("ok") else f"failed {row.get('error', '')}"
            print(f"  apply: {row['path']} -> {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
