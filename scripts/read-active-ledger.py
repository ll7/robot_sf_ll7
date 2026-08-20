#!/usr/bin/env python3
"""Read a compact snapshot of active common-Git-dir autopilot ledgers."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _default_ledger_dir() -> Path | None:
    """Resolve the repository common Git directory without assuming a worktree layout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(result.stdout.strip()) / "codex-agent-runs" / "active"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, help="Override the active ledger directory.")
    parser.add_argument("--limit", type=int, default=1, help="Maximum ledger files to include.")
    parser.add_argument("--tail-lines", type=int, default=24, help="Lines of each ledger tail.")
    parser.add_argument(
        "--no-tail", action="store_true", help="Omit ledger text from the snapshot."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def _snapshot(
    ledger_dir: Path | None, limit: int, tail_lines: int, include_tail: bool
) -> dict[str, object]:
    if ledger_dir is None:
        return {
            "schema_version": "active_ledger_snapshot.v1",
            "status": "unavailable",
            "reason": "common Git directory could not be resolved",
            "entries": [],
        }
    ledger_dir = ledger_dir.resolve()
    if not ledger_dir.is_dir():
        return {
            "schema_version": "active_ledger_snapshot.v1",
            "status": "empty",
            "ledger_dir": str(ledger_dir),
            "entries": [],
        }
    files = sorted(
        (
            path
            for path in ledger_dir.iterdir()
            if path.is_file() and not path.name.endswith(".lock")
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[: max(0, limit)]
    entries: list[dict[str, object]] = []
    for path in files:
        stat = path.stat()
        entry: dict[str, object] = {
            "path": str(path),
            "bytes": stat.st_size,
            "updated_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        }
        if include_tail:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            entry["tail"] = lines[-max(0, tail_lines) :]
        entries.append(entry)
    return {
        "schema_version": "active_ledger_snapshot.v1",
        "status": "ok" if entries else "empty",
        "ledger_dir": str(ledger_dir),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    """Print the newest active-ledger entries and fail only on unavailable resolution."""
    args = _parser().parse_args(argv)
    snapshot = _snapshot(
        args.ledger_dir or _default_ledger_dir(), args.limit, args.tail_lines, not args.no_tail
    )
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        print(f"active_ledger_snapshot.v1 status={snapshot['status']}")
        for entry in snapshot.get("entries", []):
            print(f"- {entry['path']} ({entry['bytes']} bytes, updated {entry['updated_at_utc']})")
            for line in entry.get("tail", []):
                print(f"  {line}")
    return 2 if snapshot["status"] == "unavailable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
