#!/usr/bin/env python3
"""Task-scoped registry for temporary validation copies (issue #9720).

Agents that create task-owned checkouts under ``/tmp`` (for example
``/tmp/robot-sf-pr<N>-*``) register each path here so an interrupted run can
be told apart from abandoned data, and completion can remove the copy after
validation logs are preserved.

Receipt schema: ``task_temp_registry.v1`` written atomically under
``output/validation/tmp_registry/<task>.json``. Fail-closed rules:

- only absolute paths inside the system temp root may be registered or
  removed; repository, worktree, and ``output/`` paths are refused;
- ``complete --remove`` deletes only paths previously registered under the
  same task, never symlinks, and records what was removed;
- every receipt keeps ``temporary_paths`` with registration and removal
  timestamps so ``scripts/dev/temp_copy_doctor.py`` can classify entries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SCHEMA = "task_temp_registry.v1"
DEFAULT_REGISTRY_DIR = Path("output/validation/tmp_registry")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _registry_dir(repo_root: Path, override: str) -> Path:
    return Path(override) if override else repo_root / DEFAULT_REGISTRY_DIR


def _receipt_path(registry_dir: Path, task: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in task)
    if not safe or safe in {".", ".."}:
        raise ValueError(f"unusable task id: {task!r}")
    return registry_dir / f"{safe}.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": SCHEMA,
            "task": path.stem,
            "status": "active",
            "temporary_paths": [],
            "created_at": _now(),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a {SCHEMA} receipt")
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _temp_root() -> Path:
    return Path(os.environ.get("TMPDIR", "/tmp")).resolve()


def _validate_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(f"path must be absolute: {raw}")
    resolved = path.resolve() if path.exists() else path.absolute()
    root = _temp_root()
    if not str(resolved).startswith(str(root) + os.sep) and resolved != root:
        raise ValueError(f"path escapes temp root {root}: {resolved}")
    repo = repo_root.resolve()
    if str(repo) in str(resolved):
        raise ValueError(f"path inside repository/worktree tree refused: {resolved}")
    return resolved


def cmd_register(args: argparse.Namespace, repo_root: Path) -> int:
    """Register one temporary path under the task receipt."""
    registry = _registry_dir(repo_root, args.registry_dir)
    receipt_path = _receipt_path(registry, args.task)
    payload = _load(receipt_path)
    validated = _validate_path(args.path, repo_root)
    existing = {item["path"] for item in payload["temporary_paths"]}
    if str(validated) not in existing:
        payload["temporary_paths"].append(
            {"path": str(validated), "registered_at": _now(), "removed_at": None}
        )
    payload["status"] = "active"
    payload["updated_at"] = _now()
    _write_atomic(receipt_path, payload)
    print(f"registered {validated} under task {args.task}")
    return 0


def _remove_registered(payload: dict[str, Any], *, remove: bool) -> tuple[list[str], list[str]]:
    """Remove still-present registered paths when *remove*; return outcomes."""
    removed: list[str] = []
    failures: list[str] = []
    if not remove:
        return removed, failures
    for item in payload["temporary_paths"]:
        path = Path(item["path"])
        if item.get("removed_at"):
            continue
        if path.is_symlink():
            failures.append(f"refused symlink {path}")
            continue
        if not str(path).startswith(str(_temp_root())):
            failures.append(f"refused path outside temp root {path}")
            continue
        if not path.exists():
            item["removed_at"] = _now()
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            item["removed_at"] = _now()
            removed.append(str(path))
        except OSError as err:
            failures.append(f"{path}: {err}")
    return removed, failures


def cmd_complete(args: argparse.Namespace, repo_root: Path) -> int:
    """Mark a task complete, optionally removing its registered paths."""
    registry = _registry_dir(repo_root, args.registry_dir)
    receipt_path = _receipt_path(registry, args.task)
    payload = _load(receipt_path)
    if not payload["temporary_paths"]:
        print(f"task {args.task} has no registered temporary paths", file=sys.stderr)
        return 1
    removed, failures = _remove_registered(payload, remove=args.remove)
    if not failures:
        payload["status"] = "completed"
    payload["updated_at"] = _now()
    _write_atomic(receipt_path, payload)
    for path in removed:
        print(f"removed {path}")
    for problem in failures:
        print(problem, file=sys.stderr)
    if failures:
        return 1
    print(f"task {args.task} marked {payload['status']}")
    return 0


def cmd_list(args: argparse.Namespace, repo_root: Path) -> int:
    """Print every task receipt as JSON."""
    registry = _registry_dir(repo_root, args.registry_dir)
    receipts = []
    if registry.is_dir():
        for file in sorted(registry.glob("*.json")):
            try:
                receipts.append(json.loads(file.read_text(encoding="utf-8")))
            except (OSError, ValueError) as err:
                print(f"unreadable receipt {file}: {err}", file=sys.stderr)
    print(json.dumps({"schema": SCHEMA, "receipts": receipts}, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", default="")
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register", help="register a temp path for a task")
    register.add_argument("--task", required=True)
    register.add_argument("--path", required=True)
    complete = sub.add_parser("complete", help="mark a task complete")
    complete.add_argument("--task", required=True)
    complete.add_argument(
        "--remove",
        action="store_true",
        help="delete registered paths that were not removed yet",
    )
    sub.add_parser("list", help="print all receipts as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch a registry subcommand; return a process exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        if args.command == "register":
            return cmd_register(args, repo_root)
        if args.command == "complete":
            return cmd_complete(args, repo_root)
        if args.command == "list":
            return cmd_list(args, repo_root)
    except (ValueError, OSError) as err:
        print(f"task_temp_registry: {err}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
