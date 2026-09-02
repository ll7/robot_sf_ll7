#!/usr/bin/env python3
"""Create and verify credential-free delegated-worker worktree receipts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "delegated_worktree_receipt.v1"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Machine-readable result of a receipt check."""

    schema: str
    ok: bool
    task_id: str | None
    expected_worktree: str | None
    current_worktree: str | None
    expected_ref: str | None
    current_ref: str | None
    expected_common_git_dir: str | None
    current_common_git_dir: str | None
    expected_base_commit: str | None
    current_commit: str | None
    failure: str | None


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail or result.returncode}")
    return result.stdout.strip()


def _git_optional(path: Path, *args: str) -> str | None:
    """Return optional Git output without hiding command failures."""
    result = subprocess.run(
        ["git", "-C", str(path), *args], capture_output=True, text=True, check=False
    )
    if result.returncode == 0:
        return result.stdout.strip() or None
    if args[:2] == ("symbolic-ref", "--quiet"):
        return None
    detail = (result.stderr or result.stdout).strip()
    raise ValueError(f"git {' '.join(args)} failed: {detail or result.returncode}")


def _absolute_directory(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError("worktree path must not be a symlink")
    resolved = candidate.resolve(strict=False)
    if not resolved.is_dir():
        raise ValueError("worktree path must be an existing directory")
    return resolved


def _identity(worktree: Path) -> dict[str, str]:
    top_level = Path(_git(worktree, "rev-parse", "--show-toplevel")).resolve()
    if top_level != worktree:
        raise ValueError(f"git top-level {top_level} does not match worktree {worktree}")
    common_git_dir = Path(
        _git(worktree, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).resolve()
    current_ref = _git_optional(worktree, "symbolic-ref", "--quiet", "--short", "HEAD")
    if current_ref:
        current_ref = f"refs/heads/{current_ref}"
    else:
        current_ref = "HEAD"
    return {
        "worktree": str(worktree),
        "common_git_dir": str(common_git_dir),
        "ref": current_ref,
        "commit": _git(worktree, "rev-parse", "HEAD"),
    }


def _resolve_base(worktree: Path, base_ref: str) -> str:
    return _git(worktree, "rev-parse", "--verify", f"{base_ref}^{{commit}}")


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    absolute = Path(os.path.abspath(path))
    cursor = Path(absolute.anchor)
    for component in absolute.relative_to(cursor).parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError("receipt path must not contain symlink components")
    if absolute.exists() or absolute.is_symlink():
        raise ValueError(f"refusing to overwrite existing receipt: {path}")
    parent = absolute.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise ValueError("receipt parent must not be a symlink")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, absolute)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def create_receipt(worktree: str | Path, *, task_id: str, base_ref: str) -> dict[str, str]:
    """Create one immutable receipt for an assigned worktree."""
    if not task_id or any(char in task_id for char in "\r\n"):
        raise ValueError("task_id must be a non-empty single-line value")
    assigned = _absolute_directory(worktree)
    identity = _identity(assigned)
    identity["base_commit"] = _resolve_base(assigned, base_ref)
    identity["task_id"] = task_id
    identity["schema"] = SCHEMA_VERSION
    return identity


def _load_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path)
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ValueError("receipt must be a regular file and must not be a symlink")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"receipt is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_VERSION:
        raise ValueError("receipt schema is missing or unsupported")
    required = ("task_id", "worktree", "common_git_dir", "ref", "base_commit")
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in required):
        raise ValueError("receipt has missing or malformed identity fields")
    return payload


def check_receipt(receipt: str | Path, worktree: str | Path = ".") -> CheckResult:
    """Check the current checkout and process directory against a receipt."""
    values: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "ok": False,
        "task_id": None,
        "expected_worktree": None,
        "current_worktree": None,
        "expected_ref": None,
        "current_ref": None,
        "expected_common_git_dir": None,
        "current_common_git_dir": None,
        "expected_base_commit": None,
        "current_commit": None,
        "failure": None,
    }
    try:
        expected = _load_receipt(receipt)
        values.update(
            task_id=expected["task_id"],
            expected_worktree=expected["worktree"],
            expected_ref=expected["ref"],
            expected_common_git_dir=expected["common_git_dir"],
            expected_base_commit=expected["base_commit"],
        )
        requested_worktree = _absolute_directory(worktree)
        current_directory = Path.cwd().resolve()
        values["current_worktree"] = str(current_directory)
        if requested_worktree != current_directory:
            raise ValueError(
                "current working directory mismatch: "
                f"expected {requested_worktree}, got {current_directory}"
            )
        current = _identity(current_directory)
        values.update(
            current_worktree=current["worktree"],
            current_ref=current["ref"],
            current_common_git_dir=current["common_git_dir"],
            current_commit=current["commit"],
        )
        mismatches = [
            ("worktree", expected["worktree"], current["worktree"]),
            ("ref", expected["ref"], current["ref"]),
            ("common_git_dir", expected["common_git_dir"], current["common_git_dir"]),
        ]
        for name, wanted, actual in mismatches:
            if wanted != actual:
                raise ValueError(f"{name} mismatch: expected {wanted}, got {actual}")
        _git(
            Path(current["worktree"]),
            "merge-base",
            "--is-ancestor",
            expected["base_commit"],
            "HEAD",
        )
        values["ok"] = True
    except (OSError, ValueError) as exc:
        values["failure"] = str(exc)
    return CheckResult(**values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--worktree", required=True)
    create.add_argument("--task-id", required=True)
    create.add_argument("--base-ref", default="origin/main")
    create.add_argument("--output", required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--receipt", required=True)
    check.add_argument("--worktree", default=".")
    check.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    """Run the receipt creation or read-only verification CLI."""
    args = _parser().parse_args()
    try:
        if args.command == "create":
            receipt = create_receipt(args.worktree, task_id=args.task_id, base_ref=args.base_ref)
            _write_atomic(Path(args.output), receipt)
            print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
            return 0
        result = check_receipt(args.receipt, args.worktree)
        print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
        return 0 if result.ok else 1
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {"schema": SCHEMA_VERSION, "ok": False, "failure": str(exc)}, separators=(",", ":")
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
