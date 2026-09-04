#!/usr/bin/env python3
"""Check or refresh hashes in the tracked release-assurance example."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

EVIDENCE_PATH = Path("docs/context/evidence/issue_4683_release_assurance_case_example.json")


def update_evidence_hashes(
    payload: dict[str, Any],
    read_bytes: Callable[[Path], bytes],
) -> list[str]:
    """Update every ``sha256`` field from its corresponding source bytes."""
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("Release-assurance evidence must contain an evidence list.")

    updated_paths: list[str] = []
    for entry in evidence:
        if not isinstance(entry, dict):
            raise ValueError("Each release-assurance evidence entry must be an object.")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Each release-assurance evidence entry requires a path.")
        source_path = Path(raw_path)
        digest = hashlib.sha256(read_bytes(source_path)).hexdigest()
        if entry.get("sha256") != digest:
            entry["sha256"] = digest
            updated_paths.append(raw_path)
    return updated_paths


def _repo_root() -> Path:
    """Return the repository root for the active checkout."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _working_tree_bytes(repo_root: Path, relative_path: Path) -> bytes:
    """Read one regular, tracked repository file from the working tree."""
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Evidence path must be repository-relative: {relative_path}")
    working_path = repo_root / relative_path
    if not working_path.is_file() or working_path.is_symlink():
        raise ValueError(f"Evidence path is not a regular working-tree file: {relative_path}")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_path.as_posix()],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0:
        raise ValueError(f"Evidence path is not tracked: {relative_path}")
    return working_path.read_bytes()


def _load_payload(case_path: Path) -> dict[str, Any]:
    """Load and validate the JSON root without changing its content."""
    try:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read release-assurance evidence: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Release-assurance evidence root must be an object.")
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit check/write mode and optional case path."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check working-tree source hashes without writing the evidence file.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Refresh working-tree source hashes in the evidence file; do not stage it.",
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=EVIDENCE_PATH,
        help="Release-assurance example path (repository-relative by default).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Check or refresh the tracked release-assurance example."""
    args = _parse_args(argv)
    try:
        repo_root = _repo_root()
        case_path = args.case if args.case.is_absolute() else repo_root / args.case
        payload = _load_payload(case_path)
        updated_paths = update_evidence_hashes(
            payload,
            lambda relative_path: _working_tree_bytes(repo_root, relative_path),
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"refresh assurance case hashes: {exc}", file=sys.stderr)
        return 2

    if args.check:
        status = "ok" if not updated_paths else "mismatch"
        print(
            json.dumps(
                {
                    "case": str(case_path.relative_to(repo_root)),
                    "status": status,
                    "mismatched_paths": updated_paths,
                },
                sort_keys=True,
            )
        )
        return 0 if not updated_paths else 1

    if updated_paths:
        case_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "case": str(case_path.relative_to(repo_root)),
                "status": "updated" if updated_paths else "ok",
                "updated_paths": updated_paths,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
