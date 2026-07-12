#!/usr/bin/env python3
"""Keep staged release-assurance evidence hashes aligned with staged source files."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EVIDENCE_PATH = Path("docs/context/evidence/issue_4683_release_assurance_case_example.json")


def _repo_root() -> Path:
    """Return the repository root for the active Git index."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _staged_bytes(repo_root: Path, relative_path: Path) -> bytes:
    """Read a regular tracked file from the Git index, failing closed otherwise."""
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Evidence path must be repository-relative: {relative_path}")
    working_path = repo_root / relative_path
    if not working_path.is_file():
        raise ValueError(f"Evidence path is not a regular working-tree file: {relative_path}")
    result = subprocess.run(
        ["git", "show", f":{relative_path.as_posix()}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Evidence path is not staged: {relative_path}")
    return result.stdout


def _load_staged_evidence(repo_root: Path) -> dict[str, Any]:
    """Load the staged evidence object, rejecting malformed payloads."""
    try:
        payload = json.loads(_staged_bytes(repo_root, EVIDENCE_PATH))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid staged release-assurance evidence: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Release-assurance evidence root must be an object.")
    if not isinstance(payload.get("evidence"), list):
        raise ValueError("Release-assurance evidence must contain an evidence list.")
    return payload


def _update_evidence_hashes(repo_root: Path, payload: dict[str, Any]) -> bool:
    """Update every evidence digest from its staged source content."""
    updated = False
    for entry in payload["evidence"]:
        if not isinstance(entry, dict):
            raise ValueError("Each release-assurance evidence entry must be an object.")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Each release-assurance evidence entry requires a path.")
        digest = hashlib.sha256(_staged_bytes(repo_root, Path(raw_path))).hexdigest()
        if entry.get("sha256") != digest:
            entry["sha256"] = digest
            updated = True
    return updated


def _stage_evidence(repo_root: Path) -> None:
    """Stage the regenerated evidence document."""
    subprocess.run(["git", "add", str(EVIDENCE_PATH)], cwd=repo_root, check=True)


def main() -> int:
    """Synchronize release-assurance hashes and require a reviewable recommit if changed."""
    try:
        repo_root = _repo_root()
        payload = _load_staged_evidence(repo_root)
        updated = _update_evidence_hashes(repo_root, payload)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        sys.stderr.write(f"release-assurance hash hook: {exc}\n")
        return 1

    if not updated:
        return 0

    evidence_path = repo_root / EVIDENCE_PATH
    evidence_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _stage_evidence(repo_root)
    sys.stderr.write("Updated staged release-assurance hashes; review and commit again.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
