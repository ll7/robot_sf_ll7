#!/usr/bin/env python3
"""Pre-commit hook to keep release-assurance evidence hashes aligned with source files.

This hook checks if any files referenced in the release-assurance evidence file
have changed. If they have, it updates the SHA-256 hashes in the evidence file
and adds the updated file to the commit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _sha256_file(path: Path) -> str:
    """Return sha256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_staged_files() -> list[Path]:
    """Get list of files staged for commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.strip()]


def _add_file_to_commit(path: Path) -> None:
    """Add a file to the current commit."""
    subprocess.run(
        ["git", "add", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_evidence_payload(evidence_path: Path) -> dict | None:
    """Load and validate the evidence payload."""
    if not evidence_path.exists():
        return None

    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if not isinstance(payload, dict) or "evidence" not in payload:
        return None

    return payload


def _update_evidence_hashes(payload: dict, staged_files: list[Path]) -> bool:
    """Update evidence hashes for staged files. Returns True if any updates were made."""
    evidence_entries = payload.get("evidence", [])
    updated = False

    for entry in evidence_entries:
        if not isinstance(entry, dict):
            continue

        evidence_path_str = entry.get("path")
        if not evidence_path_str:
            continue

        evidence_file = Path(evidence_path_str)

        # Check if this evidence file is staged
        if evidence_file not in staged_files:
            continue

        # Check if the file exists
        if not evidence_file.exists():
            continue

        # Calculate current hash
        current_hash = _sha256_file(evidence_file)
        recorded_hash = entry.get("sha256")

        # Update if hash is different
        if current_hash != recorded_hash:
            entry["sha256"] = current_hash
            updated = True

    return updated


def main() -> int:
    """Run the release-assurance hash check."""
    evidence_path = Path("docs/context/evidence/issue_4683_release_assurance_case_example.json")

    payload = _load_evidence_payload(evidence_path)
    if payload is None:
        return 0

    staged_files = _get_staged_files()
    updated = _update_evidence_hashes(payload, staged_files)

    if updated:
        # Write updated evidence file
        evidence_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Add updated evidence file to commit
        _add_file_to_commit(evidence_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
