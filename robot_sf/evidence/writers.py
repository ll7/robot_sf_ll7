"""Shared evidence file writers with AI-GENERATED / NEEDS-REVIEW markers.

These writers ensure all evidence tree files include the required markers
for pr_contract_check rule 4 (evidence-tree hygiene).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Return the current git worktree root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _git_commit() -> str:
    """Return the current commit hash, or ``unknown`` outside git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 hex digest for ``path``.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def review_marker(issue_ref: str) -> str:
    """Return the standard review marker comment for a given issue reference.

    Args:
        issue_ref: Issue identifier like "robot_sf#4891" or "robot_sf#4848"
    """
    return f"<!-- AI-GENERATED ({issue_ref}) - NEEDS-REVIEW -->"


def review_marker_json() -> str:
    """Return the review marker value for JSON metadata."""
    return "AI-GENERATED NEEDS-REVIEW"


def review_marker_comment() -> str:
    """Return the review marker line for CSV/text files."""
    return "# AI-GENERATED NEEDS-REVIEW"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic JSON with review marker."""
    # Add review marker at top level
    marked_payload = {"review_marker": review_marker_json(), **payload}
    path.write_text(json.dumps(marked_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with review marker header."""
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(review_marker_comment() + "\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_sha256sums(output_dir: Path) -> None:
    """Write SHA256SUMS for all generated bundle files except itself.

    Computes hashes over the marked files (including markers).
    """
    files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = []
    for path in files:
        try:
            label = path.resolve().relative_to(_repo_root()).as_posix()
        except ValueError:
            label = path.name
        lines.append(f"{sha256_file(path)}  {label}")

    content = review_marker_comment() + "\n" + "\n".join(lines) + "\n"
    (output_dir / "SHA256SUMS").write_text(content, encoding="utf-8")
