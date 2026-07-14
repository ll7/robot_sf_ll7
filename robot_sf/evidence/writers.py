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

from robot_sf.evidence.distance_convention import (
    DISTANCE_CONVENTION_FIELD,
    DistanceConvention,
    require_distance_convention,
)


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


def review_marker(issue_ref: str, marker_date: str | None = None) -> str:
    """Return the standard review marker comment for a given issue reference.

    Args:
        issue_ref: Issue identifier like "robot_sf#4891" or "robot_sf#4848"
        marker_date: Optional ISO date (YYYY-MM-DD) pinned to the bundle's
            provenance timestamp.  When provided the marker includes the date
            so that re-runs from the same bundle produce byte-identical output.
            Must never be wall-clock time; derive from metadata or pass
            explicitly via ``--marker-date``.
    """
    if marker_date is not None:
        return f"<!-- AI-GENERATED ({issue_ref}, {marker_date}) - NEEDS-REVIEW -->"
    return f"<!-- AI-GENERATED ({issue_ref}) - NEEDS-REVIEW -->"


def extract_marker_date(metadata: dict[str, Any]) -> str | None:
    """Extract YYYY-MM-DD from a bundle's ``generated_at_utc`` provenance field.

    Deterministic per the maintainer decision on #4903: the marker date is
    pinned to the bundle's provenance timestamp and never falls back to
    wall-clock time. When the provenance field is absent or empty, returns
    ``None`` so the marker omits the date rather than fabricating one from the
    current time.

    Returns:
        The ``YYYY-MM-DD`` date string, or ``None`` when provenance is absent.
    """
    generated_at = metadata.get("generated_at_utc", "")
    return generated_at[:10] if generated_at else None


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


def write_text(
    path: Path,
    content: str,
    *,
    issue_ref: str | None = None,
    marker_date: str | None = None,
) -> None:
    """Write marked Markdown/text evidence.

    Callers may provide ``issue_ref`` to have the shared writer prepend the
    canonical HTML marker. Existing generated text may omit ``issue_ref`` only
    when it already starts with an AI-GENERATED / NEEDS-REVIEW marker. This
    keeps marker ownership in one module while preserving pinned marker dates
    and byte-stable reruns.
    """
    if issue_ref is not None:
        marker = review_marker(issue_ref, marker_date=marker_date)
        if not content.startswith(marker):
            content = f"{marker}\n{content}"
    else:
        first_line = content.splitlines()[0] if content else ""
        if not (
            content.startswith(("<!-- AI-GENERATED", "# AI-GENERATED"))
            and "NEEDS-REVIEW" in first_line
        ):
            raise ValueError(
                f"generated evidence text must start with an AI-GENERATED marker: {path}"
            )
    path.write_text(content, encoding="utf-8")


def write_distance_series_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    convention: DistanceConvention | str,
    series_name: str | None = None,
) -> None:
    """Write a distance-like series CSV with an explicit convention annotation.

    Issue #5141: distance-like series exports must declare which distance
    convention they carry so a center-to-center value is never misread as
    surface clearance. This writer prepends a ``# distance_convention: <x>``
    header line next to the review marker and validates the convention value.

    Args:
        path: Output CSV path.
        rows: CSV rows (must be non-empty).
        convention: A :class:`DistanceConvention` or its string value
            (``center_center`` / ``surface_clearance`` / ``center_segment``).
        series_name: Optional series identifier for error messages; defaults
            to the file name.
    """
    resolved = require_distance_convention(
        {DISTANCE_CONVENTION_FIELD: convention}, series_name or path.name
    )
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(review_marker_comment() + "\n")
        handle.write(f"# {DISTANCE_CONVENTION_FIELD}: {resolved.value}\n")
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
