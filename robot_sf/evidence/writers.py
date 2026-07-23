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

import yaml

from robot_sf.evidence.distance_convention import (
    DISTANCE_CONVENTION_FIELD,
    DistanceConvention,
    require_distance_convention,
)

# docs/context/catalog.yaml registration (issue #6116).
#
# `scripts/dev/check_docs_evidence_integrity.py` (the blocking `docs-evidence-integrity`
# CI check) fails a PR whenever a changed `docs/context/evidence/**` file has no exact
# or ancestor-directory entry in `docs/context/catalog.yaml`. The vocabulary below
# mirrors that check's `_VALID_CATALOG_STATUSES` / `_VALID_CATALOG_FRESHNESS` (and the
# equivalent `_CATALOG_STATUSES` / `_CATALOG_DEFAULT_FRESHNESS` in
# `scripts/validation/check_docs_proof_consistency.py`). It is duplicated rather than
# imported because those are standalone `scripts/` modules the lightweight
# docs-evidence-integrity CI job installs without the `robot_sf` package (#4926/#4929
# regression note in that script).
_CATALOG_RELATIVE_PATH = Path("docs/context/catalog.yaml")
_EVIDENCE_RELATIVE_DIR = Path("docs/context/evidence")
_CATALOG_STATUSES = frozenset({"current", "historical", "superseded", "evidence", "proposal"})
_CATALOG_FRESHNESS = frozenset({"maintained", "dated", "policy", "evidence"})


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


def write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    catalog_area: str | None = None,
    catalog_status: str = "evidence",
    catalog_freshness: str = "evidence",
) -> None:
    """Write deterministic JSON with review marker.

    Pass ``catalog_area`` to also register ``path`` in
    ``docs/context/catalog.yaml`` via :func:`register_evidence` (issue #6116),
    so the ``docs-evidence-integrity`` CI check passes without a follow-up
    catalog edit. Omitted by default so existing callers are unaffected.
    """
    # Add review marker at top level
    marked_payload = {"review_marker": review_marker_json(), **payload}
    path.write_text(json.dumps(marked_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _maybe_register(
        path,
        catalog_area=catalog_area,
        catalog_status=catalog_status,
        catalog_freshness=catalog_freshness,
    )


REVIEW_SIDECAR_SCHEMA_VERSION = "evidence-review-marker.v1"


def write_review_sidecar(
    artifact_path: Path,
    *,
    repo_root: Path | None = None,
    preserved_exact_bytes: bool = True,
) -> Path:
    """Write a ``<artifact>.review.json`` sidecar carrying the review marker.

    Issue #5911: evidence artifacts whose byte-stable content must not be
    altered (for example a public-projected archive whose committed SHA-256 is
    recorded elsewhere) cannot carry an inline ``review_marker`` without
    changing their digest. This writer emits the canonical
    ``evidence-review-marker.v1`` sidecar next to the artifact so the required
    ``AI-GENERATED NEEDS-REVIEW`` marker is present via the shared convention
    without mutating the artifact bytes.

    Args:
        artifact_path: The evidence artifact the sidecar describes.
        repo_root: Optional repository root for a repo-relative ``artifact_path``
            field. Defaults to the current git worktree root.
        preserved_exact_bytes: Recorded as ``preserved_exact_bytes``; True means
            the artifact bytes were not modified after hashing.

    Returns:
        The path of the written sidecar (``<artifact_path>.review.json``).
    """
    root = repo_root if repo_root is not None else _repo_root()
    try:
        rel = artifact_path.resolve().relative_to(root.resolve())
        artifact_field = rel.as_posix()
    except ValueError:
        artifact_field = artifact_path.name
    sidecar_path = artifact_path.with_name(artifact_path.name + ".review.json")
    payload = {
        "schema_version": REVIEW_SIDECAR_SCHEMA_VERSION,
        "artifact_path": artifact_field,
        "artifact_sha256": sha256_file(artifact_path),
        "review_marker": review_marker_json(),
        "preserved_exact_bytes": preserved_exact_bytes,
    }
    sidecar_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sidecar_path


def _load_catalog_entries(catalog_file: Path) -> list[dict[str, Any]]:
    """Return the catalog's ``entries`` mappings, or ``[]`` if unparsable/empty."""
    if not catalog_file.is_file():
        return []
    payload = yaml.safe_load(catalog_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _catalog_registered_paths(entries: list[dict[str, Any]]) -> set[Path]:
    """Return repo-relative paths already registered in a catalog's entries."""
    paths: set[Path] = set()
    for entry in entries:
        value = entry.get("path")
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value.strip())
        if not candidate.is_absolute() and ".." not in candidate.parts:
            paths.add(candidate)
    return paths


def register_evidence(
    path: Path,
    *,
    area: str,
    status: str = "evidence",
    freshness: str = "evidence",
    repo_root: Path | None = None,
    catalog_path: Path | None = None,
) -> bool:
    """Register an emitted evidence artifact in ``docs/context/catalog.yaml``.

    Issue #6116: the ``docs-evidence-integrity`` CI check
    (``scripts/dev/check_docs_evidence_integrity.py``) fails a PR whenever a changed
    ``docs/context/evidence/**`` file has no exact-path or ancestor-directory entry in
    ``docs/context/catalog.yaml``. Evidence-writer call sites should invoke this after
    writing an artifact (or pass ``catalog_area`` to the ``write_*`` helpers below,
    which call it automatically) so future evidence-producing PRs register their own
    output and pass that check without a follow-up catalog edit.

    Idempotent and additive: if ``path`` (or an ancestor evidence-bundle directory) is
    already registered, this is a no-op that returns ``False``. Otherwise it appends
    one row to the end of the catalog's ``entries:`` list, preserving the file's
    existing bytes exactly (matching the append-only contract already used by
    ``scripts/tools/catalog_evidence.py``), so a diff shows only the new row.

    Args:
        path: The evidence artifact (file) or bundle directory to register. Must
            resolve inside ``docs/context/evidence/``.
        area: Catalog ``area`` tag (free-form; matches existing catalog usage).
        status: Catalog ``status``; must be a canonical value.
        freshness: Catalog ``freshness``; must be a canonical value.
        repo_root: Optional repository root. Defaults to the current git worktree
            root.
        catalog_path: Optional override for the catalog file's repo-relative path.
            Defaults to ``docs/context/catalog.yaml``.

    Returns:
        ``True`` if a new entry was appended, ``False`` if already covered.

    Raises:
        ValueError: ``path`` is outside the repository, outside
            ``docs/context/evidence/``, or ``status``/``freshness`` is not a
            canonical catalog value.
        FileNotFoundError: The target catalog file does not exist.
    """
    root = (repo_root if repo_root is not None else _repo_root()).resolve()
    try:
        rel = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"evidence path is not inside the repository root: {path}") from exc
    if _EVIDENCE_RELATIVE_DIR not in rel.parents:
        raise ValueError(f"refusing to register path outside {_EVIDENCE_RELATIVE_DIR}: {rel}")
    if status not in _CATALOG_STATUSES:
        raise ValueError(f"non-canonical catalog status {status!r} for {rel}")
    if freshness not in _CATALOG_FRESHNESS:
        raise ValueError(f"non-canonical catalog freshness {freshness!r} for {rel}")

    catalog_file = root / (catalog_path if catalog_path is not None else _CATALOG_RELATIVE_PATH)
    if not catalog_file.is_file():
        raise FileNotFoundError(f"catalog file not found, cannot register evidence: {catalog_file}")

    entries = _load_catalog_entries(catalog_file)
    registered = _catalog_registered_paths(entries)
    if rel in registered or any(parent in registered for parent in rel.parents):
        return False

    # Matches docs/context/catalog.yaml's actual convention: the sequence dash
    # for `entries:` sits at column 0 (aligned with the `entries:` key), with
    # sibling fields indented two spaces. Mixing indentation styles within one
    # YAML block sequence is a parse error, so this must match every existing
    # row exactly, not `scripts/tools/catalog_evidence.py`'s differently
    # indented (`  - path:`) block, which does not match the committed file.
    block = (
        f"- path: {rel.as_posix()}\n  status: {status}\n  freshness: {freshness}\n  area: {area}\n"
    )
    text = catalog_file.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        text += "\n"
    catalog_file.write_text(text + block, encoding="utf-8")
    return True


def _maybe_register(
    path: Path,
    *,
    catalog_area: str | None,
    catalog_status: str,
    catalog_freshness: str,
) -> None:
    """Call :func:`register_evidence` when ``catalog_area`` opts a caller in."""
    if catalog_area is None:
        return
    register_evidence(path, area=catalog_area, status=catalog_status, freshness=catalog_freshness)


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    catalog_area: str | None = None,
    catalog_status: str = "evidence",
    catalog_freshness: str = "evidence",
) -> None:
    """Write CSV rows with review marker header.

    Pass ``catalog_area`` to also register ``path`` in
    ``docs/context/catalog.yaml`` via :func:`register_evidence` (issue #6116),
    so the ``docs-evidence-integrity`` CI check passes without a follow-up
    catalog edit. Omitted by default so existing callers are unaffected.
    """
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(review_marker_comment() + "\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _maybe_register(
        path,
        catalog_area=catalog_area,
        catalog_status=catalog_status,
        catalog_freshness=catalog_freshness,
    )


def write_text(
    path: Path,
    content: str,
    *,
    issue_ref: str | None = None,
    marker_date: str | None = None,
    catalog_area: str | None = None,
    catalog_status: str = "evidence",
    catalog_freshness: str = "evidence",
) -> None:
    """Write marked Markdown/text evidence.

    Callers may provide ``issue_ref`` to have the shared writer prepend the
    canonical HTML marker. Existing generated text may omit ``issue_ref`` only
    when it already starts with an AI-GENERATED / NEEDS-REVIEW marker. This
    keeps marker ownership in one module while preserving pinned marker dates
    and byte-stable reruns.

    Pass ``catalog_area`` to also register ``path`` in
    ``docs/context/catalog.yaml`` via :func:`register_evidence` (issue #6116),
    so the ``docs-evidence-integrity`` CI check passes without a follow-up
    catalog edit. Omitted by default so existing callers are unaffected.
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
    _maybe_register(
        path,
        catalog_area=catalog_area,
        catalog_status=catalog_status,
        catalog_freshness=catalog_freshness,
    )


def write_distance_series_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    convention: DistanceConvention | str,
    series_name: str | None = None,
    catalog_area: str | None = None,
    catalog_status: str = "evidence",
    catalog_freshness: str = "evidence",
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
        catalog_area: When set, also registers ``path`` in
            ``docs/context/catalog.yaml`` via :func:`register_evidence`
            (issue #6116). Omitted by default so existing callers are
            unaffected.
        catalog_status: Catalog ``status`` used when ``catalog_area`` is set.
        catalog_freshness: Catalog ``freshness`` used when ``catalog_area`` is
            set.
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
    _maybe_register(
        path,
        catalog_area=catalog_area,
        catalog_status=catalog_status,
        catalog_freshness=catalog_freshness,
    )


def write_sha256sums(
    output_dir: Path,
    *,
    catalog_area: str | None = None,
    catalog_status: str = "evidence",
    catalog_freshness: str = "evidence",
) -> None:
    """Write SHA256SUMS for all generated bundle files except itself.

    Computes hashes over the marked files (including markers).

    ``write_sha256sums`` is typically the last step of finishing an evidence
    bundle directory, so pass ``catalog_area`` here to register the whole
    ``output_dir`` bundle in ``docs/context/catalog.yaml`` via
    :func:`register_evidence` (issue #6116): one directory-level entry covers
    every file already written under it (including this ``SHA256SUMS`` file),
    matching the ancestor-directory registration rule enforced by the
    ``docs-evidence-integrity`` CI check. Omitted by default so existing
    callers are unaffected.
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
    _maybe_register(
        output_dir,
        catalog_area=catalog_area,
        catalog_status=catalog_status,
        catalog_freshness=catalog_freshness,
    )
