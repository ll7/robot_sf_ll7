"""Public-safe projection for adversarial failure archives.

Certified adversarial archives produced on private runners embed
infrastructure-specific absolute paths (host home directories, worktree paths,
``output/`` campaign roots) inside otherwise public research evidence. Committing
those archives verbatim leaks internal layout and makes evidence
non-reproducible across machines.

This module provides a deterministic, reusable *public projection* that:

- replaces absolute private path prefixes with stable
  ``<scheme>://job-<id>/`` artifact URIs while preserving every
  candidate, metric, certification, family, seed, and archive-ID value exactly;
- records the source and projected archive SHA-256 digests plus a path-only
  transformation record (issue #5911); and
- fails closed when a projected archive still contains any absolute
  user/home/worktree path, so a leaked path can never reach the evidence tree.

The transform is purely a string-prefix substitution over path-bearing fields;
it never touches numeric, boolean, or structural values. It is the reusable
entry point promoted out of the one-off projection used to register job 13518
(PR #5905, issue #5305).

The canonical content digest is ``archive_sha256`` from
:mod:`robot_sf.adversarial.disjoint_evaluation`, so source/projected checksums
are directly comparable to the disjointness and readiness surfaces.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from robot_sf.adversarial.disjoint_evaluation import archive_sha256

PUBLIC_PROJECTION_SCHEMA_VERSION = "adversarial_archive_public_projection.v1"
DEFAULT_PUBLIC_SCHEME = "private-artifact"

# Absolute-path anchors that reveal private infrastructure. A projected public
# archive must contain none of these substrings. ``/home/`` and ``/root/`` cover
# Linux runner layouts, ``/Users/`` covers macOS, and ``worktrees/`` catches the
# linked-worktree path pattern used by this repository's automation.
_PRIVATE_PATH_ANCHORS: tuple[str, ...] = ("/home/", "/root/", "/Users/", "worktrees/")

# A Unix absolute path component: a leading ``/`` followed by path characters.
# Used by auto-detection to locate the absolute-path substrings inside
# free-form values (for example a ``replay_command`` shell line) so a single
# replacement root can be derived from the offending strings.
_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w./:-])(/[A-Za-z0-9_][A-Za-z0-9_./-]*)")


def _public_uri(scheme: str, job_id: str | int | None) -> str:
    """Return the public artifact URI prefix for ``scheme`` and optional job id."""
    if job_id is None or job_id == "":
        return f"{scheme}://"
    return f"{scheme}://job-{job_id}/"


@dataclass(frozen=True)
class PublicProjectionConfig:
    """Configuration for a public-evidence archive projection.

    Attributes:
        private_root: Absolute filesystem prefix to strip and replace with the
            public URI (for example ``/home/runner/output``). When provided,
            every occurrence is replaced verbatim; this is the deterministic,
            caller-supplied path. Mutually sufficient with auto-detection: if
            ``auto_detect_private_root`` is true, a prefix is derived from the
            offending strings when ``private_root`` is empty.
        scheme: Public artifact URI scheme (default ``private-artifact``).
        job_id: Optional job/campaign identifier embedded in the URI as
            ``<scheme>://job-<id>/``. When omitted the URI is ``<scheme>://``.
        auto_detect_private_root: When true and ``private_root`` is empty, derive
            the longest common absolute-path prefix from the offending strings
            and use it as the replacement target.
    """

    private_root: str = ""
    scheme: str = DEFAULT_PUBLIC_SCHEME
    job_id: str | int | None = None
    auto_detect_private_root: bool = False


@dataclass
class PublicProjectionResult:
    """Outcome of a public-evidence projection.

    Attributes:
        projected_archive: The public-safe archive payload. Every absolute
            private path has been replaced with a stable artifact URI; all
            research values are preserved.
        source_sha256: Canonical SHA-256 of the input archive payload.
        projected_sha256: Canonical SHA-256 of the projected archive payload.
        projection: Transformation record describing the path-only substitution.
        offending_paths: Distinct offending path strings found before projection.
    """

    projected_archive: dict[str, Any]
    source_sha256: str
    projected_sha256: str
    projection: dict[str, Any]
    offending_paths: list[str]


@dataclass
class _ProjectionState:
    """Mutable accumulator used while walking the archive payload."""

    replacements: int = 0
    offending_paths: list[str] = field(default_factory=list)
    rewritten_fields: set[str] = field(default_factory=set)


def find_offending_paths(
    archive: Any, *, anchors: Iterable[str] = _PRIVATE_PATH_ANCHORS
) -> list[str]:
    """Return distinct absolute-path strings in ``archive`` that leak infrastructure.

    The scan is recursive over the JSON-shaped payload and flags any string
    containing one of the private-path anchors (``/home/``, ``/root/``,
    ``/Users/``, ``worktrees/``). The result is sorted for deterministic output
    and contains the full offending string values so a caller can inspect which
    fields need projection.
    """
    anchor_tuple = tuple(anchors)
    found: list[str] = []
    seen: set[str] = set()

    def _visit(value: Any) -> None:
        if isinstance(value, str):
            if any(anchor in value for anchor in anchor_tuple) and value not in seen:
                seen.add(value)
                found.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                _visit(child)
        elif isinstance(value, list):
            for child in value:
                _visit(child)

    _visit(archive)
    return sorted(found)


def assert_no_private_paths(archive: Any, *, label: str = "projected archive") -> list[str]:
    """Fail closed when ``archive`` still contains a private absolute path.

    Returns the list of offending strings when empty (i.e. the archive is
    public-safe). Raises :class:`PrivatePathLeakError` with the offending values
    so the failure is actionable rather than a silent hygiene drift.
    """
    offending = find_offending_paths(archive)
    if offending:
        raise PrivatePathLeakError(label, offending)
    return offending


class PrivatePathLeakError(ValueError):
    """Raised when a projected public archive still contains a private path."""

    def __init__(self, label: str, offending_paths: list[str]) -> None:
        """Initialize with a human-readable label and the offending path strings."""
        preview = "; ".join(offending_paths[:5])
        extra = "" if len(offending_paths) <= 5 else f" (and {len(offending_paths) - 5} more)"
        super().__init__(
            f"{label} still contains {len(offending_paths)} private absolute path(s): "
            f"{preview}{extra}"
        )
        self.label = label
        self.offending_paths = offending_paths


def project_archive_to_public(
    archive: dict[str, Any],
    *,
    config: PublicProjectionConfig | None = None,
) -> PublicProjectionResult:
    """Project a private adversarial archive into a public-safe form.

    The transform replaces absolute private path prefixes with stable
    ``<scheme>://job-<id>/`` artifact URIs. It preserves candidate, metric,
    certification, family, seed, and archive-ID values exactly: only string
    path prefixes change. The result records source and projected SHA-256
    digests plus a path-only transformation record, and fails closed if a
    projected archive still contains any absolute user/home/worktree path.

    Args:
        archive: The source archive payload (must be JSON-serializable).
        config: Projection configuration. When ``None`` uses auto-detection with
            the default scheme and no job id.

    Returns:
        The :class:`PublicProjectionResult` carrying the projected archive,
        both SHA-256 digests, the transformation record, and the offending
        source paths.

    Raises:
        PrivatePathLeakError: If the projected archive still contains a private
            absolute path (fail-closed hygiene guard).
    """
    cfg = config or PublicProjectionConfig(auto_detect_private_root=True)
    source_sha256 = archive_sha256(archive)

    private_root = cfg.private_root
    if not private_root and cfg.auto_detect_private_root:
        offending = find_offending_paths(archive)
        private_root = _longest_common_absolute_prefix(offending)

    state = _ProjectionState()
    target_root = private_root.rstrip("/") if private_root else ""
    public_prefix = _public_uri(cfg.scheme, cfg.job_id)

    projected = _walk_and_replace(archive, target_root, public_prefix, state)

    projected_sha256 = archive_sha256(projected)
    # Fail closed: the projected archive must be free of private paths.
    assert_no_private_paths(projected, label="projected archive")

    projection = _build_projection_record(
        source_sha256=source_sha256,
        projected_sha256=projected_sha256,
        cfg=cfg,
        private_root=private_root,
        state=state,
    )
    return PublicProjectionResult(
        projected_archive=projected,
        source_sha256=source_sha256,
        projected_sha256=projected_sha256,
        projection=projection,
        offending_paths=state.offending_paths,
    )


def _walk_and_replace(
    value: Any, target_root: str, public_prefix: str, state: _ProjectionState
) -> Any:
    """Recursively copy ``value``, replacing private path prefixes in strings."""
    if isinstance(value, str):
        return _replace_in_string(value, target_root, public_prefix, state)
    if isinstance(value, dict):
        return {
            key: _walk_and_replace(child, target_root, public_prefix, state)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_walk_and_replace(child, target_root, public_prefix, state) for child in value]
    if isinstance(value, tuple):
        return tuple(_walk_and_replace(child, target_root, public_prefix, state) for child in value)
    return value


def _replace_in_string(
    text: str, target_root: str, public_prefix: str, state: _ProjectionState
) -> str:
    """Replace the private path prefix inside a single string value.

    Uses explicit-prefix replacement only: every occurrence of ``target_root``
    is replaced verbatim with the public URI prefix (without its trailing slash,
    so the surviving ``/raw/...`` tail joins cleanly). This handles embedded
    paths (for example a ``replay_command`` shell line) because ``str.replace``
    rewrites every occurrence.

    Paths that are NOT under ``target_root`` are deliberately left untouched.
    A stray absolute path under a different home/worktree is not something this
    projection can safely re-write, so it is preserved for the fail-closed guard
    (:func:`assert_no_private_paths`) to catch. Rewriting it would mask a leak.
    """
    if not isinstance(text, str) or not text:
        return text
    if not target_root or target_root not in text:
        return text

    original = text
    # Strip exactly one trailing path separator so the surviving ``/raw/...``
    # tail joins cleanly, but never consume the ``//`` of the ``<scheme>://``
    # authority separator. ``rstrip("/")`` would turn ``private-artifact://``
    # into ``private-artifact:``, which (a) malforms the URI and (b) when an
    # offending value equals the target root, destroys the whole string into a
    # bare ``private-artifact:`` and silently loses the path tail (issue #5911).
    if public_prefix.endswith("://"):
        replacement = public_prefix
    elif public_prefix.endswith("/"):
        replacement = public_prefix[:-1]
    else:
        replacement = public_prefix
    text = text.replace(target_root, replacement)
    if text != original:
        state.replacements += 1
        if original not in state.offending_paths:
            state.offending_paths.append(original)
        state.rewritten_fields.add("path")
    return text


def _longest_common_absolute_prefix(strings: list[str]) -> str:
    """Return the longest common leading absolute-path directory among strings.

    Used by auto-detection to derive a single replacement root from the set of
    offending strings. Considers only path-character prefixes up to the last
    ``/`` so the result is a directory rather than a partial segment. Returns
    an empty string when there are no absolute paths.
    """
    absolute = [match.group(1) for s in strings for match in _ABSOLUTE_PATH_RE.finditer(s)]
    absolute = [
        path for path in absolute if any(anchor in path for anchor in _PRIVATE_PATH_ANCHORS)
    ]
    if not absolute:
        return ""
    candidate = min(absolute, key=len)
    while candidate:
        if all(candidate in path for path in absolute):
            break
        if "/" not in candidate:
            return ""
        candidate = candidate.rsplit("/", 1)[0]
    if not candidate.endswith("/"):
        candidate += "/"
    return candidate.rstrip("/") or ""


def _build_projection_record(
    *,
    source_sha256: str,
    projected_sha256: str,
    cfg: PublicProjectionConfig,
    private_root: str,
    state: _ProjectionState,
) -> dict[str, Any]:
    """Build the path-only transformation record for a projection."""
    return {
        "schema_version": PUBLIC_PROJECTION_SCHEMA_VERSION,
        "source_archive_sha256": source_sha256,
        "projected_archive_sha256": projected_sha256,
        "scheme": cfg.scheme,
        "job_id": cfg.job_id,
        "private_pointer_scheme": _public_uri(cfg.scheme, cfg.job_id),
        # Record a hash of the private root rather than the root itself, so the
        # transformation is auditable without re-leaking the private layout.
        "private_root_sha256": _safe_sha256_text(private_root),
        "private_root_supplied": bool(cfg.private_root),
        "auto_detected_private_root": cfg.auto_detect_private_root and not cfg.private_root,
        "candidate_or_metric_values_changed": False,
        "changed_fields": "string path prefixes only",
        "rewritten_field_buckets": sorted(state.rewritten_fields) or ["path"],
        "replacement_count": state.replacements,
        "offending_path_count": len(state.offending_paths),
        "transformation": (
            "replace absolute private filesystem path prefixes with stable "
            f"{_public_uri(cfg.scheme, cfg.job_id)} artifact URIs; "
            "candidate, metric, certification, family, seed, and archive-ID "
            "values are preserved exactly"
        ),
    }


def _safe_sha256_text(text: str) -> str:
    """Return a SHA-256 digest of ``text``.

    Kept local so an empty private root hashes deterministically; the digest is
    recorded instead of the root so the private layout is never re-leaked.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_PUBLIC_SCHEME",
    "PUBLIC_PROJECTION_SCHEMA_VERSION",
    "PrivatePathLeakError",
    "PublicProjectionConfig",
    "PublicProjectionResult",
    "assert_no_private_paths",
    "find_offending_paths",
    "project_archive_to_public",
]
