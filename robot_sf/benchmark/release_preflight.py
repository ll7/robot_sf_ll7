"""Fail-closed release-readiness / claim-audit preflight checklist (issue #3081).

This module evaluates a declarative *checklist* of release prerequisites for a
research-package release (e.g. the July 2026 package tracked by issue #3081). It
maps each of that issue's acceptance criteria to a concrete, mechanically
checkable prerequisite and reports, per item, whether the prerequisite is
``complete`` or ``blocked`` with explicit gaps.

It deliberately does **not** publish, tag, upload, regenerate, or *declare*
release readiness. A passing preflight only means "no blocking gaps were found
among the declared prerequisites at evaluation time"; a maintainer still owns the
authoritative readiness decision and any publication step. The evaluator can only
ever *block*, never authorize, so it is safe to run autonomously.

Design notes:

- Every check **fails closed**: a missing file, a checksum mismatch, an
  unclassified sprint issue, or a promoted claim that depends on a
  fallback/degraded execution mode all resolve to ``blocked`` with an explicit
  reason rather than silently passing.
- The checklist is *orchestration*, not a re-implementation of existing
  validators. Where a canonical owner already validates a concern, prefer
  composing it. The release-manifest contract lives in
  :mod:`robot_sf.benchmark.release_protocol`; per-row claim wording/artifact
  validation lives in :mod:`robot_sf.benchmark.benchmark_row_claim`; the
  unavailable-execution vocabulary mirrored in :data:`EXCLUDED_CLAIM_MODES` is
  owned by ``benchmark_row_claim._NON_SUCCESS_MODES`` and
  ``failure_mechanism_classifier._UNAVAILABLE_STATUSES``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "release_preflight_checklist.v1"

# Acceptance-criteria buckets from issue #3081, used only to group checklist
# items in human-facing output. Item evaluation does not depend on these labels.
RELEASE_CRITERIA = (
    "reproduction",
    "tables_figures",
    "claim_audit",
    "sprint_issue_closure",
)

# Supported checklist item check kinds. Unknown kinds are rejected at load time
# so a typo cannot silently weaken the preflight.
SUPPORTED_CHECKS = (
    "artifact_present",
    "checksum_manifest",
    "claim_audit",
    "issue_classification_ledger",
)

# Execution modes that cannot back a promoted (paper-facing) claim. Mirrors the
# canonical exclusion vocabulary in ``benchmark_row_claim._NON_SUCCESS_MODES``
# and ``failure_mechanism_classifier._UNAVAILABLE_STATUSES``; kept as a local
# constant here to avoid importing private names across modules.
EXCLUDED_CLAIM_MODES = frozenset(
    {"fallback", "degraded", "not_available", "unavailable", "failed", "partial_failure"}
)

# Row statuses that cannot back a promoted claim. Mirrors the canonical
# ``benchmark_row_claim._NON_SUCCESS_STATUSES`` row-status vocabulary, which is
# broader than the planner-mode set above (e.g. ``excluded``, ``blocked``,
# ``revise``). A promoted claim card may carry a ``row_status`` instead of (or
# in addition to) a ``planner_mode``; checking only the mode vocabulary would
# let a non-success status slip through fail-open, so the two fields are audited
# against the union of both vocabularies.
EXCLUDED_CLAIM_ROW_STATUSES = frozenset(
    {
        "accepted_unavailable",
        "unexpected_failure",
        "fallback",
        "degraded",
        "blocked",
        "excluded",
        "revise",
        "completed_smoke_not_benchmark_evidence",
        "not_yet_populated",
    }
)

# Combined excluded vocabulary: any planner_mode or row_status value landing in
# this set blocks a promoted claim (fail-closed).
_EXCLUDED_CLAIM_VALUES = EXCLUDED_CLAIM_MODES | EXCLUDED_CLAIM_ROW_STATUSES

# Allowed terminal classifications for sprint issues, per issue #3081's body
# ("closed or classified continue, revise, stop, negative_result, invalid, or
# blocked_external").
ALLOWED_ISSUE_CLASSIFICATIONS = frozenset(
    {
        "closed",
        "continue",
        "revise",
        "stop",
        "negative_result",
        "invalid",
        "blocked_external",
    }
)


class ReleasePreflightError(ValueError):
    """Raised when a checklist definition is structurally invalid."""


@dataclass(frozen=True)
class ReleasePreflightItem:
    """A single declarative checklist item.

    Attributes:
        item_id: Unique identifier within the checklist.
        criterion: Acceptance-criterion bucket (see :data:`RELEASE_CRITERIA`).
        check: Check kind (see :data:`SUPPORTED_CHECKS`).
        description: Human-readable description of what the item proves.
        params: Check-specific parameters (e.g. ``path``, ``required_issues``).
    """

    item_id: str
    criterion: str
    check: str
    description: str
    params: dict[str, Any]


@dataclass(frozen=True)
class ReleasePreflightChecklist:
    """A loaded, validated release preflight checklist."""

    schema_version: str
    release_id: str
    description: str
    items: tuple[ReleasePreflightItem, ...]


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path) -> Any:
    """Load a JSON or YAML payload from disk.

    Returns:
        The parsed payload (mapping, list, or scalar).
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def _parse_checklist_item(index: int, raw: Any) -> ReleasePreflightItem:
    """Validate and build a single checklist item (fail-closed).

    Returns:
        The validated :class:`ReleasePreflightItem`.

    Raises:
        ReleasePreflightError: If the item is structurally invalid.
    """
    if not isinstance(raw, dict):
        raise ReleasePreflightError(f"items[{index}] must be a mapping")
    item_id = str(raw.get("item_id", "")).strip()
    if not item_id:
        raise ReleasePreflightError(f"items[{index}] is missing a non-empty item_id")

    check = str(raw.get("check", "")).strip()
    if check not in SUPPORTED_CHECKS:
        raise ReleasePreflightError(
            f"items[{index}] ({item_id}) has unknown check {check!r}; "
            f"expected one of {sorted(SUPPORTED_CHECKS)}"
        )

    criterion = str(raw.get("criterion", "")).strip()
    if criterion not in RELEASE_CRITERIA:
        raise ReleasePreflightError(
            f"items[{index}] ({item_id}) has unknown criterion {criterion!r}; "
            f"expected one of {sorted(RELEASE_CRITERIA)}"
        )

    params = {
        key: value
        for key, value in raw.items()
        if key not in {"item_id", "criterion", "check", "description"}
    }
    return ReleasePreflightItem(
        item_id=item_id,
        criterion=criterion,
        check=check,
        description=str(raw.get("description", "")).strip(),
        params=params,
    )


def load_release_preflight_checklist(path: str | Path) -> ReleasePreflightChecklist:
    """Load and structurally validate a checklist definition.

    Fails closed on an unknown schema version, duplicate item ids, unknown check
    kinds, or unknown criterion buckets so that a malformed checklist cannot be
    interpreted as "nothing to check".

    Returns:
        The validated :class:`ReleasePreflightChecklist`.

    Raises:
        ReleasePreflightError: If the definition is structurally invalid.
    """
    try:
        payload = _load_payload(Path(path))
    except (ValueError, yaml.YAMLError) as exc:
        # Surface a malformed checklist as a structural error (fail-closed)
        # rather than letting a raw parser exception escape the loader.
        raise ReleasePreflightError(f"Could not parse checklist {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleasePreflightError(f"Expected mapping payload in {path}")

    schema_version = str(payload.get("schema_version", ""))
    if schema_version != SCHEMA_VERSION:
        raise ReleasePreflightError(
            f"Unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}"
        )

    release_id = str(payload.get("release_id", "")).strip()
    if not release_id:
        raise ReleasePreflightError("Checklist is missing a non-empty release_id")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ReleasePreflightError("Checklist must declare a non-empty items list")

    items: list[ReleasePreflightItem] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_items):
        item = _parse_checklist_item(index, raw)
        if item.item_id in seen_ids:
            raise ReleasePreflightError(f"Duplicate item_id {item.item_id!r}")
        seen_ids.add(item.item_id)
        items.append(item)

    return ReleasePreflightChecklist(
        schema_version=schema_version,
        release_id=release_id,
        description=str(payload.get("description", "")).strip(),
        items=tuple(items),
    )


def _resolve_repo_path(repo_root: Path, raw_value: Any) -> tuple[Path | None, list[str]]:
    """Resolve a declared repo-relative path, fail-closed on unsafe references.

    Rejects absolute paths, worktree-local ``output/`` paths (not durable
    evidence per repository policy), and parent-escaping paths.

    Returns:
        A ``(resolved_path, gaps)`` tuple. ``resolved_path`` is ``None`` when the
        reference itself is invalid; otherwise it points inside ``repo_root``.
    """
    gaps: list[str] = []
    if not raw_value or not isinstance(raw_value, str):
        return None, ["missing or non-string path"]
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return None, [f"path must be repository-relative, got absolute {raw_value!r}"]
    posix = candidate.as_posix()
    if "output" in candidate.parts:
        gaps.append(f"path {posix!r} is worktree-local output/, not durable evidence")
    # Detect a symlink on the lexical join before `.resolve()` follows it, so a
    # symlinked artifact fails closed instead of being read as its target.
    lexical = repo_root / candidate
    if lexical.is_symlink():
        return None, gaps + [f"path {posix!r} is a symlink; fail-closed on symlinked evidence"]
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        return None, [f"path {posix!r} escapes the repository root"]
    return resolved, gaps


def _check_present(resolved: Path, posix: str) -> list[str]:
    """Return gaps if a path is not a present, regular, non-symlink file.

    Returns:
        A list of gap strings (empty when the file is present and regular).
    """
    gaps: list[str] = []
    if resolved.is_symlink():
        gaps.append(f"path {posix!r} is a symlink; fail-closed on symlinked evidence")
        return gaps
    if not resolved.exists():
        gaps.append(f"required artifact {posix!r} is missing")
        return gaps
    if not resolved.is_file():
        gaps.append(f"required artifact {posix!r} is not a regular file")
    return gaps


def _load_referenced_file(
    item: ReleasePreflightItem, repo_root: Path, kind: str
) -> tuple[Any, list[str]]:
    """Resolve, presence-check, and parse the file referenced by ``item.params['path']``.

    Returns:
        A ``(payload, gaps)`` tuple. ``payload`` is ``None`` when the path is
        unsafe, missing, or unparseable (with the reason in ``gaps``).
    """
    raw_value = item.params.get("path")
    resolved, gaps = _resolve_repo_path(repo_root, raw_value)
    if resolved is None:
        return None, gaps
    gaps += _check_present(resolved, Path(str(raw_value)).as_posix())
    if gaps:
        return None, gaps
    try:
        return _load_payload(resolved), gaps
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # ``yaml.YAMLError`` is not a ``ValueError`` subclass, so a malformed
        # referenced YAML file would otherwise escape as an unhandled crash
        # instead of failing closed to a gap.
        return None, [f"{kind} {raw_value!r} could not be parsed: {exc}"]


def _evaluate_artifact_present(item: ReleasePreflightItem, repo_root: Path) -> list[str]:
    """Validate that a single tracked artifact is present (fail-closed).

    Returns:
        A list of gap strings (empty when the artifact is present and regular).
    """
    raw_value = item.params.get("path")
    resolved, gaps = _resolve_repo_path(repo_root, raw_value)
    if resolved is None:
        return gaps
    return gaps + _check_present(resolved, Path(str(raw_value)).as_posix())


def _verify_checksum_entry(index: int, entry: Any, repo_root: Path) -> list[str]:
    """Validate one checksum-manifest entry (path present + sha256 match).

    Returns:
        A list of gap strings (empty when the entry is present and matches).
    """
    if not isinstance(entry, dict):
        return [f"entries[{index}] is not a mapping"]
    entry_path, entry_gaps = _resolve_repo_path(repo_root, entry.get("path"))
    if entry_path is None:
        return [f"entries[{index}]: {g}" for g in entry_gaps]
    posix = Path(str(entry.get("path"))).as_posix()
    present_gaps = _check_present(entry_path, posix)
    if present_gaps:
        return present_gaps
    expected = str(entry.get("sha256", "")).strip().lower()
    if not expected:
        return [f"entries[{index}] ({posix}) is missing sha256"]
    actual = _sha256_file(entry_path)
    if actual != expected:
        return [f"checksum mismatch for {posix}: expected {expected[:12]}…, got {actual[:12]}…"]
    return []


def _evaluate_checksum_manifest(item: ReleasePreflightItem, repo_root: Path) -> list[str]:
    """Validate a checksum manifest: every listed file exists and matches sha256.

    Returns:
        A list of gap strings (empty when every listed file matches its digest).
    """
    payload, gaps = _load_referenced_file(item, repo_root, "checksum manifest")
    if payload is None:
        return gaps
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        return [f"checksum manifest {item.params.get('path')!r} has no entries list"]
    for index, entry in enumerate(entries):
        gaps.extend(_verify_checksum_entry(index, entry, repo_root))
    return gaps


def _audit_promoted_claim(index: int, claim: Any) -> list[str]:
    """Validate one claim card; promoted claims must avoid excluded modes.

    Returns:
        A list of gap strings (empty when the claim is acceptable or unpromoted).
    """
    if not isinstance(claim, dict):
        return [f"claims[{index}] is not a mapping"]
    if not bool(claim.get("promoted", False)):
        return []
    claim_id = str(claim.get("claim_id", f"claims[{index}]"))
    # Audit both descriptor fields independently: a promoted claim may carry a
    # ``planner_mode``, a ``row_status``, or both, and a non-success value in
    # *either* must fail closed (checking only ``planner_mode`` would let a
    # non-success ``row_status`` slip through fail-open). Normalize hyphen to
    # underscore so e.g. ``partial-failure`` matches the underscore-spelled
    # canonical vocabulary.
    descriptors = {
        field: str(claim.get(field, "")).strip().lower().replace("-", "_")
        for field in ("planner_mode", "row_status")
    }
    present = {field: value for field, value in descriptors.items() if value}
    if not present:
        return [f"promoted claim {claim_id} is missing planner_mode/row_status"]
    gaps: list[str] = []
    for field, value in present.items():
        if value in _EXCLUDED_CLAIM_VALUES:
            gaps.append(
                f"promoted claim {claim_id} depends on excluded {field} {value!r}; "
                "fallback/degraded/unavailable/non-success rows cannot back a release claim"
            )
    return gaps


def _evaluate_claim_audit(item: ReleasePreflightItem, repo_root: Path) -> list[str]:
    """Validate that no promoted claim depends on a fallback/degraded mode.

    Returns:
        A list of gap strings (empty when every promoted claim is acceptable).
    """
    payload, gaps = _load_referenced_file(item, repo_root, "claim file")
    if payload is None:
        return gaps
    claims = payload.get("claims") if isinstance(payload, dict) else payload
    if not isinstance(claims, list) or not claims:
        return [f"claim file {item.params.get('path')!r} has no claims list"]
    for index, claim in enumerate(claims):
        gaps.extend(_audit_promoted_claim(index, claim))
    return gaps


def _evaluate_issue_ledger(item: ReleasePreflightItem, repo_root: Path) -> list[str]:
    """Validate every required sprint issue carries an allowed classification.

    Returns:
        A list of gap strings (empty when every required issue is classified).
    """
    payload, gaps = _load_referenced_file(item, repo_root, "issue ledger")
    if payload is None:
        return gaps
    ledger = payload.get("issues") if isinstance(payload, dict) else None
    if not isinstance(ledger, dict) or not ledger:
        return [f"issue ledger {item.params.get('path')!r} has no issues mapping"]
    normalized = {str(key): str(value).strip().lower() for key, value in ledger.items()}
    required = item.params.get("required_issues", [])
    if not isinstance(required, list):
        return [f"item {item.item_id} required_issues must be a list"]
    for issue in required:
        key = str(issue)
        classification = normalized.get(key)
        if classification is None:
            gaps.append(f"sprint issue #{key} is absent from the classification ledger")
        elif classification not in ALLOWED_ISSUE_CLASSIFICATIONS:
            gaps.append(
                f"sprint issue #{key} has unknown classification {classification!r}; "
                f"expected one of {sorted(ALLOWED_ISSUE_CLASSIFICATIONS)}"
            )
    return gaps


_EVALUATORS = {
    "artifact_present": _evaluate_artifact_present,
    "checksum_manifest": _evaluate_checksum_manifest,
    "claim_audit": _evaluate_claim_audit,
    "issue_classification_ledger": _evaluate_issue_ledger,
}


def evaluate_release_preflight(
    checklist: ReleasePreflightChecklist, repo_root: str | Path
) -> dict[str, Any]:
    """Evaluate every checklist item against ``repo_root`` (fail-closed).

    Each item resolves to ``complete`` only when no gaps are found; any gap makes
    it ``blocked``. The overall ``status`` is ``passed`` only when every item is
    ``complete``; otherwise it is ``blocked``. The returned payload never asserts
    that the release itself is ready for publication.

    Returns:
        A structured result with per-item statuses, gap lists, and a summary.
    """
    root = Path(repo_root)
    item_results: list[dict[str, Any]] = []
    blocked = 0
    for item in checklist.items:
        evaluator = _EVALUATORS[item.check]
        gaps = evaluator(item, root)
        status = "complete" if not gaps else "blocked"
        if gaps:
            blocked += 1
        item_results.append(
            {
                "item_id": item.item_id,
                "criterion": item.criterion,
                "check": item.check,
                "description": item.description,
                "status": status,
                "gaps": gaps,
            }
        )

    overall = "passed" if blocked == 0 else "blocked"
    return {
        "schema_version": checklist.schema_version,
        "release_id": checklist.release_id,
        "status": overall,
        "summary": {
            "total": len(item_results),
            "complete": len(item_results) - blocked,
            "blocked": blocked,
        },
        "items": item_results,
        # Explicit guard: a mechanical preflight pass is not a maintainer
        # release-readiness declaration and cannot authorize publication.
        "note": (
            "Preflight is fail-closed and advisory: 'passed' means no blocking gaps "
            "were found among declared prerequisites, not that the release is ready "
            "to publish. A maintainer owns the readiness decision."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render an evaluation result as a compact Markdown report.

    Returns:
        A Markdown string with the status header, summary, and per-item table.
    """
    summary = result["summary"]
    lines = [
        f"# Release preflight: {result['release_id']}",
        "",
        f"- Status: **{result['status']}**",
        f"- Items: {summary['total']} ({summary['complete']} complete, "
        f"{summary['blocked']} blocked)",
        "",
        "| Item | Criterion | Check | Status | Gaps |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in result["items"]:
        gaps = "; ".join(item["gaps"]) if item["gaps"] else "—"
        lines.append(
            f"| {item['item_id']} | {item['criterion']} | {item['check']} "
            f"| {item['status']} | {gaps} |"
        )
    lines += ["", f"> {result['note']}"]
    return "\n".join(lines)
