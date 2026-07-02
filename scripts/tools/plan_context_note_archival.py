#!/usr/bin/env python3
"""Plan a maintainer-reviewed archival sweep for stale/superseded context notes.

This is the *planning* half of the issue #3190 archival sweep. It consumes the
existing freshness checker (``check_context_note_freshness``) plus the context
catalog and proposes which notes should move to ``docs/context/archive/`` so a
maintainer can review the list before any bulk content move happens.

The planner is intentionally **plan-only**: it never moves, writes, or deletes a
context note. Provenance must be preserved (archive, don't delete), and the
issue requires the actual sweep to be a separate, maintainer-reviewed PR. This
tool produces that reviewable plan; applying it stays a human-gated step.

Candidate categories:

* ``superseded`` — catalog entries with ``status: superseded`` that already name
  a valid, existing replacement. These are high-confidence archive candidates:
  provenance is preserved by the named replacement, so the source note can be
  relocated under ``docs/context/archive/`` without losing the trail.
* ``stale`` — notes flagged by the checker's ``stale_current_dated`` rule (old,
  date-scoped, no inbound references). These are lower-confidence review
  candidates; a maintainer decides whether each is genuinely retired.

Conflicts (two candidates mapping to the same archive target, or a target that
already exists) are reported separately and force a non-zero exit so a plan that
cannot be applied as-is is never silently accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from scripts.tools import check_context_note_freshness as checker

if TYPE_CHECKING:
    from datetime import datetime

SCHEMA_VERSION = "context_note_archival_plan.v1"
APPROVAL_SCHEMA_VERSION = "context_archive_approved_moves.v1"
VALID_APPROVAL_CATEGORIES = frozenset({"superseded", "stale"})


@dataclass(frozen=True, slots=True)
class PlannedMove:
    """One proposed archival move (source note -> archive target)."""

    category: str
    confidence: str
    source: str
    target: str
    reason: str
    replacement: str | None = None
    last_touched_at: str | None = None
    age_days: int | None = None


@dataclass(frozen=True, slots=True)
class PlanConflict:
    """A reason a planned move cannot be applied as-is."""

    kind: str
    target: str
    message: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ApprovalFinding:
    """One fail-closed approval-manifest validation finding."""

    severity: str
    rule: str
    message: str
    path: str | None = None


def _archive_target(source: Path, *, archive_dir: Path) -> Path:
    """Return the archive destination path for a source note.

    The note keeps its file name under ``archive_dir`` so existing references can
    be repointed mechanically during the (separate) sweep PR.
    """

    return archive_dir / source.name


def _superseded_candidates(
    entries: list[dict[str, Any]],
    *,
    repo_root: Path,
    archive_dir: Path,
) -> list[PlannedMove]:
    """Return high-confidence archive candidates from valid superseded entries.

    Only entries whose named replacement exists in the repository qualify;
    superseded entries with missing/invalid replacements are integrity errors
    owned by the checker (Rule A) and are intentionally left for it to flag.
    """

    moves: list[PlannedMove] = []
    for entry in entries:
        if entry.get("status") != "superseded":
            continue
        source = checker._safe_repo_path(entry.get("path"))
        replacement = checker._safe_repo_path(entry.get("replacement"))
        if source is None or replacement is None:
            continue
        if not (repo_root / replacement).exists():
            continue
        if not (repo_root / source).is_file():
            continue
        # Already archived: nothing to plan.
        if archive_dir in source.parents:
            continue
        moves.append(
            PlannedMove(
                category="superseded",
                confidence="high",
                source=source.as_posix(),
                target=_archive_target(source, archive_dir=archive_dir).as_posix(),
                reason="superseded note with a named, existing replacement",
                replacement=replacement.as_posix(),
            )
        )
    return moves


def _stale_candidates(
    findings: list[checker.Finding],
    *,
    archive_dir: Path,
) -> list[PlannedMove]:
    """Return lower-confidence archive candidates from stale-note warnings."""

    moves: list[PlannedMove] = []
    for finding in findings:
        if finding.rule != "stale_current_dated":
            continue
        source = Path(finding.path)
        if archive_dir in source.parents:
            continue
        moves.append(
            PlannedMove(
                category="stale",
                confidence="review",
                source=finding.path,
                target=_archive_target(source, archive_dir=archive_dir).as_posix(),
                reason=finding.message,
                last_touched_at=finding.last_touched_at,
                age_days=finding.age_days,
            )
        )
    return moves


def _detect_conflicts(moves: list[PlannedMove], *, repo_root: Path) -> list[PlanConflict]:
    """Return conflicts that block applying the plan as-is.

    Two conflict kinds are surfaced:

    * ``duplicate_target`` — distinct source notes that would collide at the same
      archive path.
    * ``target_exists`` — an archive destination that already exists on disk, so
      an unconditional move would clobber a file.
    """

    by_target: dict[str, list[str]] = {}
    for move in moves:
        by_target.setdefault(move.target, []).append(move.source)

    conflicts: list[PlanConflict] = []
    for target, sources in sorted(by_target.items()):
        unique_sources = sorted(set(sources))
        if len(unique_sources) > 1:
            conflicts.append(
                PlanConflict(
                    kind="duplicate_target",
                    target=target,
                    message="multiple source notes map to the same archive target",
                    sources=tuple(unique_sources),
                )
            )
        elif (repo_root / Path(target)).exists():
            conflicts.append(
                PlanConflict(
                    kind="target_exists",
                    target=target,
                    message="archive target already exists; choose a distinct destination",
                    sources=tuple(unique_sources),
                )
            )
    return conflicts


def _approval_finding(
    rule: str,
    message: str,
    *,
    path: str | None = None,
    severity: str = "error",
) -> ApprovalFinding:
    """Return a structured approval-manifest validation finding."""

    return ApprovalFinding(severity=severity, rule=rule, message=message, path=path)


def _approval_move_to_planned(
    raw_move: object,
    *,
    index: int,
) -> tuple[PlannedMove | None, list[ApprovalFinding]]:
    """Parse one approved move into the same shape used by the generated plan."""

    if not isinstance(raw_move, dict):
        return None, [_approval_finding("move_shape", f"moves[{index}] must be a mapping")]

    findings: list[ApprovalFinding] = []
    source = checker._safe_repo_path(raw_move.get("source"))
    target = checker._safe_repo_path(raw_move.get("target"))
    category = raw_move.get("category")
    reason = raw_move.get("reason")
    replacement = checker._safe_repo_path(raw_move.get("replacement"))

    if source is None:
        findings.append(_approval_finding("source_path", "source must be a safe repository path"))
    if target is None:
        findings.append(_approval_finding("target_path", "target must be a safe repository path"))
    if category not in VALID_APPROVAL_CATEGORIES:
        findings.append(
            _approval_finding(
                "category",
                "category must be one of: stale, superseded",
                path=source.as_posix() if source else None,
            )
        )
    if not isinstance(reason, str) or not reason.strip():
        findings.append(
            _approval_finding(
                "reason",
                "reason must be a non-empty string",
                path=source.as_posix() if source else None,
            )
        )

    if findings or source is None or target is None or not isinstance(category, str):
        return None, findings

    return (
        PlannedMove(
            category=category,
            confidence="approved",
            source=source.as_posix(),
            target=target.as_posix(),
            reason=reason.strip(),
            replacement=replacement.as_posix() if replacement else None,
        ),
        findings,
    )


def validate_approval_manifest(  # noqa: C901
    manifest_path: Path,
    *,
    repo_root: Path,
    catalog_path: Path = checker.CATALOG_PATH,
    context_dir: Path = checker.CONTEXT_DIR,
    index_path: Path = checker.INDEX_PATH,
    archive_dir: Path = checker.ARCHIVE_DIR,
    max_age_days: int = 180,
    include_stale: bool = True,
) -> tuple[list[PlannedMove], list[ApprovalFinding], list[PlanConflict]]:
    """Validate a maintainer-approved archival move manifest against current candidates."""

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    findings: list[ApprovalFinding] = []
    if not isinstance(manifest, dict):
        return (
            [],
            [_approval_finding("manifest_shape", "approval manifest must be a YAML mapping")],
            [],
        )

    if manifest.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        findings.append(
            _approval_finding("schema_version", f"schema_version must be {APPROVAL_SCHEMA_VERSION}")
        )
    if manifest.get("issue") != 3190:
        findings.append(_approval_finding("issue", "issue must be 3190"))

    raw_moves = manifest.get("moves")
    if not isinstance(raw_moves, list) or not raw_moves:
        findings.append(_approval_finding("moves", "moves must be a non-empty list"))
        raw_moves = []

    approved_moves: list[PlannedMove] = []
    for index, raw_move in enumerate(raw_moves):
        move, move_findings = _approval_move_to_planned(raw_move, index=index)
        findings.extend(move_findings)
        if move is not None:
            approved_moves.append(move)

    plan_moves, _plan_conflicts = plan_archival(
        repo_root=repo_root,
        catalog_path=catalog_path,
        context_dir=context_dir,
        index_path=index_path,
        archive_dir=archive_dir,
        max_age_days=max_age_days,
        include_stale=include_stale,
    )
    planned_by_key = {(move.category, move.source, move.target): move for move in plan_moves}

    seen_sources: set[str] = set()
    seen_targets: set[str] = set()
    for move in approved_moves:
        target_path = Path(move.target)
        if archive_dir not in target_path.parents:
            findings.append(
                _approval_finding(
                    "target_archive_dir",
                    f"target must be under {archive_dir.as_posix()}",
                    path=move.source,
                )
            )
        if move.source in seen_sources:
            findings.append(
                _approval_finding(
                    "duplicate_source", "source appears more than once", path=move.source
                )
            )
        seen_sources.add(move.source)
        if move.target in seen_targets:
            findings.append(
                _approval_finding(
                    "duplicate_target",
                    "target appears more than once in approved moves",
                    path=move.source,
                )
            )
        seen_targets.add(move.target)
        if not (repo_root / Path(move.source)).is_file():
            findings.append(
                _approval_finding("source_exists", "source note does not exist", path=move.source)
            )

        planned_move = planned_by_key.get((move.category, move.source, move.target))
        if planned_move is None:
            findings.append(
                _approval_finding(
                    "not_in_current_plan",
                    "approved move is not in the current generated archival plan",
                    path=move.source,
                )
            )
            continue
        if planned_move.replacement and move.replacement != planned_move.replacement:
            findings.append(
                _approval_finding(
                    "replacement_mismatch",
                    f"replacement must match current plan: {planned_move.replacement}",
                    path=move.source,
                )
            )

    conflicts = _detect_conflicts(approved_moves, repo_root=repo_root)
    for conflict in conflicts:
        findings.append(
            _approval_finding(f"conflict_{conflict.kind}", conflict.message, path=conflict.target)
        )

    return approved_moves, findings, conflicts


def plan_archival(
    *,
    repo_root: Path,
    catalog_path: Path = checker.CATALOG_PATH,
    context_dir: Path = checker.CONTEXT_DIR,
    index_path: Path = checker.INDEX_PATH,
    archive_dir: Path = checker.ARCHIVE_DIR,
    max_age_days: int = 180,
    include_stale: bool = True,
    now: datetime | None = None,
) -> tuple[list[PlannedMove], list[PlanConflict]]:
    """Return the proposed archival moves and any blocking conflicts.

    No files are moved or written; the result is a maintainer-reviewable plan.
    """

    entries = checker._catalog_entries(repo_root / catalog_path)
    moves = _superseded_candidates(entries, repo_root=repo_root, archive_dir=archive_dir)

    if include_stale:
        findings = checker.check_freshness(
            repo_root=repo_root,
            catalog_path=catalog_path,
            context_dir=context_dir,
            index_path=index_path,
            max_age_days=max_age_days,
            now=now,
        )
        moves.extend(_stale_candidates(findings, archive_dir=archive_dir))

    moves.sort(key=lambda move: (move.category, move.source))
    conflicts = _detect_conflicts(moves, repo_root=repo_root)
    return moves, conflicts


def _summary(
    moves: list[PlannedMove], conflicts: list[PlanConflict], *, archive_dir: Path
) -> dict[str, Any]:
    """Return a compact JSON-ready plan summary."""

    counts: dict[str, int] = {}
    for move in moves:
        counts[move.category] = counts.get(move.category, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "archive_dir": archive_dir.as_posix(),
        "counts": counts,
        "conflict_count": len(conflicts),
        "exit_code": 1 if conflicts else 0,
        "moves": [asdict(move) for move in moves],
        "conflicts": [asdict(conflict) for conflict in conflicts],
        "note": "plan-only: no notes were moved or deleted; apply via a maintainer-reviewed sweep PR",
    }


def _approval_summary(
    *,
    manifest_path: Path,
    approved_moves: list[PlannedMove],
    findings: list[ApprovalFinding],
    conflicts: list[PlanConflict],
) -> dict[str, Any]:
    """Return compact JSON-ready approval validation summary."""

    return {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "manifest": manifest_path.as_posix(),
        "approved_move_count": len(approved_moves),
        "finding_count": len(findings),
        "conflict_count": len(conflicts),
        "exit_code": 1 if findings else 0,
        "moves": [asdict(move) for move in approved_moves],
        "findings": [asdict(finding) for finding in findings],
        "conflicts": [asdict(conflict) for conflict in conflicts],
        "note": "approval validation only: no notes were moved or deleted",
    }


def _print_human_report(
    moves: list[PlannedMove],
    conflicts: list[PlanConflict],
    *,
    archive_dir: Path,
    max_moves: int,
) -> None:
    """Print a grouped, bounded, human-readable archival plan."""

    if not moves and not conflicts:
        print("OK no archival candidates found")
        return

    print(
        f"Archival sweep plan (target: {archive_dir.as_posix()}/) -- plan-only, review before applying"
    )
    grouped: dict[str, list[PlannedMove]] = {}
    for move in moves:
        grouped.setdefault(move.category, []).append(move)
    for category, group in sorted(grouped.items()):
        print(f"{category}: {len(group)} candidate(s)")
        for move in group[:max_moves]:
            print(f"  [{move.confidence}] {move.source} -> {move.target}")
            print(f"      reason: {move.reason}")
        omitted = len(group) - max_moves
        if omitted > 0:
            print(f"  ... {omitted} more; use --json-output for full detail")

    if conflicts:
        print(f"conflicts: {len(conflicts)} (plan cannot be applied as-is)")
        for conflict in conflicts:
            print(f"  {conflict.kind} {conflict.target}: {conflict.message}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Plan a maintainer-reviewed archival sweep for stale/superseded "
            "docs/context notes. Plan-only: never moves or deletes notes."
        )
    )
    parser.add_argument("--max-age-days", type=int, default=180)
    parser.add_argument(
        "--no-stale",
        action="store_true",
        help="Plan only high-confidence superseded candidates; skip stale-note review candidates.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write a JSON plan to this path. Use '-' to print JSON to stdout.",
    )
    parser.add_argument(
        "--approval-manifest",
        type=Path,
        help=(
            "Validate an approved move manifest against the current plan. "
            "Schema: context_archive_approved_moves.v1."
        ),
    )
    parser.add_argument("--catalog", type=Path, default=checker.CATALOG_PATH)
    parser.add_argument("--context-dir", type=Path, default=checker.CONTEXT_DIR)
    parser.add_argument("--index", type=Path, default=checker.INDEX_PATH)
    parser.add_argument("--archive-dir", type=Path, default=checker.ARCHIVE_DIR)
    parser.add_argument(
        "--max-human-moves",
        type=int,
        default=50,
        help="Maximum moves to print per category in human-readable output.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the context-note archival-sweep planner."""

    args = _parse_args()
    repo_root = checker._repo_root()
    if args.approval_manifest:
        approved_moves, findings, conflicts = validate_approval_manifest(
            repo_root / args.approval_manifest,
            repo_root=repo_root,
            catalog_path=args.catalog,
            context_dir=args.context_dir,
            index_path=args.index,
            archive_dir=args.archive_dir,
            max_age_days=args.max_age_days,
            include_stale=not args.no_stale,
        )
        payload = _approval_summary(
            manifest_path=args.approval_manifest,
            approved_moves=approved_moves,
            findings=findings,
            conflicts=conflicts,
        )
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.json_output:
            if str(args.json_output) == "-":
                print(text, end="")
            else:
                output_path = repo_root / args.json_output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(text, encoding="utf-8")
        elif findings:
            print(f"approval manifest failed: {len(findings)} finding(s)")
            for finding in findings:
                location = f" {finding.path}" if finding.path else ""
                print(f"  {finding.rule}{location}: {finding.message}")
        else:
            print(f"OK approval manifest valid: {len(approved_moves)} move(s)")
        return int(payload["exit_code"])

    moves, conflicts = plan_archival(
        repo_root=repo_root,
        catalog_path=args.catalog,
        context_dir=args.context_dir,
        index_path=args.index,
        archive_dir=args.archive_dir,
        max_age_days=args.max_age_days,
        include_stale=not args.no_stale,
    )
    payload = _summary(moves, conflicts, archive_dir=args.archive_dir)
    if args.json_output:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if str(args.json_output) == "-":
            print(text, end="")
        else:
            output_path = repo_root / args.json_output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
    else:
        _print_human_report(
            moves,
            conflicts,
            archive_dir=args.archive_dir,
            max_moves=max(0, int(args.max_human_moves)),
        )
    return int(payload["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
