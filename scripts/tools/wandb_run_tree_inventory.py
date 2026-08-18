#!/usr/bin/env python3
"""Symlink-safe inventory and retirement planning for W&B run-tree outputs.

Weights & Biases (W&B) run directories (``wandb/offline-run-*`` trees)
routinely contain symlinks created on the training host: the ``latest-run``
pointer, media links, and absolute ``/home/<user>`` or ``/tmp/<user>``
artifact references. On another host those absolute links are broken.
Retirement tooling must surface them as blockers instead of silently treating
them as missing regular files or deleting them as cache.

This tool is strictly read-only on scanned trees. It

* records regular files, directories, symlinks, and broken links as separate
  object classes without following any link;
* records every link target verbatim via ``os.readlink`` and never repairs,
  deletes, uploads, or dereferences links;
* validates retirement plans fail-closed against the live inventory.

Safe invocation::

    python scripts/tools/wandb_run_tree_inventory.py <run-tree-root> [--json]

The CLI prints a read-only receipt and exits nonzero when the scan records
any blocker. Operator stop conditions (stop and escalate; retire nothing):

* any unexpected link target: absolute ``/home/`` or ``/tmp/`` host paths,
  ``..`` components, or targets that escape the scanned root;
* path-set drift between the retirement plan and the live tree in either
  direction (extra files in the tree or extra paths in the plan);
* checksum mismatch when checksums are requested by the calling workflow;
* any attempt to follow, repair, or delete a link during planning.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCHEMA_VERSION = "robot-sf-wandb-run-tree-inventory.v1"

REGULAR_FILE = "regular_file"
DIRECTORY = "directory"
SYMLINK_CONTAINED = "symlink_contained"
SYMLINK_BROKEN = "symlink_broken"
SYMLINK_EXTERNAL = "symlink_external"
OTHER = "other"

OBJECT_CLASSES = (
    REGULAR_FILE,
    DIRECTORY,
    SYMLINK_CONTAINED,
    SYMLINK_BROKEN,
    SYMLINK_EXTERNAL,
    OTHER,
)
SYMLINK_CLASSES = frozenset({SYMLINK_CONTAINED, SYMLINK_BROKEN, SYMLINK_EXTERNAL})
HOST_PATH_PREFIXES = ("/home/", "/tmp/")


@dataclass(frozen=True)
class InventoryEntry:
    """One scanned object recorded with lstat semantics, never followed."""

    relative_path: str
    object_class: str
    size_bytes: int | None
    link_target: str | None
    blocker: str | None


@dataclass(frozen=True)
class InventoryReport:
    """Read-only receipt for a scanned run tree."""

    root: str
    entries: tuple[InventoryEntry, ...]
    summary: dict[str, int]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class PlanVerdict:
    """Fail-closed verdict for a retirement plan."""

    ok: bool
    reasons: tuple[str, ...]


def build_inventory(root: Path) -> InventoryReport:
    """Build a read-only physical-path inventory of ``root`` without following links.

    Args:
        root: Run-tree root directory to scan. Must exist and must not itself
            be a symlink.

    Returns:
        An ``InventoryReport`` whose entries are sorted by relative path.

    Raises:
        ValueError: If ``root`` is a symlink.
        FileNotFoundError: If ``root`` does not exist or is not a directory.
    """
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError(f"inventory root must not be a symlink: {root_path}")
    if not root_path.is_dir():
        raise FileNotFoundError(f"inventory root is not a directory: {root_path}")
    root_abs = os.path.abspath(root_path)
    entries: list[InventoryEntry] = []
    _scan_dir(root_abs, "", root_abs, entries)
    entries.sort(key=lambda entry: entry.relative_path)
    blockers = tuple(
        f"{entry.relative_path}: {entry.blocker}" for entry in entries if entry.blocker
    )
    return InventoryReport(
        root=root_abs,
        entries=tuple(entries),
        summary=_summarize(entries),
        blockers=blockers,
    )


def validate_retirement_plan(
    report: InventoryReport,
    planned_paths: Sequence[str],
    allowed_link_targets: frozenset[str] = frozenset(),
) -> PlanVerdict:
    """Validate a retirement plan against a live inventory without touching the tree.

    Fail-closed rules: reject symlink roots and symlink traversal, paths that
    escape the inventory root, any plan that includes a symlink whose verbatim
    target is not in ``allowed_link_targets``, and any drift between the plan
    and the live inventory path set in either direction.

    Args:
        report: Inventory receipt for the live tree.
        planned_paths: Paths planned for retirement, relative to the inventory
            root or absolute under it.
        allowed_link_targets: Verbatim link targets allowed in the plan.

    Returns:
        A ``PlanVerdict`` with ``ok`` False and one reason per failed rule.
    """
    if not planned_paths:
        return PlanVerdict(ok=False, reasons=("empty retirement plan",))
    reasons: list[str] = []
    entries_by_path = {entry.relative_path: entry for entry in report.entries}
    planned_norms: list[str] = []
    for planned in planned_paths:
        norm = _normalize_planned_path(str(planned), report.root, reasons)
        if norm is not None:
            planned_norms.append(norm)
    for norm in planned_norms:
        _reject_symlink_ancestors(norm, entries_by_path, reasons)
        if norm != "." and norm not in entries_by_path:
            if not any(path.startswith(norm + "/") for path in entries_by_path):
                reasons.append(f"planned path not found in live inventory: {norm}")
    _reject_plan_coverage_drift(report, planned_norms, allowed_link_targets, reasons)
    return PlanVerdict(ok=not reasons, reasons=tuple(reasons))


def render_receipt(report: InventoryReport) -> str:
    """Render the read-only receipt as human-readable text."""
    lines = [
        f"W&B run-tree inventory (read-only): {report.root}",
        f"schema_version: {SCHEMA_VERSION}",
        "",
        "Summary:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in sorted(report.summary.items()))
    if report.blockers:
        lines.append("")
        lines.append("Blockers (operator stop conditions; retire nothing):")
        lines.extend(f"  {blocker}" for blocker in report.blockers)
    lines.append("")
    lines.append("Entries:")
    lines.extend(f"  {_entry_line(entry)}" for entry in report.entries)
    return "\n".join(lines)


def report_to_json(report: InventoryReport) -> str:
    """Serialize the read-only receipt as deterministic JSON."""
    payload = {"schema_version": SCHEMA_VERSION, "report": dataclasses.asdict(report)}
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    """Run the read-only W&B run-tree inventory CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="W&B run-tree root directory to scan read-only.")
    parser.add_argument("--json", action="store_true", help="Print the receipt as JSON.")
    args = parser.parse_args(argv)
    report = build_inventory(args.root)
    if args.json:
        print(report_to_json(report))
    else:
        print(render_receipt(report))
    return 1 if report.blockers else 0


def _scan_dir(
    dir_abs: str,
    rel_prefix: str,
    root_abs: str,
    entries: list[InventoryEntry],
) -> None:
    """Scan one directory level with ``os.scandir`` and recurse into real directories."""
    with os.scandir(dir_abs) as handle:
        for dir_entry in sorted(handle, key=lambda item: item.name):
            rel_path = posixpath.join(rel_prefix, dir_entry.name)
            entries.append(_classify_entry(dir_entry, rel_path, root_abs))
            if dir_entry.is_dir(follow_symlinks=False):
                _scan_dir(dir_entry.path, rel_path, root_abs, entries)


def _classify_entry(
    dir_entry: os.DirEntry[str],
    rel_path: str,
    root_abs: str,
) -> InventoryEntry:
    """Classify one directory entry using lstat semantics only."""
    if dir_entry.is_symlink():
        return _classify_symlink(dir_entry, rel_path, root_abs)
    if dir_entry.is_dir(follow_symlinks=False):
        return InventoryEntry(rel_path, DIRECTORY, None, None, None)
    if dir_entry.is_file(follow_symlinks=False):
        size = dir_entry.stat(follow_symlinks=False).st_size
        return InventoryEntry(rel_path, REGULAR_FILE, size, None, None)
    blocker = "unsupported object class; inspect manually before any retirement"
    return InventoryEntry(rel_path, OTHER, None, None, blocker)


def _classify_symlink(
    dir_entry: os.DirEntry[str],
    rel_path: str,
    root_abs: str,
) -> InventoryEntry:
    """Classify one symlink without following it; the target is recorded verbatim."""
    target = os.readlink(dir_entry.path)
    size = dir_entry.stat(follow_symlinks=False).st_size
    link_abs = posixpath.join(root_abs, rel_path)
    abs_target = (
        posixpath.normpath(target)
        if posixpath.isabs(target)
        else posixpath.normpath(posixpath.join(posixpath.dirname(link_abs), target))
    )
    contained = _is_within(root_abs, abs_target)
    reasons: list[str] = []
    if posixpath.isabs(target):
        reasons.append(_absolute_target_reason(target))
    if ".." in PurePosixPath(target).parts:
        reasons.append(f"link target contains '..' component: {target}")
    if not contained:
        reasons.append(f"link target escapes inventory root: {target}")
    exists = os.path.lexists(abs_target)
    if not exists:
        reasons.append(f"broken link: target does not exist: {target}")
    if posixpath.isabs(target) or not contained:
        object_class = SYMLINK_BROKEN if not exists else SYMLINK_EXTERNAL
    else:
        object_class = SYMLINK_CONTAINED
    blocker = "; ".join(reasons) if reasons else None
    return InventoryEntry(rel_path, object_class, size, target, blocker)


def _absolute_target_reason(target: str) -> str:
    """Return the blocker reason for an absolute link target."""
    if target.startswith(HOST_PATH_PREFIXES):
        return f"unexpected absolute host path link target: {target}"
    return f"absolute link target: {target}"


def _is_within(root_abs: str, path_abs: str) -> bool:
    """Return whether a lexical absolute path is the root or below it."""
    return path_abs == root_abs or path_abs.startswith(root_abs + os.sep)


def _summarize(entries: Sequence[InventoryEntry]) -> dict[str, int]:
    """Count inventory entries per object class plus a total."""
    summary: dict[str, int] = dict.fromkeys(OBJECT_CLASSES, 0)
    for entry in entries:
        summary[entry.object_class] += 1
    summary["total"] = len(entries)
    return summary


def _normalize_planned_path(planned: str, root_abs: str, reasons: list[str]) -> str | None:
    """Normalize one planned path relative to the inventory root, or record a rejection."""
    text = planned.strip()
    if not text:
        reasons.append("empty planned path")
        return None
    if ".." in PurePosixPath(text).parts:
        reasons.append(f"planned path contains '..' component: {planned}")
        return None
    if posixpath.isabs(text):
        norm_abs = posixpath.normpath(text)
        if not _is_within(root_abs, norm_abs):
            reasons.append(f"planned path escapes inventory root: {planned}")
            return None
        text = posixpath.relpath(norm_abs, root_abs)
    norm = posixpath.normpath(text)
    if norm == ".." or norm.startswith("../"):
        reasons.append(f"planned path escapes inventory root: {planned}")
        return None
    return norm


def _reject_symlink_ancestors(
    norm: str,
    entries_by_path: Mapping[str, InventoryEntry],
    reasons: list[str],
) -> None:
    """Reject planned paths whose ancestry inside the root contains a symlink."""
    if norm == ".":
        return
    parts = PurePosixPath(norm).parts
    for depth in range(1, len(parts)):
        ancestor = posixpath.join(*parts[:depth])
        entry = entries_by_path.get(ancestor)
        if entry is not None and entry.object_class in SYMLINK_CLASSES:
            reasons.append(
                f"planned path traverses symlink ancestor: {norm} via "
                f"{ancestor} -> {entry.link_target}"
            )


def _reject_plan_coverage_drift(
    report: InventoryReport,
    planned_norms: Sequence[str],
    allowed_link_targets: frozenset[str],
    reasons: list[str],
) -> None:
    """Reject drift between the plan and the live inventory in either direction."""
    for entry in report.entries:
        if entry.object_class == DIRECTORY:
            continue
        if not any(_covers(norm, entry.relative_path) for norm in planned_norms):
            reasons.append(
                f"live inventory path not covered by retirement plan: {entry.relative_path}"
            )
            continue
        if entry.object_class in SYMLINK_CLASSES and entry.link_target not in allowed_link_targets:
            reasons.append(
                "retirement plan includes symlink with target not in allowlist: "
                f"{entry.relative_path} -> {entry.link_target}"
            )


def _covers(norm: str, rel_path: str) -> bool:
    """Return whether a normalized planned path covers an inventory path."""
    return norm in (".", rel_path) or rel_path.startswith(norm + "/")


def _entry_line(entry: InventoryEntry) -> str:
    """Render one inventory entry as a text line."""
    line = f"{entry.relative_path}: {entry.object_class}"
    if entry.link_target is not None:
        line += f" -> {entry.link_target}"
    if entry.size_bytes is not None:
        line += f" ({entry.size_bytes} bytes)"
    if entry.blocker:
        line += " [BLOCKER]"
    return line


if __name__ == "__main__":
    raise SystemExit(main())
