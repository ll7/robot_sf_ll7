#!/usr/bin/env python3
"""Check context-note freshness, decay, and catalog integrity with dry-run archival sweep."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

CONTEXT_DIR = Path("docs/context")
CATALOG_PATH = CONTEXT_DIR / "catalog.yaml"
INDEX_PATH = CONTEXT_DIR / "INDEX.md"
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)|<((?:\./)?[^ >]+)>")


@dataclass(frozen=True, slots=True)
class Finding:
    """One context-note freshness or catalog integrity finding."""

    rule: str
    severity: str
    path: str
    message: str
    status: str | None = None
    freshness: str | None = None
    replacement: str | None = None
    last_touched_at: str | None = None
    age_days: int | None = None


@dataclass(frozen=True, slots=True)
class ProposedMove:
    """Proposed archival sweep move (dry run)."""

    source: str
    target: str
    category: str
    reason: str
    replacement: str | None = None
    confidence: str = "review"


def _run(cmd: list[str], *, cwd: Path) -> str:
    """Run a subprocess and return stdout."""
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _repo_root() -> Path:
    """Return the repository root directory."""
    return Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd())).resolve()


def _load_yaml(path: Path) -> object:
    """Load a YAML file."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_repo_path(
    raw_path: object,
    *,
    repo_root: Path | None = None,
    base_dir: Path | None = None,
) -> Path | None:
    """Return a normalized, safe repository-relative path if possible."""
    if isinstance(raw_path, Path):
        path = raw_path
    elif isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path.strip())
    else:
        return None

    if path.is_absolute():
        if repo_root is None:
            return None
        try:
            path = path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            return None
    elif base_dir is not None and not path.parts[:2] == ("docs", "context"):
        path = base_dir / path

    normalized = Path(os.path.normpath(path.as_posix()))
    if normalized == Path(".") or ".." in normalized.parts:
        return None
    return normalized


def _safe_repo_path(raw_path: object) -> Path | None:
    """Return a safe repository-relative Path if possible."""
    return _normalize_repo_path(raw_path)


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether path is parent or contained under parent."""
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_markdown_target(target: str, *, source_dir: Path, context_dir: Path) -> Path | None:
    """Resolve a Markdown target to a normalized repository-relative path."""
    direct_path = _normalize_repo_path(target)
    if direct_path is None:
        return None
    if _path_is_relative_to(direct_path, context_dir) or direct_path.parts[:2] == (
        "docs",
        "context",
    ):
        return direct_path
    return _normalize_repo_path(target, base_dir=source_dir)


def _catalog_entries(catalog_path: Path) -> list[dict[str, Any]]:
    """Return catalog entry mappings."""
    payload = _load_yaml(catalog_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{catalog_path.as_posix()} must be a YAML mapping")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{catalog_path.as_posix()} entries must be a list")
    return [entry for entry in entries if isinstance(entry, dict)]


def _markdown_targets(text: str) -> set[str]:
    """Return normalized markdown targets."""
    targets: set[str] = set()
    for first, second in MARKDOWN_LINK_RE.findall(text):
        target = (first or second).strip()
        if not target:
            continue
        if "://" in target or target.startswith("mailto:"):
            continue
        target = target.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]
        while target.startswith("./"):
            target = target[2:]
        if target:
            targets.add(target)
    return targets


def _index_references(index_path: Path, *, context_dir: Path) -> set[Path]:
    """Return note paths referenced by INDEX.md.

    Missing configured index files fail closed in ``check_freshness_decay`` so orphan
    findings are not computed from an empty, misleading reference set.
    """
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    targets = _markdown_targets(index_path.read_text(encoding="utf-8"))
    references: set[Path] = set()
    for target in targets:
        path = _resolve_markdown_target(target, source_dir=context_dir, context_dir=context_dir)
        if path is None:
            continue
        if path.suffix != ".md":
            continue
        references.add(path)
    return references


def _tracked_context_notes(repo_root: Path, context_dir: Path) -> set[Path]:
    """Return tracked context notes under context_dir (excluding INDEX/README/evidence/archive)."""
    output = _run(["git", "ls-files", "--", context_dir.as_posix()], cwd=repo_root)
    notes: set[Path] = set()
    for line in output.splitlines():
        path = Path(line.strip())
        if path.suffix != ".md":
            continue
        if path.parent != context_dir:
            continue
        if path.name in ("INDEX.md", "README.md"):
            continue
        notes.add(path)
    return notes


def _build_inbound_markdown_ref_map(repo_root: Path, context_dir: Path) -> dict[Path, set[Path]]:
    """Map referenced context note paths to tracked Markdown files that reference them."""
    output = _run(["git", "ls-files", "--", context_dir.as_posix()], cwd=repo_root)
    inbound_refs: dict[Path, set[Path]] = {}
    for line in output.splitlines():
        source_path = _normalize_repo_path(line.strip())
        if source_path is None or source_path.suffix != ".md":
            continue
        full_source_path = repo_root / source_path
        if not full_source_path.is_file():
            continue
        try:
            content = full_source_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for target in _markdown_targets(content):
            target_path = _resolve_markdown_target(
                target, source_dir=source_path.parent, context_dir=context_dir
            )
            if target_path is None or target_path.suffix != ".md" or target_path == source_path:
                continue
            inbound_refs.setdefault(target_path, set()).add(source_path)
    return inbound_refs


def _last_touch(repo_root: Path, path: Path) -> datetime | None:
    """Return the last commit touch date."""
    output = _run(["git", "log", "-1", "--format=%cI", "--", path.as_posix()], cwd=repo_root)
    if not output:
        return None
    return datetime.fromisoformat(output.replace("Z", "+00:00")).astimezone(UTC)


def _has_inbound_markdown_ref(note_path: Path, repo_root: Path, context_dir: Path) -> bool:
    """Check if any other Markdown file under context_dir references note_path."""
    return bool(_build_inbound_markdown_ref_map(repo_root, context_dir).get(note_path))


def _is_referenced_in_catalog(
    path: Path, current_entry: dict[str, Any], entries: list[dict[str, Any]], repo_root: Path
) -> bool:
    """Check if the path is listed elsewhere in catalog.yaml."""
    for entry in entries:
        if entry is current_entry:
            continue
        if _normalize_repo_path(entry.get("path"), repo_root=repo_root) == path:
            return True
        if _normalize_repo_path(entry.get("replacement"), repo_root=repo_root) == path:
            return True
    if _normalize_repo_path(current_entry.get("replacement"), repo_root=repo_root) == path:
        return True
    return False


def _check_superseded_rules(
    entries: list[dict[str, Any]],
    repo_root: Path,
    catalog_path: Path,
    archive_dir: Path,
    proposed_moves: list[ProposedMove],
) -> list[Finding]:
    """Validate Rule A: superseded catalog rows must name their replacement."""
    findings: list[Finding] = []
    for index, entry in enumerate(entries):
        path = _normalize_repo_path(entry.get("path"), repo_root=repo_root)
        status = entry.get("status")
        freshness = entry.get("freshness")
        replacement = _normalize_repo_path(entry.get("replacement"), repo_root=repo_root)

        entry_path = (
            path.as_posix() if path is not None else f"{catalog_path.as_posix()}:entries[{index}]"
        )

        if status != "superseded":
            continue

        if replacement is None:
            findings.append(
                Finding(
                    rule="superseded_replacement",
                    severity="error",
                    path=entry_path,
                    message="superseded catalog entry must name a replacement",
                    status=str(status),
                    freshness=str(freshness) if freshness is not None else None,
                )
            )
        elif not (repo_root / replacement).exists():
            findings.append(
                Finding(
                    rule="superseded_replacement",
                    severity="error",
                    path=entry_path,
                    message=f"superseded replacement file does not exist in repository: {replacement.as_posix()}",
                    status=str(status),
                    freshness=str(freshness) if freshness is not None else None,
                    replacement=replacement.as_posix(),
                )
            )
        elif path is not None and (repo_root / path).is_file() and archive_dir not in path.parents:
            proposed_moves.append(
                ProposedMove(
                    source=path.as_posix(),
                    target=(archive_dir / path.name).as_posix(),
                    category="superseded",
                    reason="superseded note with a named, existing replacement",
                    replacement=replacement.as_posix(),
                    confidence="high",
                )
            )
    return findings


def _check_stale_rules(
    entries: list[dict[str, Any]],
    repo_root: Path,
    archive_dir: Path,
    max_age_days: int,
    now: datetime,
    inbound_markdown_refs: dict[Path, set[Path]],
    proposed_moves: list[ProposedMove],
) -> list[Finding]:
    """Validate Rule B: current dated notes older than max age with no inbound references."""
    findings: list[Finding] = []
    for entry in entries:
        path = _normalize_repo_path(entry.get("path"), repo_root=repo_root)
        status = entry.get("status")
        freshness = entry.get("freshness")
        if status != "current" or freshness != "dated" or path is None:
            continue
        if not (repo_root / path).is_file():
            continue

        touched_at = _last_touch(repo_root, path)
        if touched_at is None:
            continue
        age_days = int((now - touched_at).days)
        if age_days <= max_age_days:
            continue

        has_md_ref = bool(inbound_markdown_refs.get(path))
        has_catalog_ref = _is_referenced_in_catalog(path, entry, entries, repo_root)

        if not (has_md_ref or has_catalog_ref):
            findings.append(
                Finding(
                    rule="stale_current_dated",
                    severity="warning",
                    path=path.as_posix(),
                    message=f"current dated context note is older than {max_age_days} days and has no inbound references",
                    status="current",
                    freshness="dated",
                    last_touched_at=touched_at.isoformat(),
                    age_days=age_days,
                )
            )
            if archive_dir not in path.parents:
                proposed_moves.append(
                    ProposedMove(
                        source=path.as_posix(),
                        target=(archive_dir / path.name).as_posix(),
                        category="stale",
                        reason=f"current dated context note is older than {max_age_days} days and has no inbound references",
                        confidence="review",
                    )
                )
    return findings


def _check_orphan_rules(
    tracked_notes: set[Path],
    entries: list[dict[str, Any]],
    indexed_paths: set[Path],
    repo_root: Path,
) -> list[Finding]:
    """Validate Rule C: orphan notes absent from both INDEX.md and catalog.yaml."""
    catalog_paths_and_reps: set[Path] = set()
    for entry in entries:
        if (p := _normalize_repo_path(entry.get("path"), repo_root=repo_root)) is not None:
            catalog_paths_and_reps.add(p)
        if (r := _normalize_repo_path(entry.get("replacement"), repo_root=repo_root)) is not None:
            catalog_paths_and_reps.add(r)

    findings: list[Finding] = []
    for path in sorted(tracked_notes):
        if path in catalog_paths_and_reps or path in indexed_paths:
            continue
        findings.append(
            Finding(
                rule="orphan_context_note",
                severity="warning",
                path=path.as_posix(),
                message="context note is absent from both docs/context/INDEX.md and catalog.yaml",
            )
        )
    return findings


def check_freshness_decay(
    *,
    repo_root: Path,
    catalog_path: Path = CATALOG_PATH,
    context_dir: Path = CONTEXT_DIR,
    index_path: Path = INDEX_PATH,
    max_age_days: int = 180,
    now: datetime | None = None,
) -> tuple[list[Finding], list[ProposedMove], list[str]]:
    """Scan docs/context notes for staleness, superseded replacement, and orphan signals."""
    repo_root = repo_root.resolve()
    normalized_catalog_path = _normalize_repo_path(catalog_path, repo_root=repo_root)
    normalized_context_dir = _normalize_repo_path(context_dir, repo_root=repo_root)
    normalized_index_path = _normalize_repo_path(index_path, repo_root=repo_root)
    if normalized_catalog_path is None:
        raise ValueError(f"catalog path must be repository-relative: {catalog_path}")
    if normalized_context_dir is None:
        raise ValueError(f"context dir must be repository-relative: {context_dir}")
    if normalized_index_path is None:
        raise ValueError(f"index path must be repository-relative: {index_path}")
    catalog_path = normalized_catalog_path
    context_dir = normalized_context_dir
    index_path = normalized_index_path

    now = (now or datetime.now(UTC)).astimezone(UTC)
    archive_dir = context_dir / "archive"
    entries = _catalog_entries(repo_root / catalog_path)

    findings: list[Finding] = []
    proposed_moves: list[ProposedMove] = []
    conflicts: list[str] = []
    try:
        indexed_paths = _index_references(repo_root / index_path, context_dir=context_dir)
        should_check_orphans = True
    except FileNotFoundError:
        indexed_paths = set()
        should_check_orphans = False
        findings.append(
            Finding(
                rule="missing_context_index",
                severity="error",
                path=index_path.as_posix(),
                message=(
                    "configured context index is missing; orphan checks fail closed because "
                    "empty index references would be misleading"
                ),
            )
        )
    tracked_notes = _tracked_context_notes(repo_root, context_dir)
    inbound_markdown_refs = _build_inbound_markdown_ref_map(repo_root, context_dir)

    findings.extend(
        _check_superseded_rules(entries, repo_root, catalog_path, archive_dir, proposed_moves)
    )
    findings.extend(
        _check_stale_rules(
            entries,
            repo_root,
            archive_dir,
            max_age_days,
            now,
            inbound_markdown_refs,
            proposed_moves,
        )
    )
    if should_check_orphans:
        findings.extend(_check_orphan_rules(tracked_notes, entries, indexed_paths, repo_root))

    by_target: dict[str, list[str]] = {}
    for move in proposed_moves:
        by_target.setdefault(move.target, []).append(move.source)

    for target, sources in sorted(by_target.items()):
        unique_sources = sorted(set(sources))
        if len(unique_sources) > 1:
            conflicts.append(
                f"duplicate_target {target}: multiple source notes map to this target: {', '.join(unique_sources)}"
            )
        elif (repo_root / Path(target)).exists():
            conflicts.append(
                f"target_exists {target}: target archive destination already exists for source: {unique_sources[0]}"
            )

    return findings, proposed_moves, conflicts


def _print_human_report(
    findings: list[Finding],
    proposed_moves: list[ProposedMove],
    conflicts: list[str],
    *,
    strict: bool,
) -> None:
    """Print human-readable report."""
    if findings:
        print("--- FRESHNESS DECAY FINDINGS ---")
        grouped: dict[str, list[Finding]] = {}
        for f in findings:
            grouped.setdefault(f.rule, []).append(f)
        for rule, group in sorted(grouped.items()):
            print(f"{rule}: {len(group)} finding(s)")
            for f in group:
                level = "ERROR" if f.severity == "error" else "WARN"
                print(f"  {level} {f.path}: {f.message}")
    else:
        print("OK: No freshness decay or integrity findings.")

    print("\n--- ARCHIVAL SWEEP DRY-RUN REPORT ---")
    if proposed_moves:
        print(f"Proposed {len(proposed_moves)} archival move(s):")
        for move in proposed_moves:
            print(f"  [{move.confidence}] {move.source} -> {move.target}")
            print(f"      reason: {move.reason}")
    else:
        print("No new archival candidates proposed.")

    if conflicts:
        print("\n--- ARCHIVAL SWEEP CONFLICTS ---")
        for conflict in conflicts:
            print(f"  CRITICAL: {conflict}")


def main() -> int:
    """Run CLI wrapper."""
    parser = argparse.ArgumentParser(
        description="Scans docs/context/ notes for staleness, superseded replacement errors, and orphans."
    )
    parser.add_argument("--max-age-days", type=int, default=180)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write JSON report to this path. Use '-' to print JSON to stdout.",
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--context-dir", type=Path, default=CONTEXT_DIR)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()

    repo_root = _repo_root()
    findings, proposed_moves, conflicts = check_freshness_decay(
        repo_root=repo_root,
        catalog_path=args.catalog,
        context_dir=args.context_dir,
        index_path=args.index,
        max_age_days=args.max_age_days,
    )

    has_errors = any(f.severity == "error" for f in findings)
    has_warnings_under_strict = args.strict and any(f.severity == "warning" for f in findings)
    has_conflicts = len(conflicts) > 0

    exit_code = 0
    if has_errors or has_warnings_under_strict or has_conflicts:
        exit_code = 1

    summary = {
        "schema_version": "context_note_freshness_decay.v1",
        "strict": args.strict,
        "exit_code": exit_code,
        "findings": [asdict(f) for f in findings],
        "proposed_moves": [asdict(m) for m in proposed_moves],
        "conflicts": conflicts,
    }

    if args.json_output:
        text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if str(args.json_output) == "-":
            print(text, end="")
        else:
            output_file = repo_root / args.json_output
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(text, encoding="utf-8")
    else:
        _print_human_report(findings, proposed_moves, conflicts, strict=args.strict)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
