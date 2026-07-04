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
ARCHIVE_DIR = CONTEXT_DIR / "archive"
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


def _safe_repo_path(raw_path: object) -> Path | None:
    """Return a safe repository-relative Path if possible."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path.strip())
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


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
    """Return note paths referenced by INDEX.md."""
    if not index_path.exists():
        return set()
    targets = _markdown_targets(index_path.read_text(encoding="utf-8"))
    references: set[Path] = set()
    for target in targets:
        path = Path(target)
        if path.suffix != ".md":
            continue
        if path.parts[:2] == ("docs", "context"):
            references.add(path)
        else:
            references.add(context_dir / path)
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


def _last_touch(repo_root: Path, path: Path) -> datetime | None:
    """Return the last commit touch date."""
    output = _run(["git", "log", "-1", "--format=%cI", "--", path.as_posix()], cwd=repo_root)
    if not output:
        return None
    return datetime.fromisoformat(output.replace("Z", "+00:00")).astimezone(UTC)


def _has_inbound_markdown_ref(note_path: Path, repo_root: Path, context_dir: Path) -> bool:
    """Check if any other Markdown file under context_dir references note_path."""
    output = _run(["git", "ls-files", "--", context_dir.as_posix()], cwd=repo_root)
    for line in output.splitlines():
        source_path = Path(line.strip())
        if source_path.suffix != ".md" or source_path == note_path:
            continue
        full_source_path = repo_root / source_path
        if not full_source_path.is_file():
            continue
        try:
            content = full_source_path.read_text(encoding="utf-8")
        except OSError:
            continue
        targets = _markdown_targets(content)
        for target in targets:
            target_path = Path(target)
            if target_path.parts[:2] == ("docs", "context"):
                resolved = target_path
            else:
                resolved = Path(os.path.normpath(source_path.parent / target_path))
            if resolved == note_path:
                return True
    return False


def _is_referenced_in_catalog(
    path: Path, current_entry: dict[str, Any], entries: list[dict[str, Any]]
) -> bool:
    """Check if the path is listed elsewhere in catalog.yaml."""
    for entry in entries:
        if entry is current_entry:
            continue
        if _safe_repo_path(entry.get("path")) == path:
            return True
        if _safe_repo_path(entry.get("replacement")) == path:
            return True
    if _safe_repo_path(current_entry.get("replacement")) == path:
        return True
    return False


def _check_superseded_rules(
    entries: list[dict[str, Any]],
    repo_root: Path,
    catalog_path: Path,
    proposed_moves: list[ProposedMove],
) -> list[Finding]:
    """Validate Rule A: superseded catalog rows must name their replacement."""
    findings: list[Finding] = []
    for index, entry in enumerate(entries):
        path = _safe_repo_path(entry.get("path"))
        status = entry.get("status")
        freshness = entry.get("freshness")
        replacement = _safe_repo_path(entry.get("replacement"))

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
        elif path is not None and (repo_root / path).is_file() and ARCHIVE_DIR not in path.parents:
            proposed_moves.append(
                ProposedMove(
                    source=path.as_posix(),
                    target=(ARCHIVE_DIR / path.name).as_posix(),
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
    context_dir: Path,
    max_age_days: int,
    now: datetime,
    proposed_moves: list[ProposedMove],
) -> list[Finding]:
    """Validate Rule B: current dated notes older than max age with no inbound references."""
    findings: list[Finding] = []
    for entry in entries:
        path = _safe_repo_path(entry.get("path"))
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

        has_md_ref = _has_inbound_markdown_ref(path, repo_root, context_dir)
        has_catalog_ref = _is_referenced_in_catalog(path, entry, entries)

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
            if ARCHIVE_DIR not in path.parents:
                proposed_moves.append(
                    ProposedMove(
                        source=path.as_posix(),
                        target=(ARCHIVE_DIR / path.name).as_posix(),
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
) -> list[Finding]:
    """Validate Rule C: orphan notes absent from both INDEX.md and catalog.yaml."""
    catalog_paths_and_reps: set[Path] = set()
    for entry in entries:
        if (p := _safe_repo_path(entry.get("path"))) is not None:
            catalog_paths_and_reps.add(p)
        if (r := _safe_repo_path(entry.get("replacement"))) is not None:
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
    now = (now or datetime.now(UTC)).astimezone(UTC)
    entries = _catalog_entries(repo_root / catalog_path)
    indexed_paths = _index_references(repo_root / index_path, context_dir=context_dir)
    tracked_notes = _tracked_context_notes(repo_root, context_dir)

    findings: list[Finding] = []
    proposed_moves: list[ProposedMove] = []
    conflicts: list[str] = []

    findings.extend(_check_superseded_rules(entries, repo_root, catalog_path, proposed_moves))
    findings.extend(
        _check_stale_rules(entries, repo_root, context_dir, max_age_days, now, proposed_moves)
    )
    findings.extend(_check_orphan_rules(tracked_notes, entries, indexed_paths))

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
