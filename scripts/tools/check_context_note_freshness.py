#!/usr/bin/env python3
"""Check context-note freshness, supersession integrity, and index coverage."""

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
EVIDENCE_DIR = CONTEXT_DIR / "evidence"
ARCHIVE_DIR = CONTEXT_DIR / "archive"
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)|<((?:\./)?[^ >]+)>")


@dataclass(frozen=True, slots=True)
class Finding:
    """One context-note freshness finding."""

    rule: str
    severity: str
    path: str
    message: str
    status: str | None = None
    freshness: str | None = None
    replacement: str | None = None
    last_touched_at: str | None = None
    age_days: int | None = None


def _run(cmd: list[str], *, cwd: Path) -> str:
    """Run a command and return stdout."""

    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def _repo_root() -> Path:
    """Return the current repository root."""

    return Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd())).resolve()


def _load_yaml(path: Path) -> object:
    """Load a YAML file."""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _safe_repo_path(raw_path: object) -> Path | None:
    """Return a safe repository-relative path value, if present."""

    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path.strip())
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _catalog_entries(catalog_path: Path) -> list[dict[str, Any]]:
    """Return catalog entry mappings from the context catalog."""

    payload = _load_yaml(catalog_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{catalog_path.as_posix()} must be a YAML mapping")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{catalog_path.as_posix()} entries must be a list")
    return [entry for entry in entries if isinstance(entry, dict)]


def _markdown_targets(text: str) -> set[str]:
    """Return normalized Markdown link targets without anchors or query strings."""

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
    """Return context note paths referenced by docs/context/INDEX.md."""

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
    """Return tracked Markdown context notes directly under context_dir, excluding INDEX.md and README.md."""

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
    """Return the last committed touch time for a path, if available."""

    output = _run(["git", "log", "-1", "--format=%cI", "--", path.as_posix()], cwd=repo_root)
    if not output:
        return None
    return datetime.fromisoformat(output.replace("Z", "+00:00")).astimezone(UTC)


def _superseded_replacement_findings(
    entries: list[dict[str, Any]], *, catalog_path: Path, repo_root: Path
) -> list[Finding]:
    """Return hard errors for superseded entries without valid replacement metadata in the repo."""

    findings: list[Finding] = []
    for index, entry in enumerate(entries):
        path = _safe_repo_path(entry.get("path"))
        status = entry.get("status")
        if status != "superseded":
            continue
        freshness = entry.get("freshness")
        replacement = _safe_repo_path(entry.get("replacement"))

        entry_path = (
            path.as_posix() if path is not None else f"{catalog_path.as_posix()}:entries[{index}]"
        )

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
    return findings


def _catalog_paths_and_replacements(entries: list[dict[str, Any]]) -> set[Path]:
    """Return safe catalog path and replacement values from entries."""

    paths = set()
    for entry in entries:
        if (p := _safe_repo_path(entry.get("path"))) is not None:
            paths.add(p)
        if (r := _safe_repo_path(entry.get("replacement"))) is not None:
            paths.add(r)
    return paths


def _is_listed_elsewhere_in_catalog(
    path: Path, current_entry: dict[str, Any], entries: list[dict[str, Any]]
) -> bool:
    """Return whether the path is listed in catalog.yaml in another context."""

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


def _has_inbound_markdown_reference(
    note_path: Path,
    repo_root: Path,
    context_dir: Path,
) -> bool:
    """Return whether any tracked markdown file under context_dir (other than note_path) references note_path."""

    output = _run(["git", "ls-files", "--", context_dir.as_posix()], cwd=repo_root)
    for line in output.splitlines():
        source_path = Path(line.strip())
        if source_path.suffix != ".md":
            continue
        if source_path == note_path:
            continue
        full_source_path = repo_root / source_path
        if not full_source_path.is_file():
            continue
        try:
            content = full_source_path.read_text(encoding="utf-8")
        except Exception:
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


def _stale_current_dated_findings(
    entries: list[dict[str, Any]],
    *,
    repo_root: Path,
    context_dir: Path,
    max_age_days: int,
    now: datetime,
) -> list[Finding]:
    """Return warnings for old current+dated notes with no inbound references."""

    findings: list[Finding] = []
    for entry in entries:
        path = _safe_repo_path(entry.get("path"))
        if entry.get("status") != "current" or entry.get("freshness") != "dated" or path is None:
            continue
        if not (repo_root / path).is_file():
            continue
        touched_at = _last_touch(repo_root, path)
        if touched_at is None:
            continue
        age_days = int((now - touched_at).days)
        if age_days <= max_age_days:
            continue

        # Check inbound references
        has_md_ref = _has_inbound_markdown_reference(path, repo_root, context_dir)
        has_catalog_ref = _is_listed_elsewhere_in_catalog(path, entry, entries)

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
    return findings


def _orphan_context_note_findings(
    *,
    tracked_notes: set[Path],
    catalog_paths_and_replacements: set[Path],
    index_references: set[Path],
) -> list[Finding]:
    """Return warnings for tracked notes absent from catalog and index."""

    findings: list[Finding] = []
    for path in sorted(tracked_notes):
        if path in catalog_paths_and_replacements or path in index_references:
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


def check_freshness(
    *,
    repo_root: Path,
    catalog_path: Path = CATALOG_PATH,
    context_dir: Path = CONTEXT_DIR,
    index_path: Path = INDEX_PATH,
    max_age_days: int = 180,
    now: datetime | None = None,
) -> list[Finding]:
    """Return context-note freshness findings."""

    now = (now or datetime.now(UTC)).astimezone(UTC)
    catalog_full_path = repo_root / catalog_path
    entries = _catalog_entries(catalog_full_path)

    indexed_paths = _index_references(repo_root / index_path, context_dir=context_dir)
    findings = _superseded_replacement_findings(
        entries, catalog_path=catalog_path, repo_root=repo_root
    )
    findings.extend(
        _stale_current_dated_findings(
            entries,
            repo_root=repo_root,
            context_dir=context_dir,
            max_age_days=max_age_days,
            now=now,
        )
    )

    catalog_paths_and_reps = _catalog_paths_and_replacements(entries)
    findings.extend(
        _orphan_context_note_findings(
            tracked_notes=_tracked_context_notes(repo_root, context_dir),
            catalog_paths_and_replacements=catalog_paths_and_reps,
            index_references=indexed_paths,
        )
    )
    return findings


def _summary(findings: list[Finding], *, strict: bool) -> dict[str, Any]:
    """Return a compact JSON-ready summary."""

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.rule] = counts.get(finding.rule, 0) + 1
    has_errors = any(finding.severity == "error" for finding in findings)
    has_strict_warnings = strict and any(finding.severity == "warning" for finding in findings)
    return {
        "schema_version": "context_note_freshness.v1",
        "strict": strict,
        "counts": counts,
        "exit_code": 1 if has_errors or has_strict_warnings else 0,
        "findings": [asdict(finding) for finding in findings],
    }


def _print_human_report(findings: list[Finding], *, max_findings: int) -> None:
    """Print a grouped, bounded human-readable report."""

    if not findings:
        print("OK context note freshness check passed")
        return

    grouped: dict[str, list[Finding]] = {}
    for finding in findings:
        grouped.setdefault(finding.rule, []).append(finding)
    for rule, group in sorted(grouped.items()):
        print(f"{rule}: {len(group)} finding(s)")
        for finding in group[:max_findings]:
            level = "ERROR" if finding.severity == "error" else "WARN"
            print(f"  {level} {finding.path}: {finding.message}")
        omitted = len(group) - max_findings
        if omitted > 0:
            print(f"  ... {omitted} more; use --json-output for full detail")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Check docs/context note freshness and catalog/index coverage."
    )
    parser.add_argument("--max-age-days", type=int, default=180)
    parser.add_argument("--strict", action="store_true", help="Treat warning findings as errors.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write a JSON report to this path. Use '-' to print JSON to stdout.",
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--context-dir", type=Path, default=CONTEXT_DIR)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    parser.add_argument(
        "--max-human-findings",
        type=int,
        default=50,
        help="Maximum findings to print per rule in human-readable output.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the context-note freshness checker."""

    args = _parse_args()
    repo_root = _repo_root()
    findings = check_freshness(
        repo_root=repo_root,
        catalog_path=args.catalog,
        context_dir=args.context_dir,
        index_path=args.index,
        max_age_days=args.max_age_days,
    )
    payload = _summary(findings, strict=bool(args.strict))
    if args.json_output:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if str(args.json_output) == "-":
            print(text, end="")
        else:
            output_path = repo_root / args.json_output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
    else:
        _print_human_report(findings, max_findings=max(0, int(args.max_human_findings)))
    return int(payload["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
