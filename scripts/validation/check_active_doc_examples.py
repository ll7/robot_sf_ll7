#!/usr/bin/env python3
"""Check active documentation for stale command and artifact examples."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


ALLOW_MARKERS = ("active-docs-check: allow", "active-docs: allow")
DEFAULT_EXCLUDED_DIRS = (
    Path("docs/context"),
    Path("docs/context_packs"),
    Path("docs/dev/issues"),
    Path("docs/experiments"),
    Path("docs/superpowers"),
)
DEFAULT_ROOT_FILES = (
    Path("README.md"),
    Path("AGENTS.md"),
    Path("CHANGELOG.md"),
    Path("ACKNOWLEDGMENTS.md"),
)
DEFAULT_SCAN_ROOTS = (Path("docs"), Path("scripts"))
SPEC_DOC_NAMES = {"quickstart.md", "readme.md"}
TEXT_SUFFIXES = {".md", ".py", ".rst", ".txt"}


@dataclass(frozen=True)
class Diagnostic:
    """One stale active-documentation example."""

    path: Path
    line: int
    rule: str
    message: str
    text: str


@dataclass(frozen=True)
class Rule:
    """Line-oriented active-docs rule."""

    name: str
    pattern: re.Pattern[str]
    message: str


RULES = (
    Rule(
        name="legacy-results-path",
        pattern=re.compile(r"(?<![\w/-])results/"),
        message="legacy results/ path should use output/ unless the line is explicitly historical",
    ),
    Rule(
        name="nested-output-results-path",
        pattern=re.compile(r"\boutput/results/"),
        message="output/results/ is stale; use a canonical output/<artifact-family>/ path",
    ),
    Rule(
        name="benchmark-quickstart-output-path",
        pattern=re.compile(r"\boutput/benchmark_quickstart/"),
        message="output/benchmark_quickstart/ is stale; use output/benchmarks/ or a named demo root",
    ),
    Rule(
        name="bare-python-scripts-command",
        pattern=re.compile(r"\bpython\s+scripts/"),
        message="script commands should normally use uv run python scripts/...",
    ),
    Rule(
        name="setup-pip-install",
        pattern=re.compile(r"\bpip\s+install\b"),
        message="setup docs should prefer uv sync, uv run, or the official uv installer",
    ),
    Rule(
        name="unsupported-robot-sf-bench-run-flag",
        pattern=re.compile(
            r"\brobot_sf_bench\s+run\b[^\n`]*(?:--suite|--episodes|--snqi-|--quiet|--fail-fast)"
        ),
        message="robot_sf_bench run examples should use the current --matrix/--out contract",
    ),
)
SCRIPT_COMMAND_RE = re.compile(
    r"(?:uv\s+run\s+)?python\s+(?P<script>scripts/[A-Za-z0-9_./-]+\.py)\b"
)


def _repo_root() -> Path:
    """Return the repository root for the current checkout."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to resolve repository root")
    return Path(proc.stdout.strip()).resolve()


def _is_within(path: Path, root: Path) -> bool:
    """Return whether a repository-relative path is at or below root."""
    return path == root or root in path.parents


def _has_allow_marker(line: str) -> bool:
    """Return whether a source line carries an explicit active-docs allow marker."""
    return any(marker in line for marker in ALLOW_MARKERS)


def _looks_historical_line(line: str) -> bool:
    """Return whether a single line explicitly describes historical or legacy state."""
    lowered = line.lower()
    return "legacy" in lowered or "historical" in lowered or "migrated" in lowered


def _root_doc_paths(repo_root: Path) -> set[Path]:
    """Return existing top-level active documentation files."""
    return {path for path in DEFAULT_ROOT_FILES if (repo_root / path).is_file()}


def _is_default_scan_candidate(relative: Path, *, include_cli_sources: bool) -> bool:
    """Return whether a repository-relative path is in the default active-doc set."""
    if relative.suffix.lower() not in TEXT_SUFFIXES:
        return False
    if any(_is_within(relative, excluded) for excluded in DEFAULT_EXCLUDED_DIRS):
        return False
    if relative.suffix == ".py" and not include_cli_sources:
        return False
    return not (
        relative.parts[0] == "scripts" and relative.suffix != ".md" and not include_cli_sources
    )


def _scan_root_doc_paths(repo_root: Path, *, include_cli_sources: bool) -> set[Path]:
    """Return active docs found below default scan roots."""
    paths: set[Path] = set()
    for root in DEFAULT_SCAN_ROOTS:
        absolute_root = repo_root / root
        if not absolute_root.exists():
            continue
        for candidate in absolute_root.rglob("*"):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(repo_root)
            if _is_default_scan_candidate(relative, include_cli_sources=include_cli_sources):
                paths.add(relative)
    return paths


def _spec_doc_paths(repo_root: Path) -> set[Path]:
    """Return specs quickstart/readme paths for explicit opt-in scans."""
    specs_root = repo_root / "specs"
    if not specs_root.exists():
        return set()
    return {
        candidate.relative_to(repo_root)
        for candidate in specs_root.rglob("*.md")
        if candidate.name.lower() in SPEC_DOC_NAMES
    }


def _default_paths(
    repo_root: Path, *, include_specs: bool, include_cli_sources: bool
) -> list[Path]:
    """Return default active documentation paths."""
    paths = _root_doc_paths(repo_root)
    paths.update(_scan_root_doc_paths(repo_root, include_cli_sources=include_cli_sources))

    if include_specs:
        paths.update(_spec_doc_paths(repo_root))

    return sorted(paths)


def _normalize_explicit_paths(paths: Sequence[str], repo_root: Path) -> list[Path]:
    """Normalize explicit CLI paths into repository-relative paths."""
    normalized: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(f"path is outside repository: {raw_path}") from exc
        if ".." in path.parts:
            raise ValueError(f"path must stay inside repository: {raw_path}")
        normalized.append(path)
    return normalized


def _phantom_script_diagnostics(
    path: Path, line_no: int, line: str, repo_root: Path
) -> list[Diagnostic]:
    """Return diagnostics for script commands that reference missing script files."""
    diagnostics: list[Diagnostic] = []
    for match in SCRIPT_COMMAND_RE.finditer(line):
        script_path = Path(match.group("script"))
        if any(token in script_path.as_posix() for token in ("...", "<", ">")):
            continue
        if (repo_root / script_path).is_file():
            continue
        diagnostics.append(
            Diagnostic(
                path=path,
                line=line_no,
                rule="phantom-script-path",
                message=f"referenced script does not exist: {script_path.as_posix()}",
                text=line.strip(),
            )
        )
    return diagnostics


def _rule_matches(rule: Rule, line: str) -> bool:
    """Return whether rule matches a line after rule-specific false-positive handling."""
    for match in rule.pattern.finditer(line):
        if rule.name == "bare-python-scripts-command" and "uv run" in line[: match.start()]:
            continue
        return True
    return False


def scan_file(path: Path, repo_root: Path) -> list[Diagnostic]:
    """Scan one repository-relative text file."""
    absolute_path = repo_root / path
    try:
        lines = absolute_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []

    diagnostics: list[Diagnostic] = []
    for line_no, line in enumerate(lines, start=1):
        if _has_allow_marker(line):
            continue
        for rule in RULES:
            if not _rule_matches(rule, line):
                continue
            if rule.name == "legacy-results-path" and _looks_historical_line(line):
                continue
            diagnostics.append(
                Diagnostic(
                    path=path,
                    line=line_no,
                    rule=rule.name,
                    message=rule.message,
                    text=line.strip(),
                )
            )
        diagnostics.extend(_phantom_script_diagnostics(path, line_no, line, repo_root))
    return diagnostics


def scan_paths(paths: Iterable[Path], repo_root: Path) -> list[Diagnostic]:
    """Scan repository-relative paths and return all diagnostics."""
    diagnostics: list[Diagnostic] = []
    for path in paths:
        if (repo_root / path).is_file():
            diagnostics.extend(scan_file(path, repo_root))
    return diagnostics


def _format_diagnostic(diagnostic: Diagnostic) -> str:
    """Format one diagnostic for terminal output."""
    return (
        f"{diagnostic.path}:{diagnostic.line}: {diagnostic.rule}: "
        f"{diagnostic.message}\n    {diagnostic.text}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional repository-relative files to scan instead of the default active-doc set.",
    )
    parser.add_argument(
        "--include-specs",
        action="store_true",
        help="Include specs/** quickstart/readme files in the default scan.",
    )
    parser.add_argument(
        "--include-cli-sources",
        action="store_true",
        help="Include Python CLI sources under docs/scripts surfaces.",
    )
    parser.add_argument(
        "--fail-on-diagnostic",
        action="store_true",
        help="Exit non-zero when stale active-doc examples are found.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the active-doc examples check."""
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root()

    try:
        paths = (
            _normalize_explicit_paths(args.paths, repo_root)
            if args.paths
            else _default_paths(
                repo_root,
                include_specs=args.include_specs,
                include_cli_sources=args.include_cli_sources,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))

    diagnostics = scan_paths(paths, repo_root)
    if diagnostics:
        for diagnostic in diagnostics:
            print(_format_diagnostic(diagnostic))
        print(f"\nFound {len(diagnostics)} active-doc example diagnostic(s).", file=sys.stderr)
        return 1 if args.fail_on_diagnostic else 0

    print(f"Scanned {len(paths)} active-doc file(s); no stale examples found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
