#!/usr/bin/env python3
"""Validate internal links and anchors on the curated documentation surface.

Curated set: ``README.md``, ``CONTRIBUTING.md``, ``docs/index.rst``,
``docs/adoption_path.md``, ``docs/user-guide.md`` (issue #8725).

Resolves repository-relative file targets, directory README targets, ``#fragment``
anchors (Markdown GitHub slugs, explicit reStructuredText ``.. _label:`` targets),
and known generated-file aliases — without network access. External URLs are
skipped, never fetched.

Exit status is ``0`` when every curated internal link resolves and ``1`` when any
finding is reported. ``--format json`` emits the byte-stable machine-readable
report (sorted by source, line, target); ``--format text`` emits one line per
finding for human review.
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CURATED_PAGES = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/index.rst",
    "docs/adoption_path.md",
    "docs/user-guide.md",
)

GENERATED_ALIASES = frozenset({"examples/README.md"})

MD_LINK_RE = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
RST_LINK_RE = re.compile(r"`[^`<]*<([^`>]+)>`_")
RST_LABEL_RE = re.compile(r"^\s*\.\.\s+_([^:]+):")
RST_REF_RE = re.compile(r":(?:doc|ref):`([^`]+)`")
RST_TOCTREE_RE = re.compile(r"^(?P<indent>[ \t]*)\.\.\s+toctree::\s*(?:#.*)?$")
MD_ATX_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
MD_ANCHOR_RE = re.compile(r'<a\s+(?:name|id)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(```|~~~)")

EXTERNAL_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|mailto:|#?$)")


def github_slug(text: str) -> str:
    """Return the GitHub-style fragment slug for one Markdown heading."""
    text = re.sub(r"<[^>]+>", "", text.strip().lower())
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text).strip("-")


def markdown_fragments(path: Path) -> set[str]:
    """Collect linkable fragments (slugs + explicit anchors) of one Markdown file."""
    fragments: set[str] = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = MD_ATX_RE.match(line)
        if match:
            fragments.add(github_slug(match.group(2)))
        for anchor in MD_ANCHOR_RE.findall(line):
            fragments.add(anchor)
    return fragments


def rst_labels(path: Path) -> set[str]:
    """Collect explicit ``.. _label:`` link targets of one reST file."""
    labels: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = RST_LABEL_RE.match(line)
        if match:
            labels.add(match.group(1).strip())
    return labels


def _toctree_target(entry: str) -> str:
    """Return the target from one Sphinx toctree entry."""
    if entry.endswith(">") and "<" in entry:
        return entry[entry.rfind("<") + 1 : -1].strip()
    return entry


def _toctree_entries(
    lines: list[str], directive_line: int, directive_indent: int
) -> tuple[list[tuple[int, str]], int]:
    """Return entries after one toctree directive and the first unconsumed line."""
    entries: list[tuple[int, str]] = []
    number = directive_line + 1
    while number < len(lines):
        line = lines[number]
        if not line.strip():
            number += 1
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= directive_indent:
            break
        entry = line.strip()
        if not entry.startswith((":", "..")):
            target = _toctree_target(entry)
            if target:
                entries.append((number + 1, target))
        number += 1
    return entries, number


def iter_links(source: Path) -> list[tuple[int, str]]:
    """Return ``(line_number, target)`` pairs for one curated page."""
    links: list[tuple[int, str]] = []
    suffix = source.suffix.lower()
    in_fence = False
    lines = source.read_text(encoding="utf-8").splitlines()
    if suffix == ".rst":
        number = 0
        while number < len(lines):
            line = lines[number]
            toctree = RST_TOCTREE_RE.match(line)
            if toctree:
                entries, number = _toctree_entries(lines, number, len(toctree.group("indent")))
                links.extend(entries)
                continue
            links.extend((number + 1, target) for target in RST_LINK_RE.findall(line))
            links.extend((number + 1, f"#{target}") for target in RST_REF_RE.findall(line))
            number += 1
        return links
    for number, line in enumerate(lines, 1):
        if suffix == ".md" and FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if suffix == ".md":
            links.extend((number, target) for _, target in MD_LINK_RE.findall(line))
    return links


@dataclass(frozen=True)
class Finding:
    """One unresolved curated documentation link with a stable reason code."""

    source: str
    line: int
    target: str
    resolved: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-serializable finding record."""
        return {
            "source": self.source,
            "line": self.line,
            "target": self.target,
            "resolved": self.resolved,
            "reason": self.reason,
        }


def _repository_relative(path: Path) -> str | None:
    """Return a normalized repository-relative path, or None when outside the repository."""
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None


def _stable_unresolved_path(source: Path, path_part: str) -> str:
    """Return a root-independent path for a failed resolution attempt."""
    relative = _repository_relative(source.parent / path_part)
    if relative is not None:
        return relative
    if Path(path_part).is_absolute():
        return Path(path_part).as_posix()
    return posixpath.normpath(path_part)


def _resolve_candidates(source: Path, path_part: str) -> list[Path]:
    """Return exact and extensionless Sphinx candidates in deterministic order."""
    candidate = source.parent / path_part
    candidates = [candidate]
    path = Path(path_part)
    if not path_part.endswith(("/", "\\")) and path.suffix == "" and path.name not in {".", ".."}:
        candidates.extend(
            (
                candidate.with_name(f"{candidate.name}.rst"),
                candidate.with_name(f"{candidate.name}.md"),
            )
        )
    return candidates


def _resolve_candidate(candidate: Path) -> tuple[Path | None, str | None]:
    """Resolve one candidate and return a stable failure reason when it is unusable."""
    candidate = candidate.resolve()
    if _repository_relative(candidate) is None:
        return None, "path_escape"
    if not candidate.exists():
        reason = "case_mismatch" if _has_case_mismatch(candidate) else "missing_file"
        return None, reason
    if candidate.is_dir():
        for index in ("README.md", "index.rst", "index.md"):
            if (candidate / index).is_file():
                return candidate / index, None
        return None, "missing_file"
    if not candidate.is_file():
        return None, "missing_file"
    if _repository_relative(candidate) in GENERATED_ALIASES:
        manifest = REPO_ROOT / "examples" / "examples_manifest.yaml"
        if not manifest.is_file():
            return None, "alias_drift"
    return candidate, None


def _resolve_file(
    source: Path, line: int, target: str, path_part: str
) -> tuple[Path | None, str | None]:
    """Resolve one file or directory-index target; return its path and failure reason."""
    saw_case_mismatch = False
    for raw_candidate in _resolve_candidates(source, path_part):
        resolved, reason = _resolve_candidate(raw_candidate)
        if resolved is not None:
            return resolved, None
        if reason in {"path_escape", "alias_drift"}:
            return None, reason
        saw_case_mismatch |= reason == "case_mismatch"
    if saw_case_mismatch:
        return None, "case_mismatch"
    return None, "missing_file"


def _source_label(source: Path) -> str:
    """Return the repository-relative source label for stable reports."""
    return _repository_relative(source) or source.as_posix()


def _has_case_mismatch(candidate: Path) -> bool:
    """Return whether a missing path differs only by case from an existing path."""
    current = REPO_ROOT.resolve()
    for component in candidate.relative_to(current).parts:
        if not current.is_dir():
            return False
        names = {entry.name for entry in current.iterdir()}
        if component in names:
            current /= component
            continue
        return any(name.casefold() == component.casefold() for name in names)
    return False


def _check_fragment(source: Path, line: int, target: str, resolved: Path) -> Finding | None:
    """Return a missing_fragment Finding when the #fragment is not linkable, else None."""
    _, _, fragment = target.partition("#")
    if not fragment:
        return None
    if resolved.suffix.lower() == ".md":
        anchors = markdown_fragments(resolved)
    elif resolved.suffix.lower() == ".rst":
        anchors = rst_labels(resolved)
    else:
        return Finding(
            _source_label(source), line, target, _source_label(resolved), "missing_fragment"
        )
    if fragment not in anchors:
        return Finding(
            _source_label(source), line, target, _source_label(resolved), "missing_fragment"
        )
    return None


def check_target(source: Path, line: int, target: str) -> Finding | None:
    """Return a Finding when one curated link fails to resolve, else None."""
    if EXTERNAL_RE.match(target) or target.startswith(("http://", "https://", "mailto:")):
        return None
    path_part, _, _ = target.partition("#")
    if not path_part:
        return _check_fragment(source, line, target, source)
    resolved, reason = _resolve_file(source, line, target, path_part)
    if resolved is None:
        assert reason is not None
        return Finding(
            _source_label(source),
            line,
            target,
            _stable_unresolved_path(source, path_part),
            reason,
        )
    return _check_fragment(source, line, target, resolved)


def check_curated(root: Path | None = None) -> list[Finding]:
    """Check every curated page; findings sort by (source, line, target)."""
    global REPO_ROOT
    previous_root = REPO_ROOT
    root = (REPO_ROOT if root is None else Path(root)).resolve()
    REPO_ROOT = root
    try:
        findings: list[Finding] = []
        for page in CURATED_PAGES:
            source = root / page
            if not source.is_file():
                findings.append(Finding(page, 0, page, page, "missing_file"))
                continue
            for line, target in iter_links(source):
                finding = check_target(source, line, target)
                if finding is not None:
                    findings.append(finding)
        findings.sort(key=lambda f: (f.source, f.line, f.target))
        return findings
    finally:
        REPO_ROOT = previous_root


def main(argv: list[str] | None = None) -> int:
    """Run the curated link check; return 1 under --check when findings exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 when findings exist")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    args = parser.parse_args(argv)
    findings = check_curated()
    if args.format == "json":
        print(json.dumps([f.as_dict() for f in findings], indent=2, sort_keys=False))
    else:
        for finding in findings:
            print(
                f"{finding.source}:{finding.line}: {finding.target} "
                f"-> {finding.resolved} [{finding.reason}]"
            )
        print(f"{len(findings)} finding(s) across {len(CURATED_PAGES)} curated pages.")
    if args.check and findings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
