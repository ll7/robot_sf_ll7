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


def iter_links(source: Path) -> list[tuple[int, str]]:
    """Return ``(line_number, target)`` pairs for one curated page."""
    links: list[tuple[int, str]] = []
    suffix = source.suffix.lower()
    in_fence = False
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if suffix == ".md" and FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if suffix == ".md":
            links.extend((number, target) for _, target in MD_LINK_RE.findall(line))
        elif suffix == ".rst":
            links.extend((number, target) for target in RST_LINK_RE.findall(line))
            links.extend((number, f"#{target}") for target in RST_REF_RE.findall(line))
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


def _resolve_file(
    source: Path, line: int, target: str, path_part: str
) -> tuple[Path | None, str | None]:
    """Resolve the file part of one link; return (path, reason) with path None on failure."""
    candidate = (source.parent / path_part).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError:
        return None, "path_escape"
    if not candidate.exists() and _has_case_mismatch(candidate):
        return None, "case_mismatch"
    if candidate.is_dir():
        for index in ("README.md", "index.rst", "index.md"):
            if (candidate / index).is_file():
                return candidate / index, None
        return None, "missing_file"
    if not candidate.is_file():
        return None, "missing_file"
    if str(candidate.relative_to(REPO_ROOT)) in GENERATED_ALIASES:
        manifest = REPO_ROOT / "examples" / "examples_manifest.yaml"
        if not manifest.is_file():
            return None, "alias_drift"
    return candidate, None


def _source_label(source: Path) -> str:
    """Return the repository-relative source label for stable reports."""
    try:
        return str(source.relative_to(REPO_ROOT))
    except ValueError:
        return str(source)


def _has_case_mismatch(candidate: Path) -> bool:
    """Return whether a missing path differs only by case from an existing path."""
    current = REPO_ROOT
    for component in candidate.relative_to(REPO_ROOT).parts:
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
        return Finding(_source_label(source), line, target, str(resolved), "missing_fragment")
    if fragment not in anchors:
        return Finding(_source_label(source), line, target, str(resolved), "missing_fragment")
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
        return Finding(_source_label(source), line, target, path_part, reason)
    return _check_fragment(source, line, target, resolved)


def check_curated(root: Path = REPO_ROOT) -> list[Finding]:
    """Check every curated page; findings sort by (source, line, target)."""
    global REPO_ROOT
    REPO_ROOT = root
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
