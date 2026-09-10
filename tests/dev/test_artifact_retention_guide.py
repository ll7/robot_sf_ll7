"""Consistency checks for the artifact retention and cleanup guide."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUIDE = _REPO_ROOT / "docs/context/artifact_retention_and_cleanup.md"
_PRIMARY_DOCS = (
    _REPO_ROOT / "docs/context/README.md",
    _REPO_ROOT / "docs/context/artifact_evidence_vocabulary.md",
    _REPO_ROOT / "docs/dev/worktree_lifecycle.md",
)
_REQUIRED_HEADINGS = (
    "# Artifact Retention, Preservation, and Cleanup Guide",
    "## 1. Retention classes in operational terms",
    "## 2. Copies are not custody",
    '## 3. Gates before "preserved" or "cleanup-eligible"',
    "## 4. Checked workflows",
    "## 5. Failure examples",
)
_LINK_RE = re.compile(r"\]\(([^)]+)\)")


def test_guide_exists_with_required_sections() -> None:
    """The guide exists and keeps its operational section structure."""
    text = _GUIDE.read_text(encoding="utf-8")
    for heading in _REQUIRED_HEADINGS:
        assert heading in text, heading


def test_guide_links_resolve_to_repository_paths() -> None:
    """Every repository-relative link in the guide resolves to a tracked path."""
    text = _GUIDE.read_text(encoding="utf-8")
    for target in _LINK_RE.findall(text):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (_GUIDE.parent / target.split("#", 1)[0]).resolve()
        assert resolved.exists(), f"missing guide link target: {target}"


def test_primary_docs_link_to_the_guide() -> None:
    """Onboarding, vocabulary, and lifecycle docs point at the guide."""
    for doc in _PRIMARY_DOCS:
        assert "artifact_retention_and_cleanup.md" in doc.read_text(encoding="utf-8"), doc


def test_guide_has_no_absolute_home_paths() -> None:
    """The guide must not embed private absolute home paths."""
    text = _GUIDE.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert not re.search(r"[A-Za-z]:\\\\Users\\\\", text)
