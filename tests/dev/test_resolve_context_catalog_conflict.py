"""Tests for the generated-file conflict resolver (issue #6781)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from scripts.dev import resolve_context_catalog_conflict as resolver

if TYPE_CHECKING:
    from pathlib import Path


CATALOG_HEADER = """version: 1
updated_at: 2026-07-28
policy: context-catalog.v1
description: 'catalog'
status_values:
  - evidence
freshness_values:
  - evidence
entries:
"""


def _catalog_entry(path: str, area: str = "benchmark_evidence") -> str:
    return f"- path: {path}\n  status: evidence\n  freshness: evidence\n  area: {area}\n"


def test_resolve_catalog_union_dedupes_and_preserves_fields(tmp_path: Path) -> None:
    """Both sides' entries are kept, deduped by path, fields complete, sorted."""
    # THEIRS adds a new entry plus a different-area duplicate of docs/a.json.
    conflicted = (
        CATALOG_HEADER
        + "<<<<<<< HEAD\n"
        + _catalog_entry("docs/a.json")
        + _catalog_entry("docs/b.json")
        + "=======\n"
        + _catalog_entry("docs/a.json", area="stale_dup")
        + _catalog_entry("docs/c.json")
        + ">>>>>>> origin/main\n"
    )
    resolved = resolver.resolve_catalog(*resolver._split_conflict_sides(conflicted))
    assert "<<<<<<<" not in resolved and ">>>>>>>" not in resolved and "=======" not in resolved
    parsed = yaml.safe_load(resolved)
    paths = [e["path"] for e in parsed["entries"]]
    assert paths == sorted(paths)  # sorted
    assert set(paths) == {"docs/a.json", "docs/b.json", "docs/c.json"}  # union + dedupe
    # The HEAD instance of the duplicate wins and keeps its complete fields.
    a = next(e for e in parsed["entries"] if e["path"] == "docs/a.json")
    assert a["status"] == "evidence" and a["area"] == "benchmark_evidence"


def test_resolve_index_unions_paragraphs(tmp_path: Path) -> None:
    conflicted = (
        "# Context Retrieval Index\n\n<<<<<<< HEAD\nalpha entry.\n\nbravo entry.\n"
        "=======\nbravo entry.\n\ncharlie entry.\n>>>>>>> origin/main\n"
    )
    resolved = resolver.resolve_index(*resolver._split_conflict_sides(conflicted))
    assert "<<<<<<<" not in resolved
    assert "alpha entry" in resolved and "bravo entry" in resolved and "charlie entry" in resolved
    # Dedupe: bravo appears once.
    assert resolved.count("bravo entry") == 1


def test_resolve_conflict_file_refuses_clean_file(tmp_path: Path) -> None:
    clean = tmp_path / "catalog.yaml"
    clean.write_text(CATALOG_HEADER + _catalog_entry("docs/a.json"), encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="no conflict markers"):
        resolver.resolve_conflict_file(clean)


def test_resolver_roundtrips_a_real_style_conflict(tmp_path: Path) -> None:
    """Mirror the #6662/#6725 pattern: both sides add distinct complete entries."""
    conflicted = (
        CATALOG_HEADER
        + "<<<<<<< HEAD\n"
        + _catalog_entry("docs/context/evidence/issue_6676_smoke")
        + "=======\n"
        + _catalog_entry("docs/context/evidence/issue_6412_pkg")
        + _catalog_entry("docs/context/evidence/issue_6158.md.review.json")
        + ">>>>>>> origin/main\n"
    )
    path = tmp_path / "catalog.yaml"
    path.write_text(conflicted, encoding="utf-8")
    resolver.resolve_conflict_file(path)
    result = path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(result)
    paths = [e["path"] for e in parsed["entries"]]
    assert set(paths) == {
        "docs/context/evidence/issue_6676_smoke",
        "docs/context/evidence/issue_6412_pkg",
        "docs/context/evidence/issue_6158.md.review.json",
    }
    # No entry lost its fields (the #6662 bug class).
    for entry in parsed["entries"]:
        assert entry.get("status") and entry.get("freshness") and entry.get("area")
