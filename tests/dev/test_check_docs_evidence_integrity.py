"""Tests for the changed-path docs/evidence integrity check (issue #3476)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.dev.check_docs_evidence_integrity import check_files

if TYPE_CHECKING:
    from pathlib import Path


def test_valid_files_pass(tmp_path: Path) -> None:
    """Well-formed JSON/YAML and a resolvable relative link must produce no problems."""
    (tmp_path / "target.md").write_text("# target\n", encoding="utf-8")
    (tmp_path / "data.json").write_text('{"ok": true}\n', encoding="utf-8")
    (tmp_path / "data.yaml").write_text("schema_version: 1\nitems: [a, b]\n", encoding="utf-8")
    (tmp_path / "note.md").write_text(
        "See [target](./target.md) and [home](https://example.com) and [anchor](#section).\n",
        encoding="utf-8",
    )

    problems = check_files(["data.json", "data.yaml", "note.md"], root=tmp_path)

    assert problems == []


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    """Malformed JSON in a changed evidence file must be flagged."""
    (tmp_path / "broken.json").write_text("{not valid json}", encoding="utf-8")

    problems = check_files(["broken.json"], root=tmp_path)

    assert len(problems) == 1
    assert "invalid JSON" in problems[0]


def test_invalid_yaml_is_reported(tmp_path: Path) -> None:
    """Malformed YAML in a changed catalogue/manifest file must be flagged."""
    (tmp_path / "broken.yaml").write_text("key: [unclosed\n", encoding="utf-8")

    problems = check_files(["broken.yaml"], root=tmp_path)

    assert len(problems) == 1
    assert "invalid YAML" in problems[0]


def test_broken_relative_link_is_reported(tmp_path: Path) -> None:
    """A repo-local relative link to a missing file must be flagged."""
    (tmp_path / "note.md").write_text("See [gone](./missing.md).\n", encoding="utf-8")

    problems = check_files(["note.md"], root=tmp_path)

    assert len(problems) == 1
    assert "broken repo-local link" in problems[0]
    assert "./missing.md" in problems[0]


def test_external_and_anchor_links_are_not_flagged(tmp_path: Path) -> None:
    """External URLs, mailto, and pure anchors must never be treated as broken paths."""
    (tmp_path / "note.md").write_text(
        "[web](https://example.com/x) [mail](mailto:a@b.c) [top](#top) [bare](some/other.md)\n",
        encoding="utf-8",
    )

    problems = check_files(["note.md"], root=tmp_path)

    assert problems == []


def test_relative_link_escaping_repo_is_reported(tmp_path: Path) -> None:
    """A relative link resolving outside the repository root must be flagged."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (tmp_path / "outside.md").write_text("# outside\n", encoding="utf-8")
    (root / "docs" / "note.md").write_text("[up](../../outside.md)\n", encoding="utf-8")

    problems = check_files(["docs/note.md"], root=root)

    assert len(problems) == 1
    assert "escapes the repository" in problems[0]


def test_link_with_anchor_fragment_resolves_to_file(tmp_path: Path) -> None:
    """A relative link with an anchor fragment must validate the file part only."""
    (tmp_path / "target.md").write_text("# target\n", encoding="utf-8")
    (tmp_path / "note.md").write_text("[sec](./target.md#section)\n", encoding="utf-8")

    assert check_files(["note.md"], root=tmp_path) == []
