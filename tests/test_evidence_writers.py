"""Tests for the shared evidence writers module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.evidence.writers import (
    review_marker,
    review_marker_comment,
    review_marker_json,
    sha256_file,
    write_csv,
    write_json,
    write_sha256sums,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestReviewMarker:
    """Test the review marker helpers."""

    def test_review_marker_html_comment(self) -> None:
        marker = review_marker("robot_sf#4891")
        assert marker == "<!-- AI-GENERATED (robot_sf#4891) - NEEDS-REVIEW -->"

    def test_review_marker_with_date(self) -> None:
        marker = review_marker("robot_sf#4891", marker_date="2026-07-09")
        assert marker == "<!-- AI-GENERATED (robot_sf#4891, 2026-07-09) - NEEDS-REVIEW -->"

    def test_review_marker_none_date_omits_date(self) -> None:
        marker = review_marker("robot_sf#4891", marker_date=None)
        assert marker == "<!-- AI-GENERATED (robot_sf#4891) - NEEDS-REVIEW -->"

    def test_review_marker_json(self) -> None:
        assert review_marker_json() == "AI-GENERATED NEEDS-REVIEW"

    def test_review_marker_comment(self) -> None:
        assert review_marker_comment() == "# AI-GENERATED NEEDS-REVIEW"


class TestSha256File:
    """Test the sha256_file helper."""

    def test_known_content(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello\n", encoding="utf-8")
        digest = sha256_file(test_file)
        assert len(digest) == 64
        assert digest == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


class TestWriteJson:
    """Test the write_json helper."""

    def test_adds_review_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        payload = {"key": "value"}
        write_json(path, payload)
        content = path.read_text(encoding="utf-8")
        assert "AI-GENERATED NEEDS-REVIEW" in content
        assert '"review_marker"' in content

    def test_preserves_original_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        payload = {"key": "value", "nested": {"a": 1}}
        write_json(path, payload)
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["key"] == "value"
        assert data["nested"]["a"] == 1
        assert data["review_marker"] == "AI-GENERATED NEEDS-REVIEW"


class TestWriteCsv:
    """Test the write_csv helper."""

    def test_prepends_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        rows = [{"col1": "val1", "col2": "val2"}]
        write_csv(path, rows)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# AI-GENERATED NEEDS-REVIEW\n")

    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        rows = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        write_csv(path, rows)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0] == "# AI-GENERATED NEEDS-REVIEW"
        assert "a,b" in lines[1]
        assert "1,2" in lines[2]


class TestWriteSha256sums:
    """Test the write_sha256sums helper."""

    def test_prepends_marker(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("data", encoding="utf-8")
        write_sha256sums(tmp_path)
        sha_file = tmp_path / "SHA256SUMS"
        content = sha_file.read_text(encoding="utf-8")
        assert content.startswith("# AI-GENERATED NEEDS-REVIEW\n")

    def test_excludes_self_from_hash(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("data", encoding="utf-8")
        write_sha256sums(tmp_path)
        sha_file = tmp_path / "SHA256SUMS"
        content = sha_file.read_text(encoding="utf-8")
        assert "SHA256SUMS" not in content.split("\n", 1)[1]
