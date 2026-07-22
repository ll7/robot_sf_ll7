"""Migration tests: AI-GENERATED markers and byte-determinism per provenance timestamp.

These tests prove that the shared evidence writers produce output carrying the
required AI-GENERATED markers and are byte-deterministic when called with the
same parameters (including the same ``marker_date``).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from robot_sf.evidence.writers import (
    review_marker_json,
    write_csv,
    write_json,
    write_sha256sums,
    write_text,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestWriteJsonMarker:
    """write_json always includes AI-GENERATED NEEDS-REVIEW."""

    def test_marker_present(self, tmp_path: Path) -> None:
        path = tmp_path / "report.json"
        write_json(path, {"key": "value"})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["review_marker"] == review_marker_json()

    def test_byte_deterministic(self, tmp_path: Path) -> None:
        payload = {"a": 1, "b": [2, 3]}
        path1 = tmp_path / "r1.json"
        path2 = tmp_path / "r2.json"
        write_json(path1, payload)
        write_json(path2, payload)
        assert path1.read_bytes() == path2.read_bytes()


class TestWriteCsvMarker:
    """write_csv always prepends the comment marker."""

    def test_marker_present(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        write_csv(path, [{"col": "val"}])
        assert path.read_text(encoding="utf-8").startswith("# AI-GENERATED NEEDS-REVIEW")

    def test_byte_deterministic(self, tmp_path: Path) -> None:
        rows = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        path1 = tmp_path / "d1.csv"
        path2 = tmp_path / "d2.csv"
        write_csv(path1, rows)
        write_csv(path2, rows)
        assert path1.read_bytes() == path2.read_bytes()


class TestWriteTextMarker:
    """write_text prepends AI-GENERATED marker when issue_ref is given."""

    def test_marker_with_issue_ref(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        write_text(path, "# Report\n", issue_ref="robot_sf#4921", marker_date="2026-07-14")
        content = path.read_text(encoding="utf-8")
        assert "<!-- AI-GENERATED (robot_sf#4921, 2026-07-14) - NEEDS-REVIEW -->" in content

    def test_byte_deterministic_with_date(self, tmp_path: Path) -> None:
        content = "# Report\n"
        path1 = tmp_path / "r1.md"
        path2 = tmp_path / "r2.md"
        write_text(path1, content, issue_ref="robot_sf#4921", marker_date="2026-07-14")
        write_text(path2, content, issue_ref="robot_sf#4921", marker_date="2026-07-14")
        assert path1.read_bytes() == path2.read_bytes()

    def test_different_date_produces_different_bytes(self, tmp_path: Path) -> None:
        content = "# Report\n"
        path1 = tmp_path / "r1.md"
        path2 = tmp_path / "r2.md"
        write_text(path1, content, issue_ref="robot_sf#4921", marker_date="2026-07-14")
        write_text(path2, content, issue_ref="robot_sf#4921", marker_date="2026-07-15")
        assert path1.read_bytes() != path2.read_bytes()


class TestWriteSha256sumsMarker:
    """write_sha256sums prepends the comment marker."""

    def test_marker_present(self, tmp_path: Path) -> None:
        (tmp_path / "data.txt").write_text("hello\n", encoding="utf-8")
        write_sha256sums(tmp_path)
        sha_file = tmp_path / "SHA256SUMS"
        content = sha_file.read_text(encoding="utf-8")
        assert content.startswith("# AI-GENERATED NEEDS-REVIEW")

    def test_byte_deterministic(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        out1.mkdir()
        out2.mkdir()
        for src in ("a.txt", "b.txt"):
            (out1 / src).write_text((tmp_path / src).read_text(encoding="utf-8"), encoding="utf-8")
            (out2 / src).write_text((tmp_path / src).read_text(encoding="utf-8"), encoding="utf-8")
        write_sha256sums(out1)
        write_sha256sums(out2)
        assert (out1 / "SHA256SUMS").read_bytes() == (out2 / "SHA256SUMS").read_bytes()
