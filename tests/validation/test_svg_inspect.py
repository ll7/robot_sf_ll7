"""Tests for scripts.validation.svg_inspect helpers."""

from pathlib import Path

import pytest

from scripts.validation.svg_inspect import _resolve_json_output_path


def test_resolve_json_output_path_allows_paths_within_base(tmp_path: Path) -> None:
    """Relative JSON output paths should resolve under the trusted base directory."""
    resolved = _resolve_json_output_path(Path("output/reports/inspection.json"), base_dir=tmp_path)

    assert resolved == (tmp_path / "output/reports/inspection.json").resolve()


def test_resolve_json_output_path_rejects_escape_from_base(tmp_path: Path) -> None:
    """Traversal outside the trusted base directory must be rejected."""
    with pytest.raises(ValueError, match="--json path must stay inside working directory"):
        _resolve_json_output_path(Path("../outside.json"), base_dir=tmp_path)
