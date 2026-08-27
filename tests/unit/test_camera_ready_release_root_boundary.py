"""Core-lane coverage for frozen release-checkout path containment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.camera_ready._run_state import _resolve_path
from robot_sf.benchmark.release_protocol import _scenario_matrix_include_paths

if TYPE_CHECKING:
    from pathlib import Path


def test_explicit_release_root_rejects_absolute_sidecar_escape(tmp_path: Path) -> None:
    """An absolute campaign sidecar cannot escape the frozen release checkout."""
    repository_root = tmp_path / "frozen-release"
    repository_root.mkdir()
    escaped = tmp_path / "outside.yaml"
    escaped.write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes repository_root"):
        _resolve_path(
            str(escaped),
            base_dir=repository_root / "configs",
            repository_root=repository_root,
        )


def test_explicit_release_root_rejects_existing_relative_sidecar_escape(tmp_path: Path) -> None:
    """An existing tooling-relative sidecar cannot override the frozen checkout root."""
    repository_root = tmp_path / "frozen-release"
    repository_root.mkdir()
    tooling_config = tmp_path / "tooling" / "configs"
    tooling_config.mkdir(parents=True)
    (tooling_config / "sidecar.yaml").write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes repository_root"):
        _resolve_path(
            "sidecar.yaml",
            base_dir=tooling_config,
            repository_root=repository_root,
        )


def test_explicit_release_root_rejects_scenario_matrix_escape(tmp_path: Path) -> None:
    """Scenario traversal must use the frozen checkout rather than the tooling checkout."""
    repository_root = tmp_path / "frozen-release"
    repository_root.mkdir()
    escaped = tmp_path / "outside.yaml"
    escaped.write_text("scenarios: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scenario matrix include escapes repository"):
        _scenario_matrix_include_paths(escaped, repository_root=repository_root)
