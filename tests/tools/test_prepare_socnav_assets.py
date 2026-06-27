"""Tests for SocNav third-party asset preparation and validation helper."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from scripts.tools import manage_external_data, prepare_socnav_assets
from scripts.tools.prepare_socnav_assets import copy_available_assets, evaluate_assets

if TYPE_CHECKING:
    from pathlib import Path


def _mkdir(path: Path) -> None:
    """Create an asset fixture directory tree."""
    path.mkdir(parents=True, exist_ok=True)


def _mkasset(path: Path) -> None:
    """Create an asset fixture directory backed by a real file (restored-real)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "data.bin").write_bytes(b"asset")


def _schematic_layout(socnav_root: Path) -> None:
    """Stage all schematic-required asset directories with real files."""
    _mkasset(socnav_root / "wayptnav_data")
    _mkasset(socnav_root / "sd3dis" / "stanford_building_parser_dataset")
    _mkasset(socnav_root / "sd3dis" / "stanford_building_parser_dataset" / "traversibles")


def test_evaluate_assets_reports_missing_required_for_schematic(tmp_path: Path) -> None:
    """Schematic profile should require wayptnav + sbpd + traversibles."""
    socnav_root = tmp_path / "socnavbench"
    _mkdir(socnav_root)

    report = evaluate_assets(socnav_root, render_mode="schematic")
    assert report["ok"] is False
    missing = set(report["missing_required"])
    assert "wayptnav_data" in missing
    assert "sbpd_dataset" in missing
    assert "sbpd_traversibles" in missing
    assert "surreal_meshes" not in missing


def test_evaluate_assets_available_state_for_restored_real_assets(tmp_path: Path) -> None:
    """Schematic-required assets backed by real files report `available` and pass."""
    socnav_root = tmp_path / "socnavbench"
    _schematic_layout(socnav_root)

    report = evaluate_assets(socnav_root, render_mode="schematic")
    assert report["ok"] is True
    assert report["missing_required"] == []
    assert report["placeholder_required"] == []
    by_key = {entry["key"]: entry for entry in report["assets"]}
    assert by_key["wayptnav_data"]["status"] == "available"
    assert by_key["wayptnav_data"]["has_real_files"] is True


def test_evaluate_assets_placeholder_shell_is_not_counted_as_restored(tmp_path: Path) -> None:
    """An empty required directory is a placeholder, never restored evidence (fail-closed)."""
    socnav_root = tmp_path / "socnavbench"
    # wayptnav_data exists but is an empty shell; the other required assets are real.
    _mkdir(socnav_root / "wayptnav_data")
    _mkasset(socnav_root / "sd3dis" / "stanford_building_parser_dataset")
    _mkasset(socnav_root / "sd3dis" / "stanford_building_parser_dataset" / "traversibles")

    report = evaluate_assets(socnav_root, render_mode="schematic")
    assert report["ok"] is False
    assert "wayptnav_data" in report["missing_required"]
    assert report["placeholder_required"] == ["wayptnav_data"]
    by_key = {entry["key"]: entry for entry in report["assets"]}
    assert by_key["wayptnav_data"]["status"] == "placeholder"
    assert by_key["wayptnav_data"]["exists"] is True
    assert by_key["wayptnav_data"]["has_real_files"] is False


def test_evaluate_assets_excludes_surreal_assets_in_schematic(tmp_path: Path) -> None:
    """SURREAL assets are excluded (optional) in schematic mode and never block readiness."""
    socnav_root = tmp_path / "socnavbench"
    _schematic_layout(socnav_root)

    report = evaluate_assets(socnav_root, render_mode="schematic")
    by_key = {entry["key"]: entry for entry in report["assets"]}
    assert by_key["surreal_meshes"]["status"] == "excluded"
    assert by_key["surreal_meshes"]["required"] is False
    assert "surreal_meshes" not in report["missing_required"]
    # Excluded assets must not flip overall readiness even when absent.
    assert report["ok"] is True


def test_default_socnav_root_honors_external_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-flag SocNav asset validation should use the shared external data root."""
    shared_root = tmp_path / "robot_sf_external_data"
    monkeypatch.setenv(manage_external_data.EXTERNAL_DATA_ROOT_ENV, str(shared_root))

    reloaded = importlib.reload(prepare_socnav_assets)

    assert reloaded.DEFAULT_SOCNAV_ROOT == shared_root.resolve() / "socnavbench"
    monkeypatch.delenv(manage_external_data.EXTERNAL_DATA_ROOT_ENV)
    importlib.reload(prepare_socnav_assets)


def test_copy_available_assets_stages_from_source_root(tmp_path: Path) -> None:
    """Copy helper should transfer discovered source directories into SocNav root."""
    source_root = tmp_path / "source"
    socnav_root = tmp_path / "socnavbench"

    _mkasset(source_root / "wayptnav_data")
    _mkasset(source_root / "sd3dis" / "sd3dis-public" / "stanford_building_parser_dataset")
    _mkasset(
        source_root
        / "sd3dis"
        / "sd3dis-public"
        / "stanford_building_parser_dataset"
        / "traversibles"
    )
    _mkasset(source_root / "surreal" / "code" / "human_meshes")
    _mkasset(source_root / "surreal" / "code" / "human_textures")

    actions = copy_available_assets(
        socnav_root=socnav_root,
        source_root=source_root,
        render_mode="full-render",
        overwrite=False,
    )

    assert actions
    assert (socnav_root / "wayptnav_data").is_dir()
    assert (socnav_root / "sd3dis" / "stanford_building_parser_dataset").is_dir()
    assert (socnav_root / "sd3dis" / "stanford_building_parser_dataset" / "traversibles").is_dir()
    assert (socnav_root / "surreal" / "code" / "human_meshes").is_dir()
    assert (socnav_root / "surreal" / "code" / "human_textures").is_dir()

    report = evaluate_assets(socnav_root, render_mode="full-render")
    assert report["ok"] is True


def test_copy_available_assets_skips_self_copy_on_overwrite(tmp_path: Path) -> None:
    """Overwrite mode must not delete assets when source and destination are identical."""
    socnav_root = tmp_path / "socnavbench"
    _mkdir(socnav_root / "wayptnav_data")
    marker = socnav_root / "wayptnav_data" / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    actions = copy_available_assets(
        socnav_root=socnav_root,
        source_root=socnav_root,
        render_mode="schematic",
        overwrite=True,
    )

    assert "skip self-copy: wayptnav_data" in actions
    assert marker.exists()
