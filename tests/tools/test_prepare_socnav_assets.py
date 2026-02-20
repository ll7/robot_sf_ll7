"""Tests for SocNav third-party asset preparation and validation helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools.prepare_socnav_assets import copy_available_assets, evaluate_assets

if TYPE_CHECKING:
    from pathlib import Path


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def test_copy_available_assets_stages_from_source_root(tmp_path: Path) -> None:
    """Copy helper should transfer discovered source directories into SocNav root."""
    source_root = tmp_path / "source"
    socnav_root = tmp_path / "socnavbench"

    _mkdir(source_root / "wayptnav_data")
    _mkdir(
        source_root
        / "sd3dis"
        / "sd3dis-public"
        / "stanford_building_parser_dataset"
        / "traversibles"
    )
    _mkdir(source_root / "surreal" / "code" / "human_meshes")
    _mkdir(source_root / "surreal" / "code" / "human_textures")

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
