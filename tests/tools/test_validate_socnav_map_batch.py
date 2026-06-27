"""Tests for the SocNavBench map-import batch validator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.tools.validate_socnav_map_batch import (
    BLOCKED,
    READY,
    conversion_readiness,
    validate_batch,
)

if TYPE_CHECKING:
    from pathlib import Path

PLANNED_MAP_SVG = "maps/svg_maps/socnavbench/socnavbench_eth.svg"


def _write_manifest(path: Path) -> None:
    """Write a minimal SocNavBench import manifest fixture."""
    payload = {
        "version": 1,
        "batches": [
            {
                "batch_id": "eth_first",
                "status": "blocked_pending_source_assets",
                "maps": [
                    {
                        "map_id": "socnavbench_eth",
                        "upstream_map_name": "ETH",
                        "source_assets": [
                            {
                                "key": "eth_mesh_dir",
                                "relative_path": (
                                    "sd3dis/stanford_building_parser_dataset/mesh/ETH"
                                ),
                                "kind": "directory",
                                "required_for_conversion": True,
                            },
                            {
                                "key": "eth_traversible_pickle",
                                "relative_path": (
                                    "sd3dis/stanford_building_parser_dataset/"
                                    "traversibles/ETH/data.pkl"
                                ),
                                "kind": "file",
                                "required_for_conversion": True,
                            },
                        ],
                        "planned_outputs": {"map_svg": PLANNED_MAP_SVG},
                    }
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _stage_eth_assets(root: Path) -> None:
    """Stage the exact ETH mesh dir and traversible pickle under ``root``."""
    base = root / "sd3dis" / "stanford_building_parser_dataset"
    (base / "mesh" / "ETH").mkdir(parents=True)
    traversible = base / "traversibles" / "ETH" / "data.pkl"
    traversible.parent.mkdir(parents=True)
    traversible.write_bytes(b"fixture")


def test_validate_batch_reports_missing_required_assets(tmp_path: Path) -> None:
    """Missing ETH source assets should fail closed before conversion."""
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)

    report = validate_batch(
        manifest_path=manifest,
        socnav_root=tmp_path / "socnavbench",
        batch_id="eth_first",
    )

    assert report["ok"] is False
    assert {item["asset_key"] for item in report["missing_required"]} == {
        "eth_mesh_dir",
        "eth_traversible_pickle",
    }


def test_validate_batch_accepts_staged_eth_assets(tmp_path: Path) -> None:
    """Validator should pass once the exact ETH source assets are staged."""
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)
    root = tmp_path / "socnavbench"
    _stage_eth_assets(root)

    report = validate_batch(
        manifest_path=manifest,
        socnav_root=root,
        batch_id="eth_first",
    )

    assert report["ok"] is True
    assert report["missing_required"] == []
    assert report["maps"][0]["map_id"] == "socnavbench_eth"


def test_validate_batch_rejects_empty_relative_path(tmp_path: Path) -> None:
    """Validator should fail fast when a source asset omits its relative path."""
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["batches"][0]["maps"][0]["source_assets"][0]["relative_path"] = ""
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty relative path"):
        validate_batch(
            manifest_path=manifest,
            socnav_root=tmp_path / "socnavbench",
            batch_id="eth_first",
        )


def test_validate_batch_rejects_parent_relative_path(tmp_path: Path) -> None:
    """Validator should reject parent-relative source asset paths before resolution."""
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["batches"][0]["maps"][0]["source_assets"][0]["relative_path"] = "../secret"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="relative path within the root"):
        validate_batch(
            manifest_path=manifest,
            socnav_root=tmp_path / "socnavbench",
            batch_id="eth_first",
        )


def test_conversion_readiness_blocks_when_traversible_missing(tmp_path: Path) -> None:
    """Preflight must fail closed and name the missing traversible while blocked."""
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)

    report = conversion_readiness(
        manifest_path=manifest,
        socnav_root=tmp_path / "socnavbench",
        batch_id="eth_first",
        repo_root=tmp_path / "repo",
    )

    assert report["conversion_ready"] is False
    assert report["status"] == BLOCKED
    missing_keys = {item["asset_key"] for item in report["missing_required"]}
    assert "eth_traversible_pickle" in missing_keys
    # The next-action hint must point at staging the missing traversible path.
    assert "traversibles/ETH/data.pkl" in report["next_action"]
    # Nothing was staged, so there is no placeholder masquerading as a conversion.
    assert report["placeholder_risk"] == []


def test_conversion_readiness_ready_when_assets_staged(tmp_path: Path) -> None:
    """Preflight should clear conversion only once every required asset is staged."""
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)
    root = tmp_path / "socnavbench"
    _stage_eth_assets(root)

    report = conversion_readiness(
        manifest_path=manifest,
        socnav_root=root,
        batch_id="eth_first",
        repo_root=tmp_path / "repo",
    )

    assert report["conversion_ready"] is True
    assert report["status"] == READY
    assert report["missing_required"] == []
    assert report["placeholder_risk"] == []


def test_conversion_readiness_flags_placeholder_when_blocked(tmp_path: Path) -> None:
    """A pre-existing planned SVG while blocked is a provenance/placeholder risk."""
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)
    repo_root = tmp_path / "repo"
    placeholder = repo_root / PLANNED_MAP_SVG
    placeholder.parent.mkdir(parents=True)
    placeholder.write_text("<svg></svg>", encoding="utf-8")

    report = conversion_readiness(
        manifest_path=manifest,
        socnav_root=tmp_path / "socnavbench",
        batch_id="eth_first",
        repo_root=repo_root,
    )

    assert report["conversion_ready"] is False
    assert [risk["relative_path"] for risk in report["placeholder_risk"]] == [PLANNED_MAP_SVG]
    assert report["placeholder_risk"][0]["output_key"] == "map_svg"


def test_validate_batch_rejects_absolute_relative_path(tmp_path: Path) -> None:
    """Validator should reject absolute source asset paths before resolution."""
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["batches"][0]["maps"][0]["source_assets"][0]["relative_path"] = "/tmp/eth"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="relative path within the root"):
        validate_batch(
            manifest_path=manifest,
            socnav_root=tmp_path / "socnavbench",
            batch_id="eth_first",
        )
