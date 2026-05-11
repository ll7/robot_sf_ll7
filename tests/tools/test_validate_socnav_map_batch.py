"""Tests for the SocNavBench map-import batch validator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.tools.validate_socnav_map_batch import validate_batch

if TYPE_CHECKING:
    from pathlib import Path


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
                    }
                ],
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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
    (root / "sd3dis" / "stanford_building_parser_dataset" / "mesh" / "ETH").mkdir(parents=True)
    traversible = (
        root / "sd3dis" / "stanford_building_parser_dataset" / "traversibles" / "ETH" / "data.pkl"
    )
    traversible.parent.mkdir(parents=True)
    traversible.write_bytes(b"fixture")

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
