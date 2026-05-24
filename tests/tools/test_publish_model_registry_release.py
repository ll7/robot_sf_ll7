"""Tests for the model-registry release publication helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from scripts.tools.publish_model_registry_release import apply_manifest, inventory

if TYPE_CHECKING:
    from pathlib import Path


def test_apply_manifest_adds_public_release_pointer(tmp_path: Path) -> None:
    """Manifest application should add verified GitHub release pointers."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        (
            "version: 1\nmodels:\n"
            "  - model_id: public_model\n"
            "    local_path: output/model_cache/public_model/model.zip\n"
            "    wandb_file: model.zip\n"
            "  - model_id: local_model\n"
            "    local_path: model/local/model.zip\n"
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "robot-sf-model-registry-release.v1",
                "repo": "ll7/robot_sf_ll7",
                "tag": "artifact/models-2026-05-registry-v1",
                "models": [
                    {
                        "model_id": "public_model",
                        "asset_name": "public_model-model.zip",
                        "metadata_asset": "public_model-metadata.json",
                        "sha256": "abc123",
                        "size_bytes": 12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sha256s_path = tmp_path / "SHA256SUMS"
    sha256s_path.write_text("abc123  public_model-model.zip\n", encoding="utf-8")

    result = apply_manifest(
        registry_path=registry_path,
        manifest_path=manifest_path,
        sha256s_path=sha256s_path,
        output_path=None,
        write=True,
    )

    assert result["updated"] == ["public_model"]
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert data["models"][0]["github_release"] == {
        "repository": "ll7/robot_sf_ll7",
        "tag": "artifact/models-2026-05-registry-v1",
        "asset_name": "public_model-model.zip",
        "file_name": "model.zip",
        "sha256": "abc123",
        "size_bytes": 12,
        "metadata_asset_name": "public_model-metadata.json",
    }
    assert "github_release" not in data["models"][1]


def test_inventory_reports_publication_state(tmp_path: Path) -> None:
    """Inventory should identify public, W&B-backed, and local-only registry rows."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        (
            "version: 1\nmodels:\n"
            "  - model_id: public_model\n"
            "    github_release:\n"
            "      repository: ll7/robot_sf_ll7\n"
            "    wandb_run_path: ll7/robot_sf/run\n"
            "  - model_id: local_model\n"
            "    local_only: true\n"
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"repo": "ll7/robot_sf_ll7", "tag": "v1", "models": [{"model_id": "public_model"}]}
        ),
        encoding="utf-8",
    )

    result = inventory(registry_path=registry_path, manifest_path=manifest_path)

    assert result["models"] == [
        {
            "model_id": "public_model",
            "has_github_release": True,
            "has_wandb_pointer": True,
            "local_only": False,
            "in_manifest": True,
        },
        {
            "model_id": "local_model",
            "has_github_release": False,
            "has_wandb_pointer": False,
            "local_only": True,
            "in_manifest": False,
        },
    ]
