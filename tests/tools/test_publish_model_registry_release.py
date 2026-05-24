"""Tests for model-registry GitHub release publication helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from scripts.tools import publish_model_registry_release

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, payload: bytes | str) -> None:
    """Write a test fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def _registry_text(model_path: Path) -> str:
    """Return a small registry containing one W&B-backed model."""
    return (
        f"""
version: 1
models:
  - model_id: demo_model
    display_name: Demo model
    local_path: {model_path.as_posix()}
    config_path: configs/training/demo.yaml
    commit: abc123
    wandb_run_id: run123
    wandb_run_path: ll7/robot_sf/run123
    wandb_entity: ll7
    wandb_project: robot_sf
    wandb_file: model.zip
    tags:
      - demo
  - model_id: local_only_model
    local_path: local/model.zip
""".strip()
        + "\n"
    )


def test_publish_model_registry_release_stages_assets_and_manifest(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Dry-run publication should stage model, metadata, manifest, and checksums."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(model_path))
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    exit_code = publish_model_registry_release.main(
        [
            "--registry-path",
            str(registry_path),
            "--tag",
            "artifact/models-test",
            "--staging-dir",
            str(tmp_path / "staging"),
        ]
    )

    assert exit_code == 0
    plan = json.loads(capsys.readouterr().out)
    assert [item["model_id"] for item in plan["published"]] == ["demo_model"]
    asset_name = "demo_model-model.zip"
    assert (tmp_path / "staging" / asset_name).read_bytes() == b"checkpoint"
    assert (tmp_path / "staging" / "demo_model-metadata.json").exists()
    manifest = json.loads((tmp_path / "staging" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "robot-sf-model-registry-release.v1"
    assert manifest["models"][0]["release_asset_url"].endswith(
        "/releases/download/artifact/models-test/demo_model-model.zip"
    )
    assert "demo_model-model.zip" in (tmp_path / "staging" / "SHA256SUMS").read_text(
        encoding="utf-8"
    )
    assert plan["upload_command"][0:3] == ["gh", "release", "upload"]


def test_publish_model_registry_release_updates_registry_output(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Update mode should write github_release pointers without removing W&B provenance."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    output_path = tmp_path / "updated_registry.yaml"
    _write(registry_path, _registry_text(model_path))
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    publish_model_registry_release.main(
        [
            "--registry-path",
            str(registry_path),
            "--tag",
            "artifact/models-test",
            "--staging-dir",
            str(tmp_path / "staging"),
            "--update-registry",
            "--allow-registry-update-without-upload",
            "--registry-output",
            str(output_path),
        ]
    )
    capsys.readouterr()

    updated = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    entry = updated["models"][0]
    assert entry["model_id"] == "demo_model"
    assert entry["public_artifact_source"] == "github_release"
    assert entry["github_release"]["asset_name"] == "demo_model-model.zip"
    assert entry["github_release"]["sha256"]
    assert entry["wandb_run_path"] == "ll7/robot_sf/run123"


def test_publish_model_registry_release_reports_missing_local_without_download(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Missing local artifacts should be reported instead of silently skipped."""
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(tmp_path / "missing" / "model.zip"))
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    publish_model_registry_release.main(
        [
            "--registry-path",
            str(registry_path),
            "--tag",
            "artifact/models-test",
            "--staging-dir",
            str(tmp_path / "staging"),
        ]
    )
    plan = json.loads(capsys.readouterr().out)

    assert plan["published"] == []
    assert plan["skipped"] == [
        {
            "model_id": "demo_model",
            "reason": "missing local_path; rerun with --download-missing to hydrate from registry",
        }
    ]


def test_publish_model_registry_release_requires_upload_before_registry_update(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Registry updates should not write unpublished release pointers by default."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(model_path))
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    try:
        publish_model_registry_release.main(
            [
                "--registry-path",
                str(registry_path),
                "--tag",
                "artifact/models-test",
                "--staging-dir",
                str(tmp_path / "staging"),
                "--update-registry",
                "--registry-output",
                str(tmp_path / "updated_registry.yaml"),
            ]
        )
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("--update-registry without upload did not fail closed")
