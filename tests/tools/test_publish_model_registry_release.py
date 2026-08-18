"""Tests for model-registry GitHub release publication helper."""

from __future__ import annotations

import json
import os
import tarfile
from typing import TYPE_CHECKING

import pytest
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


def _registry_text(model_path: Path, *, include_licensing: bool = True) -> str:
    """Return a small registry containing one W&B-backed model."""
    license_path = model_path.parent / "LICENSE"
    model_card_path = model_path.parent / "MODEL_CARD.md"
    _write(license_path, "MIT License\nCopyright (c) fixture\n")
    _write(model_card_path, "# Fixture model\n\nFor tests only.\n")
    licensing = f"""
    licensing:
      license_spdx: MIT
      copyright: Fixture copyright holder
      redistribution_basis: Fixture permits redistribution
      source_repository: https://example.invalid/fixture
      source_revision: 0123456789abcdef0123456789abcdef01234567
      source_archive_sha256: 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
      weights_origin: trained-here
      training_code_license: GPL-3.0-only
      training_data:
        provenance: synthetic fixture
        license_spdx: CC0-1.0
      license_file: {license_path}
      model_card_file: {model_card_path}
      included_notices: []
"""
    if not include_licensing:
        licensing = ""
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
{licensing}
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
    legal_bundle = tmp_path / "staging" / "demo_model-legal.tar.gz"
    with tarfile.open(legal_bundle, "r:gz") as archive:
        assert set(archive.getnames()) == {"LICENSE", "MODEL_CARD.md", "PROVENANCE.json"}
    manifest = json.loads((tmp_path / "staging" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "robot-sf-model-registry-release.v1"
    assert manifest["models"][0]["release_asset_url"].endswith(
        "/releases/download/artifact/models-test/demo_model-model.zip"
    )
    assert "demo_model-model.zip" in (tmp_path / "staging" / "SHA256SUMS").read_text(
        encoding="utf-8"
    )
    assert plan["upload_command"][0:3] == ["gh", "release", "upload"]


def test_legal_bundle_is_byte_reproducible(tmp_path: Path, monkeypatch) -> None:
    """Repeated staging of the same rights evidence must preserve its checksum."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(model_path))
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    args = [
        "--registry-path",
        str(registry_path),
        "--tag",
        "artifact/models-test",
        "--staging-dir",
        str(tmp_path / "staging-one"),
    ]
    publish_model_registry_release.main(args)
    first = (tmp_path / "staging-one" / "demo_model-legal.tar.gz").read_bytes()
    args[-1] = str(tmp_path / "staging-two")
    publish_model_registry_release.main(args)
    second = (tmp_path / "staging-two" / "demo_model-legal.tar.gz").read_bytes()

    assert first == second


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

    with pytest.raises(ValueError, match="selected artifacts are unavailable"):
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
    capsys.readouterr()


def test_publish_model_registry_release_blocks_existing_public_rows_without_legal_bundle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Existing public rows must be remediated before any new release can be staged."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    data = yaml.safe_load(_registry_text(model_path))
    data["models"][0]["github_release"] = {"asset_name": "old-model.zip"}
    data["models"][0]["public_artifact_source"] = "github_release"
    _write(registry_path, yaml.safe_dump(data, sort_keys=False))
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="existing public rows"):
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


def test_publish_model_registry_release_blocks_missing_weight_rights(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A local checkpoint cannot be published without explicit rights evidence."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(model_path, include_licensing=False))
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
            ]
        )
    except ValueError as exc:
        assert "licensing mapping is required" in str(exc)
    else:
        raise AssertionError("missing weight rights evidence was not rejected")


def test_model_licensing_preflight_is_read_only_and_hashes_dynamic_files(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """The licensing preflight proves registry legal files without publication side effects."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(model_path))
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)
    before = registry_path.read_bytes()

    exit_code = publish_model_registry_release.main(
        ["--registry-path", str(registry_path), "--validate-licensing"]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "robot-sf-model-licensing-preflight.v1"
    assert report["status"] == "passed"
    assert report["read_only"] is True
    assert report["uploads_performed"] is False
    assert report["registry_writes_performed"] is False
    row = report["rows"][0]
    assert row["model_id"] == "demo_model"
    assert {item["path"] for item in row["files"]} == {
        "output/model_cache/demo_model/LICENSE",
        "output/model_cache/demo_model/MODEL_CARD.md",
    }
    assert not (tmp_path / "output" / "model_registry_release").exists()
    assert registry_path.read_bytes() == before


def test_model_licensing_preflight_blocks_missing_rights_without_writing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """A W&B-backed row without licensing evidence fails closed in read-only mode."""
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(
        registry_path, _registry_text(tmp_path / "missing" / "model.zip", include_licensing=False)
    )
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)
    before = registry_path.read_bytes()

    exit_code = publish_model_registry_release.main(
        ["--registry-path", str(registry_path), "--validate-licensing"]
    )

    assert exit_code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "blocked"
    assert any("licensing mapping is required" in issue for issue in report["issues"])
    assert registry_path.read_bytes() == before
    assert not (tmp_path / "output" / "model_registry_release").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_model_licensing_preflight_rejects_symlinked_legal_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Legal evidence cannot be supplied through a symlinked repository path."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(model_path))
    license_path = model_path.parent / "LICENSE"
    outside_license = tmp_path / "outside-LICENSE"
    _write(outside_license, "MIT License\n")
    license_path.unlink()
    license_path.symlink_to(outside_license)
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    exit_code = publish_model_registry_release.main(
        ["--registry-path", str(registry_path), "--validate-licensing"]
    )

    assert exit_code == 2
    report = json.loads(capsys.readouterr().out)
    assert any("must not traverse symlinks" in issue for issue in report["issues"])


def test_undeclared_licensing_is_reported_but_can_be_allowed_in_the_pr_gate(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """A row with no licensing mapping is a known rights gap, strict only outside the PR gate."""
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(
        registry_path, _registry_text(tmp_path / "missing" / "model.zip", include_licensing=False)
    )
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    strict_code = publish_model_registry_release.main(
        ["--registry-path", str(registry_path), "--validate-licensing"]
    )
    strict_report = json.loads(capsys.readouterr().out)
    allowed_code = publish_model_registry_release.main(
        [
            "--registry-path",
            str(registry_path),
            "--validate-licensing",
            "--allow-undeclared-licensing",
        ]
    )
    allowed_report = json.loads(capsys.readouterr().out)

    assert strict_code == 2
    assert allowed_code == 0
    assert allowed_report["status"] == "blocked"
    assert strict_report["undeclared_licensing_issues"]
    assert allowed_report["undeclared_licensing_issues"]
    assert allowed_report["invalid_licensing_evidence_issues"] == []


def test_invalid_declared_licensing_evidence_still_fails_the_pr_gate(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Declared licensing whose legal file is missing cannot be waived by the PR-gate flag."""
    model_path = tmp_path / "output" / "model_cache" / "demo_model" / "model.zip"
    _write(model_path, b"checkpoint")
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(model_path))
    (model_path.parent / "LICENSE").unlink()
    monkeypatch.setattr(publish_model_registry_release, "get_repository_root", lambda: tmp_path)

    exit_code = publish_model_registry_release.main(
        [
            "--registry-path",
            str(registry_path),
            "--validate-licensing",
            "--allow-undeclared-licensing",
        ]
    )

    assert exit_code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["invalid_licensing_evidence_issues"]
    assert any("does not exist" in issue for issue in report["invalid_licensing_evidence_issues"])


def test_allow_undeclared_licensing_requires_the_preflight_mode(tmp_path: Path) -> None:
    """The PR-gate waiver cannot leak into the publication path."""
    registry_path = tmp_path / "model" / "registry.yaml"
    _write(registry_path, _registry_text(tmp_path / "missing" / "model.zip"))

    with pytest.raises(SystemExit) as exc:
        publish_model_registry_release.main(
            [
                "--registry-path",
                str(registry_path),
                "--tag",
                "artifact/models-fixture",
                "--allow-undeclared-licensing",
            ]
        )

    assert exc.value.code == 2
