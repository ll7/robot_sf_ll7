"""Tests for the ``robot-sf models`` list/verify/download UX (issue #5797)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf import cli_models
from robot_sf.cli import main

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from pathlib import Path


def _write_registry(tmp_path: Path, models: list[dict]) -> Path:
    """Write a minimal model registry.yaml and return its path."""
    import yaml

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump({"version": 1, "models": models}), encoding="utf-8")
    return registry_path


def test_models_list_reports_registered_ids_and_presence(tmp_path: Path) -> None:
    """``models list`` should summarize every registry row with local presence."""
    artifact = tmp_path / "good_model.zip"
    artifact.write_bytes(b"good-bytes")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "good_model",
                "display_name": "A good model",
                "tags": ["ppo", "promoted"],
                "local_path": str(artifact),
                "benchmark_promotion": {"claim_boundary": "benchmark_promoted"},
            },
            {
                "model_id": "absent_model",
                "display_name": "An absent model",
                "tags": [],
                "local_path": str(tmp_path / "missing.zip"),
            },
        ],
    )

    rows = cli_models.list_models(registry_path=registry_path)

    assert [row["model_id"] for row in rows] == ["good_model", "absent_model"]
    good, absent = rows
    assert good["present_locally"] is True
    assert good["claim_boundary"] == "benchmark_promoted"
    assert good["tags"] == ["ppo", "promoted"]
    assert absent["present_locally"] is False
    assert absent["claim_boundary"] is None


def test_models_verify_reports_passing_checksum(tmp_path: Path) -> None:
    """``models verify`` should PASS for a local file matching a pinned SHA256."""
    import hashlib

    content = b"verified-model-bytes"
    digest = hashlib.sha256(content).hexdigest()
    artifact = tmp_path / "model.zip"
    artifact.write_bytes(content)
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "pinned_model",
                "display_name": "Pinned model",
                "local_path": str(artifact),
                "github_release": {"sha256": digest, "asset_name": "model.zip"},
            }
        ],
    )

    report = cli_models.verify_models(registry_path=registry_path)

    assert report["ok"] is True
    assert report["pinned_checksums"] == 1
    assert report["passed"] == 1
    result = report["results"][0]
    assert result["status"] == "ok"
    assert result["pinned"] is True
    assert result["observed_sha256"] == digest


def test_models_verify_reports_corrupted_checksum_failure(tmp_path: Path) -> None:
    """``models verify`` should FAIL when the local SHA256 does not match the pin."""
    artifact = tmp_path / "model.zip"
    artifact.write_bytes(b"corrupted-bytes")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "corrupt_model",
                "display_name": "Corrupt model",
                "local_path": str(artifact),
                "github_release": {
                    "sha256": "0" * 64,
                    "asset_name": "model.zip",
                },
            }
        ],
    )

    report = cli_models.verify_models(registry_path=registry_path)

    assert report["ok"] is False
    result = report["results"][0]
    assert result["status"] == "mismatch"
    assert result["pinned"] is True
    assert result["expected_sha256"] == "0" * 64
    assert result["observed_sha256"] != "0" * 64


def test_models_verify_missing_pinned_artifact_is_failure(tmp_path: Path) -> None:
    """A pinned-but-absent artifact should be reported missing (not pass)."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "absent_pinned",
                "display_name": "Absent pinned model",
                "local_path": str(tmp_path / "nope.zip"),
                "github_release": {"sha256": "a" * 64, "asset_name": "model.zip"},
            }
        ],
    )

    report = cli_models.verify_models(registry_path=registry_path)

    assert report["ok"] is False
    assert report["results"][0]["status"] == "missing"


def test_models_cli_verify_friendly_exit_code_on_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should exit non-zero and print a human checksum-mismatch report."""
    artifact = tmp_path / "model.zip"
    artifact.write_bytes(b"corrupted-bytes")
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "corrupt_model",
                "display_name": "Corrupt model",
                "local_path": str(artifact),
                "github_release": {"sha256": "f" * 64, "asset_name": "model.zip"},
            }
        ],
    )

    rc = main(["models", "verify", "--registry", str(registry_path)])

    assert rc == 2
    out = capsys.readouterr().out
    assert "corrupt_model" in out
    assert "mismatch" in out


def test_models_cli_list_friendly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI ``models list`` friendly path should list registered ids."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "demo_model",
                "display_name": "Demo model",
                "tags": ["ppo"],
                "local_path": str(tmp_path / "demo.zip"),
            }
        ],
    )

    rc = main(["models", "list", "--registry", str(registry_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "demo_model" in out
    assert "Demo model" in out
