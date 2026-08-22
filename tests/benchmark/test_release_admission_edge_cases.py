"""Behavior-focused edge tests for the fail-closed release admission helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark import checkpoint_staging_receipt as staging
from robot_sf.benchmark.identity.hash_utils import sha256_file

if TYPE_CHECKING:
    from pathlib import Path


def _receipt_fixture(tmp_path: Path) -> tuple[SimpleNamespace, Path, Path, Path, dict]:
    """Build a minimal valid receipt fixture for edge-case tests."""
    config = tmp_path / "campaign.yaml"
    config.write_text("name: release\n", encoding="utf-8")
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"checkpoint")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": [
                    {
                        "model_id": "ppo_release",
                        "github_release": {"sha256": sha256_file(checkpoint)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reference = SimpleNamespace(
        planner_key="ppo",
        algo="ppo",
        kind="model_id",
        value="ppo_release",
        implicit=False,
    )
    cfg = SimpleNamespace(references=[reference])
    payload = {
        "schema_version": staging.CHECKPOINT_STAGING_RECEIPT_SCHEMA,
        "status": "ok",
        "mode": "enforced_staged",
        "stage": True,
        "submit_safe": True,
        "generated_at_utc": "2026-08-21T12:00:00Z",
        "campaign_config_sha256": sha256_file(config),
        "checkpoint_registry_sha256": sha256_file(registry),
        "arms": [
            {
                "planner_key": "ppo",
                "algo": "ppo",
                "kind": "model_id",
                "value": "ppo_release",
                "implicit": False,
                "status": "staged",
                "resolved_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "hash_source": "computed_file",
            }
        ],
    }
    receipt = tmp_path / "receipt.json"
    return cfg, config, registry, receipt, payload


def _write(path: Path, payload: object) -> None:
    """Write one JSON fixture."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _patch_references(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the receipt validator consume the local fixture reference."""
    monkeypatch.setattr(
        staging,
        "iter_campaign_arm_checkpoint_references",
        lambda cfg: cfg.references,
    )


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (None, "generated_at_utc is missing"),
        ("not-a-timestamp", "generated_at_utc is invalid"),
        ("2026-08-21T12:00:00", "generated_at_utc must include a timezone"),
    ],
)
def test_checkpoint_receipt_rejects_invalid_timestamps(value: object, match: str) -> None:
    """Receipt timestamps must be present, parseable, and timezone-aware."""
    with pytest.raises(staging.CheckpointStagingReceiptError, match=match):
        staging._parse_generated_at(value)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "wrong", "schema must be"),
        ("status", "failed", "status is not ok"),
        ("mode", "best_effort", "enforced-staged mode"),
    ],
)
def test_checkpoint_receipt_rejects_invalid_header_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    match: str,
) -> None:
    """Schema, status, and staging-mode drift cannot pass admission."""
    cfg, config, registry, receipt, payload = _receipt_fixture(tmp_path)
    payload[field] = value
    _write(receipt, payload)
    _patch_references(monkeypatch)
    with pytest.raises(staging.CheckpointStagingReceiptError, match=match):
        staging.validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 21, 12, tzinfo=UTC),
        )


def test_checkpoint_receipt_rejects_future_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A receipt generated materially in the future is not trusted."""
    cfg, config, registry, receipt, payload = _receipt_fixture(tmp_path)
    payload["generated_at_utc"] = "2026-08-21T13:00:00Z"
    _write(receipt, payload)
    _patch_references(monkeypatch)
    with pytest.raises(staging.CheckpointStagingReceiptError, match="in the future"):
        staging.validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 21, 12, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("arms", "match"),
    [
        ([], "no covered arms"),
        (["not-an-object"], "arms must be JSON objects"),
    ],
)
def test_checkpoint_receipt_rejects_missing_or_malformed_arm_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arms: list[object],
    match: str,
) -> None:
    """A receipt must enumerate at least one JSON arm object."""
    cfg, config, registry, receipt, payload = _receipt_fixture(tmp_path)
    payload["arms"] = arms
    _write(receipt, payload)
    _patch_references(monkeypatch)
    with pytest.raises(staging.CheckpointStagingReceiptError, match=match):
        staging.validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda arm: arm.update(resolved_path="missing.zip"), "no longer resolves"),
        (lambda arm: arm.update(checkpoint_sha256="short"), "no valid SHA-256"),
    ],
)
def test_checkpoint_receipt_rejects_unmaterialized_arm_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    match: str,
) -> None:
    """Every receipt arm must still point to a checksum-verifiable file."""
    cfg, config, registry, receipt, payload = _receipt_fixture(tmp_path)
    mutation(payload["arms"][0])
    _write(receipt, payload)
    _patch_references(monkeypatch)
    with pytest.raises(staging.CheckpointStagingReceiptError, match=match):
        staging.validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("not-json", "unreadable"),
        (["not", "an", "object"], "must be a JSON object"),
    ],
)
def test_checkpoint_receipt_rejects_unreadable_or_non_mapping_payload(
    tmp_path: Path,
    content: object,
    match: str,
) -> None:
    """Receipt input must be a readable JSON object."""
    receipt = tmp_path / "receipt.json"
    if isinstance(content, str):
        receipt.write_text(content, encoding="utf-8")
    else:
        _write(receipt, content)
    with pytest.raises(staging.CheckpointStagingReceiptError, match=match):
        staging.validate_checkpoint_staging_receipt(
            SimpleNamespace(references=[]),
            receipt,
            campaign_config_path=tmp_path / "campaign.yaml",
        )


def test_checkpoint_receipt_rejects_missing_receipt(tmp_path: Path) -> None:
    """A missing receipt is a hard admission failure."""
    with pytest.raises(staging.CheckpointStagingReceiptError, match="not found"):
        staging.validate_checkpoint_staging_receipt(
            SimpleNamespace(references=[]),
            tmp_path / "missing.json",
            campaign_config_path=tmp_path / "campaign.yaml",
        )
